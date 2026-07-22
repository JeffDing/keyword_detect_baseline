# 基线架构与实现分析文档

## 一、项目结构

```
baseline/
├── config.py            # 全局配置（音频参数、训练参数、路径）
├── data.py              # 数据加载、预处理（ZIP缓存、WAV读取、Mel频谱计算）
├── model.py             # 模型定义（WhisperBackbone + FrameMatcher + SiameseKWS，支持TP）
├── train.py             # 训练入口（NPU初始化、训练循环、难负样本挖掘、SpecAugment、评估）
├── infer.py             # 推理入口（模型加载、NPU预热、TTA、多检查点集成、Platt校准、CSV输出）
├── distributed_utils.py # 分布式工具（TP/DP通信组、梯度all-reduce、状态字典gather/shard）
└── architecture.md      # 本文档
```

## 二、训练机制

### 2.1 框架与硬件

- **框架**: PyTorch + torch_npu（华为昇腾NPU适配层）
- **硬件**: Ascend 910B2 + CANN 8.5.1
- **设备选择**: 自动检测NPU可用性，不可用则回退CPU
- **分布式**: 支持张量并行（TP）和数据并行（DP），通过 `--tp` 和 `--dp` 参数配置

### 2.2 模型架构 — SiameseKWS（孪生网络关键词检测）

```
输入: enroll_mel, query_mel
  │                │
  ▼                ▼
WhisperBackbone  WhisperBackbone  (共享权重，冻结)
  │                │
  ▼                ▼
FrameMatcher (可训练投影头)
  │  use_mlp=True (默认):
  │    ColumnParallelLinear(1280 → 256) → GELU → RowParallelLinear(256 → 256)
  │  use_mlp=False:
  │    ColumnParallelLinear(1280 → 256)
  │  L2归一化 + 可学习温度
  │  对称max-mean软对齐匹配
  ▼
scale * score + bias  →  二分类logit
```

- **WhisperBackbone**: 加载Whisper large-v2预训练编码器，**冻结全部参数**，仅做特征提取。输出维度为 `n_audio_state=1280`
- **FrameMatcher**: 可训练投影头，将1280维Whisper特征投影到256维（`embed_dim`可配置），L2归一化后做对称max-mean软对齐。包含可学习温度参数 `log_temp`（初始0.0，即温度=1.0）
- **SiameseKWS**: 组合Backbone + Matcher，添加可学习scale(初始8.0)和bias(初始0.0)

### 2.3 张量并行（TP）策略

FrameMatcher中的投影层支持张量并行：

- **use_mlp=True（默认）**: `ColumnParallelLinear → GELU → RowParallelLinear`
  - 列并行将输出维度按列切分到各TP rank（每个rank持有 `embed_dim/tp` 列）
  - 行并行在输出维度上all-reduce恢复完整维度
  - 中间GELU无需通信
- **use_mlp=False**: `ColumnParallelLinear → all_gather → normalize`
  - 列并行输出 `(B, T, embed_dim/tp)`，需all_gather恢复后归一化

### 2.4 损失函数与优化器

- **损失**: BCEWithLogitsLoss，pos_weight=3.0（应对正负样本不平衡）
- **优化器**: Adam，lr=1e-3
- **学习率调度**: CosineWarmupScheduler
  - 前 `warmup_epochs`（默认5）个epoch线性warmup
  - 之后余弦退火衰减至0
  - 按step级别调度，非epoch级别

### 2.5 训练流程

1. 预加载3个ZIP文件到RAM（train、dev_seen、dev_unseen）
2. 加载训练对 + 相似词负样本对
3. 按 `train_subset` 随机采样训练子集
4. 每个epoch：
   - 随机打乱训练数据（DP模式下每个rank取 `train_pairs[dp_rank::dp]` 子集）
   - 多线程读取WAV → NPU计算Mel频谱 → SpecAugment（可选） → AMP前向传播 → 计算loss
   - 每 `hard_mining_every`（默认5）个epoch触发**难负样本挖掘**：基于全局相似度找top-k最相似负样本，以 `hard_weight`（默认2.0）加权累加loss
   - 反向传播 → DP梯度all-reduce（若dp>1） → 参数更新 → 学习率调度
5. 每个epoch结束后在dev_seen和dev_unseen上评估AUC，取均值
6. 保存最佳checkpoint到 `best.pt`，同时维护top-k（默认3）checkpoint

### 2.6 难负样本挖掘

- 触发条件：每 `hard_mining_every`（默认5）个epoch
- 流程：获取当前batch的Whisper特征 → 计算全局相似度矩阵（均值embedding的点积） → 对每个正样本找top-k最相似负样本 → 构建难负样本对 → 以 `hard_weight` 加权累加到总loss
- 目的：让模型更关注"易混淆"负样本，提升判别能力

### 2.7 SpecAugment

- 默认开启（`spec_augment=True`）
- 频域掩蔽：`freq_mask=2`，随机遮蔽最多2条频率带
- 时域掩蔽：`time_mask=10`，随机遮蔽最多10帧时间步
- 仅在训练模式且 `spec_augment=True` 时应用

### 2.8 Checkpoint保存策略

- **best.pt**: 当mean AUC超过历史最佳时保存
- **top1.pt ~ top3.pt**: 维护top-3 checkpoint列表，按AUC降序排列
- checkpoint内容：`model`（完整state_dict）、`embed_dim`、`whisper_model`、`use_mlp`、`auc`、`epoch`
- TP模式下保存前先gather完整state_dict，确保checkpoint不含分片信息

## 三、推理机制

### 3.1 推理流程

1. NPU环境初始化 + TBE模块检查
2. 加载checkpoint → 构建模型 → 加载权重（TP模式下shard state_dict） → eval模式
3. **NPU预热**：用随机数据执行一次完整前向传播，触发算子编译缓存
4. 可选：Platt校准（使用dev集拟合IsotonicRegression）
5. 批量预测（bs=256）：多线程读取WAV → NPU Mel频谱 → AMP推理 → sigmoid转概率
6. 可选：TTA增强 / 多检查点集成
7. 输出CSV：`id,posterior`，包含seen和unseen两个评估集结果

### 3.2 NPU预热

- 使用随机数据（2条480000采样点）执行dummy forward
- 触发NPU算子JIT编译，后续推理无需重复编译
- `torch.npu.synchronize()` 确保编译完成

### 3.3 TTA（Test-Time Augmentation）

- 通过 `--tta N` 启用，推荐3-5次
- 对每条Mel频谱做N次随机掩蔽增强（频域1-3带 + 时域5-30帧）
- 对所有增强组合（原始 + N次enroll增强 × N次query增强）的logit取均值
- 注意：TTA仅在单模型模式下生效，ensemble模式不支持TTA

### 3.4 多检查点集成

- 通过 `--ensemble` 启用
- 加载 `top1.pt ~ top3.pt`（存在的），对logit取均值
- 每个模型独立推理，结果累加后除以模型数量

### 3.5 Platt校准

- 通过 `--calibrate` 启用
- 使用dev_seen + dev_unseen的预测概率和标签拟合IsotonicRegression
- 校准后概率范围裁剪到[0, 1]
- 可与单模型或ensemble模式配合使用

### 3.6 分布式推理

- 支持TP和DP分布式推理
- DP模式：将数据按rank分片，各rank独立推理后通过pickle文件收集结果
- TP模式：模型参数按rank分片加载，推理时通过集合通信协同

## 四、性能优化

### 4.1 NPU上的Mel频谱计算（~135倍加速）

- 将Mel频谱计算从CPU迁移到NPU
- 大batch拆分为sub_bs=32的子batch，避免显存溢出
- NPU与CPU计算差异 < 5e-5，不影响结果

### 4.2 AMP混合精度

- 训练和推理均使用 `torch.npu.amp.autocast(enabled=True)`
- 自动将合适算子转为FP16，减少显存和计算量
- 推理输出时显式转回FP32

### 4.3 ZIP文件RAM缓存

- 一次性将整个ZIP读入内存，后续读取全在RAM中完成
- 双层缓存：原始字节缓存（`_ZIP_RAM_CACHE`）+ 按进程ID+路径索引的ZipFile对象缓存（`_ZIP_CACHE`）

### 4.4 多线程数据读取替代DataLoader

- 使用 `ThreadPoolExecutor(max_workers=8)` 替代DataLoader
- 原因：NPU环境下多进程fork与NPU运行时冲突，num_workers>0会崩溃

### 4.5 Numpy层面批量处理

- numpy层面完成pad/trim → np.stack一次性堆叠 → 单次torch.from_numpy转换
- 比逐条转换快约14倍

### 4.6 Whisper编码器冻结

- 冻结Whisper编码器全部参数，仅训练投影头 + 温度 + scale + bias
- 无需存储编码器梯度，大幅减少显存和计算量

### 4.7 其他优化

- **NPU预热**：消除首次推理的算子编译延迟
- **线程数限制**：OPENBLAS/OMP/MKL线程数限制为4，避免与NPU争抢资源
- **ASCEND日志级别**：设为3，减少I/O开销

## 五、关键代码位置索引

| 功能 | 文件 | 行号 |
|------|------|------|
| 音频配置 | config.py | 10-16 |
| 训练配置 | config.py | 19-41 |
| 路径配置 | config.py | 44-73 |
| ZIP RAM缓存 | data.py | 18-37 |
| WAV读取与重采样 | data.py | 40-49 |
| 多线程批量读取 | data.py | 106-132 |
| WhisperBackbone | model.py | 9-31 |
| ColumnParallelLinear | model.py | 34-65 |
| RowParallelLinear | model.py | 68-108 |
| FrameMatcher | model.py | 111-167 |
| SiameseKWS | model.py | 169-193 |
| SpecAugment | train.py | 56-68 |
| NPU Mel频谱 | train.py | 71-95 |
| 评估函数 | train.py | 98-119 |
| 难负样本挖掘 | train.py | 122-149 |
| CosineWarmupScheduler | train.py | 152-167 |
| 训练worker主循环 | train.py | 170-328 |
| TTA增强 | infer.py | 72-84 |
| 模型加载 | infer.py | 87-106 |
| 单模型预测 | infer.py | 109-204 |
| Platt校准 | infer.py | 207-251 |
| NPU预热 | infer.py | 254-265 |
| 推理worker主循环 | infer.py | 268-413 |
| 分布式初始化 | distributed_utils.py | 15-26 |
| TP/DP通信组创建 | distributed_utils.py | 29-49 |
| 状态字典gather | distributed_utils.py | 76-101 |
| 状态字典shard | distributed_utils.py | 104-128 |
| DP梯度all-reduce | distributed_utils.py | 131-137 |
