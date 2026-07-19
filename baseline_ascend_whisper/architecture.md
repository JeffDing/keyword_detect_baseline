# 基线架构与实现分析文档

## 一、项目结构

```
baseline_ascend_whisper/
├── config.py       # 全局配置（音频参数、训练参数、路径）
├── data.py         # 数据加载、预处理（ZIP缓存、WAV读取、Mel频谱计算）
├── model.py        # 模型定义（WhisperBackbone + FrameMatcher + SiameseKWS）
├── train.py        # 训练入口（NPU初始化、训练循环、难负样本挖掘、评估）
├── infer.py        # 推理入口（模型加载、NPU预热、批量预测、CSV输出）
├── run_train.sh    # 训练启动脚本（CANN环境变量）
├── run_infer.sh    # 推理启动脚本（CANN环境变量+可见设备）
└── docs/           # 开发文档
```

## 二、训练机制

### 2.1 框架与硬件

- **框架**: PyTorch + torch_npu（华为昇腾NPU适配层）
- **硬件**: Ascend 910B2 + CANN 8.5.1
- **设备选择**: 自动检测NPU可用性，不可用则回退CPU

### 2.2 模型架构 — SiameseKWS（孪生网络关键词检测）

```
输入: enroll_mel, query_mel
  │                │
  ▼                ▼
WhisperBackbone  WhisperBackbone  (共享权重，冻结)
  │                │
  ▼                ▼
FrameMatcher (可训练投影头)
  │  - Linear(1280 → 64) + L2归一化
  │  - 对称max-mean软对齐匹配
  ▼
scale * score + bias  →  二分类logit
```

- **WhisperBackbone**: 加载Whisper large-v2预训练编码器，**冻结全部参数**（~637M），仅做特征提取
- **FrameMatcher**: 可训练投影头，将1280维Whisper特征投影到64维，L2归一化后做对称max-mean软对齐
- **SiameseKWS**: 组合Backbone + Matcher，添加可学习scale(初始8.0)和bias(初始0.0)

### 2.3 损失函数与优化器

- **损失**: BCEWithLogitsLoss，pos_weight=3.0（应对正负样本不平衡）
- **优化器**: Adam，lr=1e-3，无学习率调度

### 2.4 训练流程

1. 预加载3个ZIP文件到RAM
2. 每个epoch：
   - 随机打乱训练数据
   - 多线程读取WAV → NPU计算Mel频谱 → AMP前向传播 → 计算loss
   - 每5个epoch触发**难负样本挖掘**：基于全局相似度找top-k最相似负样本，以2.0权重加权累加loss
   - 反向传播 + 参数更新
3. 每个epoch结束后在dev_seen和dev_unseen上评估AUC，取均值
4. 当mean AUC超过历史最佳时保存checkpoint

### 2.5 难负样本挖掘

- 触发条件：每5个epoch
- 流程：获取当前batch特征 → 计算全局相似度矩阵 → 对每个正样本找top-k最相似负样本 → 构建难负样本对 → 以hard_weight=2.0加权累加到总loss
- 目的：让模型更关注"易混淆"负样本，提升判别能力

## 三、推理机制

### 3.1 推理流程

1. NPU环境初始化 + TBE模块检查
2. 加载checkpoint → 构建模型 → 加载权重 → eval模式
3. **NPU预热**：用随机数据执行一次完整前向传播，触发算子编译缓存
4. 批量预测（bs=256）：多线程读取WAV → NPU Mel频谱 → AMP推理 → sigmoid转概率
5. 输出CSV：`id,posterior`，包含seen和unseen两个评估集结果

### 3.2 NPU预热

- 使用随机数据执行dummy forward
- 触发NPU算子JIT编译，后续推理无需重复编译
- `torch.npu.synchronize()` 确保编译完成

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
- 双层缓存：原始字节缓存 + 按进程ID+路径索引的ZipFile对象缓存

### 4.4 多线程数据读取替代DataLoader

- 使用 `ThreadPoolExecutor(max_workers=8)` 替代DataLoader
- 原因：NPU环境下多进程fork与NPU运行时冲突，num_workers>0会崩溃

### 4.5 Numpy层面批量处理

- numpy层面完成pad/trim → np.stack一次性堆叠 → 单次torch.from_numpy转换
- 比逐条转换快约14倍

### 4.6 Whisper编码器冻结

- 冻结~637M参数，仅训练投影头（64维）+ scale + bias
- 无需存储梯度，大幅减少显存和计算量

### 4.7 其他优化

- **NPU预热**：消除首次推理的算子编译延迟
- **线程数限制**：OPENBLAS/OMP/MKL线程数限制为4，避免与NPU争抢资源
- **ASCEND日志级别**：设为3，减少I/O开销

## 五、关键代码位置索引

| 功能 | 文件 | 行号 |
|------|------|------|
| 音频/训练配置 | config.py | 10-35 |
| ZIP RAM缓存 | data.py | 18-37 |
| 多线程批量读取 | data.py | 106-132 |
| WhisperBackbone | model.py | 9-31 |
| FrameMatcher | model.py | 34-58 |
| SiameseKWS | model.py | 61-81 |
| NPU Mel频谱(训练) | train.py | 39-63 |
| 难负样本挖掘 | train.py | 93-120 |
| 训练主循环 | train.py | 123-210 |
| AMP训练 | train.py | 179 |
| NPU预热 | infer.py | 112-123 |
| 批量预测 | infer.py | 56-100 |
