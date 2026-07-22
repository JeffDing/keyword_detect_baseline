# Keyword Detect Baseline (Ascend NPU)

基于 Whisper 预训练编码器 + 逐帧匹配 + 难负样本挖掘的关键词检测基线，针对 Ascend NPU 优化。支持张量并行（TP）、数据并行（DP）、TTA、多检查点集成和Platt校准。

## 环境要求

- Ascend 910B2 + CANN 8.5.1
- PyTorch + torch_npu
- whisper, soundfile, torchaudio, scikit-learn

训练/推理前确保已加载 CANN 环境：

```bash
source /usr/local/Ascend/cann-8.5.1/set_env.sh
```

## 快速开始

### 1. 准备相似词对（可选）

如需使用相似音负样本增强，创建 `similar_word_pairs.csv`：

```csv
id,similar_id
word1,word2
word3,word4
```

### 2. 训练

```bash
python train.py --nproc_per_node 4 --tp 2 --dp 2 --epochs 5 --bs 128 --lr 1e-3
```

### 3. 推理

```bash
python infer.py --nproc_per_node 4 --tp 2 --dp 2 --ckpt checkpoints/best.pt --out submission.csv
```

带TTA和校准的推理：

```bash
python infer.py --nproc_per_node 4 --tp 2 --dp 2 --ckpt checkpoints/best.pt --tta 3 --calibrate --out submission.csv
```

多检查点集成推理：

```bash
python infer.py --nproc_per_node 4 --tp 2 --dp 2 --ensemble --calibrate --out submission.csv
```

## 参数说明

### train.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 5 | 训练轮数 |
| `--bs` | 128 | batch size |
| `--lr` | 1e-3 | 学习率 |
| `--subset` | 500000 | 训练子集大小，越小分数通常越低 |
| `--embed_dim` | 256 | 帧级投影维度 |
| `--use_mlp` | True | 使用两层MLP投影头（Column→GELU→Row） |
| `--no_mlp` | False | 禁用MLP，改用单层线性投影 |
| `--warmup_epochs` | 5 | 学习率warmup轮数 |
| `--hard_mining_every` | 5 | 每隔多少 epoch 做一次难负样本挖掘 |
| `--hard_top_k` | 5 | 每个正样本挖掘的难负样本数 |
| `--hard_weight` | 2.0 | 难负样本 loss 的权重 |
| `--spec_augment` | True | 启用SpecAugment数据增强 |
| `--no_spec_augment` | False | 禁用SpecAugment |
| `--save_top_k` | 3 | 保存top-k个最佳checkpoint |
| `--out` | checkpoints/best.pt | 最佳模型保存路径 |
| `--tp` | 1 | 张量并行度 |
| `--dp` | 1 | 数据并行度 |
| `--nproc_per_node` | 1 | 使用的NPU卡数（须等于 tp × dp） |

### infer.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--ckpt` | checkpoints/best.pt | 模型路径 |
| `--out` | submission.csv | 输出 CSV 路径 |
| `--bs` | 256 | batch size |
| `--tta` | 0 | TTA 次数（0=不使用，推荐3-5） |
| `--ensemble` | False | 使用 top1~top3 检查点集成 |
| `--calibrate` | False | 使用 dev 集做 IsotonicRegression 校准 |
| `--tp` | 1 | 张量并行度 |
| `--dp` | 1 | 数据并行度 |
| `--nproc_per_node` | 1 | 使用的NPU卡数（须等于 tp × dp） |

## 模型架构

- **Whisper 预训练编码器**：large-v2，提取逐帧1280维音频特征（冻结，NPU运行）
- **帧级投影头**：将 Whisper 特征映射到256维embedding空间（可训练，支持TP）
  - use_mlp=True：`Linear(1280→256) → GELU → Linear(256→256)` + L2归一化
  - use_mlp=False：`Linear(1280→256)` + L2归一化
- **可学习温度**：控制软对齐的锐度
- **对称 max-mean 软对齐**：双向帧级匹配
- **可学习scale + bias**：将匹配分数映射到二分类logit

## 训练策略

- **损失函数**：BCEWithLogitsLoss（pos_weight=3.0，应对正负样本不平衡）
- **优化器**：Adam（lr=1e-3）
- **学习率调度**：CosineWarmupScheduler（线性warmup + 余弦退火）
- **SpecAugment**：频域掩蔽（2带）+ 时域掩蔽（10帧）
- **难负样本挖掘**：每5个epoch基于全局相似度挖掘top-5难负样本，以2.0权重加权
- **相似词增强**：加载similar_word_pairs.csv作为额外负样本
- **Checkpoint保存**：保存最佳 + top-3 checkpoint

## NPU 性能优化

推理流水线采用三阶段设计，最大化 NPU 利用率：

1. **批量数据读取**：`batch_read_pairs` 使用多线程从 RAM 中的 zip 读取 wav，numpy 层面 pad/trim 后一次性转 tensor
2. **NPU mel 计算**：`npu_batch_mel` 在 NPU 上批量计算 mel 频谱（torch.stft），比 CPU 快 ~135 倍
3. **NPU 推理**：AMP autocast + NPU 前向推理

### 关键技术点

- `torch.stft` 在 NPU 上走 AICPU fallback，但批量计算仍远快于 CPU
- NPU mel 与 CPU mel 数值差异 < 5e-5，不影响推理结果
- DataLoader `num_workers > 0` 在 NPU 环境下会崩溃（fork 与 NPU 运行时冲突），改用 `ThreadPoolExecutor`
- zip 文件预加载到 RAM，避免磁盘 I/O 瓶颈
- numpy 层面 pad/trim + `np.stack` + 一次性 `torch.from_numpy`，比逐条 `torch.from_numpy` + `torch.stack` 快 ~14 倍
- NPU预热消除首次推理的算子编译延迟

## Whisper 模型类型

可在 `config.py` 的 `AudioConfig` 中修改 `whisper_model_name`，支持：

- `tiny.en` / `tiny`
- `base.en` / `base`
- `small.en` / `small`
- `medium.en` / `medium`
- `large-v2`（默认）

## 相似词数据格式

`similar_word_pairs.csv` 格式：

```csv
id,similar_id
hi,haier
hello,hallo
```

用于训练时额外添加相似音负样本对。

## 注意事项

- 相似词对文件为可选，用于难负样本挖掘
- `--subset` 建议大于 batch size，避免训练 step 为空
- `--nproc_per_node` 必须等于 `--tp × --dp`
- `--embed_dim` 必须能被 `--tp` 整除
- 推理输出 `submission.csv` 格式：`id,posterior`
- TTA仅在单模型模式下生效，ensemble模式不支持TTA
