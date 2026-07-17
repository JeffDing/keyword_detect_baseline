# Keyword Detect Baseline (Ascend NPU)

基于 Whisper 预训练编码器 + 逐帧匹配 + 难负样本挖掘的关键词检测基线，针对 Ascend NPU 优化。

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

如需使用难负样本挖掘，创建 `similar_word_pairs.csv`：

```csv
id,similar_id
word1,word2
word3,word4
```

### 2. 训练

```bash
bash run_train.sh --subset 100 --bs 16 --hard_mining_every 100 --hard_top_k 1
```

### 3. 推理

```bash
bash run_infer.sh --ckpt checkpoints/best.pt --out submission.csv
```

## 参数说明

### train.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 10 | 训练轮数 |
| `--bs` | 128 | batch size |
| `--lr` | 1e-3 | 学习率 |
| `--subset` | 500000 | 训练子集大小 |
| `--hard_mining_every` | 5 | 每隔多少 epoch 做一次难负样本挖掘 |
| `--hard_top_k` | 5 | 每个正样本挖掘的难负样本数 |
| `--hard_weight` | 2.0 | 难负样本 loss 的权重 |
| `--out` | checkpoints/best.pt | 模型保存路径 |

### infer.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--ckpt` | checkpoints/best.pt | 模型路径 |
| `--out` | submission.csv | 输出 CSV 路径 |
| `--bs` | 256 | batch size |

## 模型架构

- **Whisper 预训练编码器**：提取逐帧音频特征（冻结，NPU 运行）
- **帧级投影头**：将 Whisper 特征映射到 embedding 空间（可训练，NPU 运行）
- **对称 max-mean 软对齐**：双向帧级匹配（NPU 运行）
- **难负样本挖掘**：在线挖掘相似音难负样本

## NPU 性能优化

推理流水线采用三阶段设计，最大化 NPU 利用率：

1. **批量数据读取**：`batch_read_pairs` 使用多线程从 RAM 中的 zip 读取 wav，numpy 层面 pad/trim 后一次性转 tensor
2. **NPU mel 计算**：`npu_batch_mel` 在 NPU 上批量计算 mel 频谱（torch.stft），比 CPU 快 ~135 倍
3. **NPU 推理**：AMP autocast + NPU 前向推理

### 推理速度

| 阶段 | 耗时（256样本/batch） |
|------|----------------------|
| 数据读取 | ~0.6s |
| NPU mel 计算 | ~0.1s |
| NPU 推理 | ~0.3s |
| **总计** | **~1.0s/batch** |

10 万样本推理约 7 分钟（~237 samples/s）。

### 关键技术点

- `torch.stft` 在 NPU 上走 AICPU fallback，但批量计算仍远快于 CPU
- NPU mel 与 CPU mel 数值差异 < 5e-5，不影响推理结果
- DataLoader `num_workers > 0` 在 NPU 环境下会崩溃（fork 与 NPU 运行时冲突），改用 `ThreadPoolExecutor`
- zip 文件预加载到 RAM，避免磁盘 I/O 瓶颈
- numpy 层面 pad/trim + `np.stack` + 一次性 `torch.from_numpy`，比逐条 `torch.from_numpy` + `torch.stack` 快 ~14 倍

## Whisper 模型类型

可在 `config.py` 的 `AudioConfig` 中修改 `whisper_model_name`，支持：

- `tiny.en` / `tiny`
- `base.en` / `base`
- `small.en` / `small`
- `medium.en` / `medium`
- `large-v2`

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
- `--subset` 建议大于 batch size，避免 `drop_last` 导致训练 step 为空
- 推理输出 `submission.csv` 格式：`id,posterior`
