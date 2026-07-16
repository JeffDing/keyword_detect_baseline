# Keyword Detect Baseline (Ascend NPU)

将原始 `baseline` 迁移为 Ascend NPU 可用版本，使用 Whisper 预训练编码器 + 逐帧匹配 + 难负样本挖掘。

## 环境初始化

训练/推理前确保已加载 CANN 环境：

```bash
source /usr/local/Ascend/cann-8.5.1/set_env.sh
```

或直接使用目录下脚本：

```bash
./run_train.sh ...
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
bash run_train.sh --subset 100 --bs 16 --hard_mining_every 100 --hard_top_k 1 --workers 0
```

> NPU 环境建议保持 `--workers 0`，避免多进程 te_fusion 初始化导致卡住。

### 3. 推理

```bash
python3 infer.py --ckpt checkpoints/best.pt --out submission.csv
```

## 参数说明

### train.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 10 | 训练轮数 |
| `--bs` | 128 | batch size |
| `--lr` | 1e-3 | 学习率 |
| `--subset` | 500000 | 训练子集大小 |
| `--workers` | 0 | DataLoader 进程数，NPU 环境建议保持 0 避免卡住 |
| `--hard_mining_every` | 5 | 每隔多少 epoch 做一次难负样本挖掘 |
| `--hard_top_k` | 5 | 每个正样本挖掘的难负样本数 |
| `--hard_weight` | 2.0 | 难负样本 loss 的权重 |
| `--persistent_workers` | false | 是否保持 DataLoader 工作进程常驻 |
| `--prefetch_factor` | 2 | 每个 worker 预取的样本批数 |
| `--out` | checkpoints/best.pt | 模型保存路径 |

### infer.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--ckpt` | checkpoints/best.pt | 模型路径 |
| `--out` | submission.csv | 输出 CSV 路径 |
| `--bs` | 256 | batch size |
| `--workers` | 8 | DataLoader 进程数 |

## Whisper 模型类型

可在 `config.py` 的 `AudioConfig` 中修改 `whisper_model_name`，支持：

- `tiny.en` / `tiny`
- `base.en` / `base`
- `small.en` / `small`
- `medium.en` / `medium`
- `large-v2`（最新大模型）

```python
@dataclass
class AudioConfig:
    ...
    whisper_model_name: str = "tiny"
```

- `.en` 表示该模型仅使用英文数据训练，适合纯英文任务
- 不加 `.en` 的是多语言模型
- `large-v2` 更大、更慢、更吃显存/内存

## 模型架构

- Whisper 预训练编码器：提取逐帧音频特征，运行在 NPU
- 帧级投影头：将 Whisper 特征映射到 embedding 空间，运行在 NPU
- 对称 max-mean 软对齐：双向帧级匹配，运行在 NPU
- 难负样本挖掘：在线挖掘相似音难负样本

## NPU 适配说明

- Whisper 编码器在 NPU 上运行
- 可训练部分（投影头、scale、bias）在 NPU 上运行
- 自动选择 `npu`，不可用时回退 `cpu`
- 默认初始化单卡环境变量，避免多进程 te_fusion 初始化问题
- DataLoader 默认 `--workers 0`，避免多进程 te_fusion 初始化导致卡住

## 相似词数据格式

`similar_word_pairs.csv` 格式：

```csv
id,similar_id
hi,haier
hello,hallo
```

用于训练时额外添加相似音负样本对。

## 注意事项

- Whisper 编码器在 NPU 上运行，减少 CPU->NPU 的数据拷贝开销
- 相似词对文件为可选，用于难负样本挖掘
- 当前已移除特征缓存机制，每次训练/推理都会实时计算 Whisper 特征
- 默认每 5 个 epoch 做一次难负样本挖掘，可根据显存/速度需求调整
- `--subset` 建议大于 batch size，避免 `drop_last` 导致训练 step 为空
- NPU 环境下建议保持 `--workers 0`，多进程 DataLoader 可能导致 te_fusion 初始化卡住
