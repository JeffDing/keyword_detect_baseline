# Keyword Detect Baseline (Ascend NPU)

将原始 `baseline` 迁移为 Ascend NPU 可用版本，并提供 mel 特征预缓存能力，以缓解 CPU 数据预处理瓶颈。

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

### 1. 预缓存 mel 特征（建议先执行）

```bash
python3 precache.py --subset all --save-cache
```

生成后默认缓存目录为 `baseline_ascend/cache`。

### 2. 训练

```bash
# 使用缓存
python3 train.py --cache-dir cache --workers 0

# 不使用缓存
python3 train.py --workers 0
```

### 3. 推理

```bash
python3 infer.py --cache-dir cache --ckpt checkpoints/best.pt --out submission.csv
```

## 参数说明

### train.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 10 | 训练轮数 |
| `--bs` | 128 | batch size |
| `--lr` | 1e-3 | 学习率 |
| `--subset` | 500000 | 训练子集大小 |
| `--workers` | 8 | DataLoader 进程数 |
| `--cache-dir` | cache | mel 缓存目录，留空则不用缓存 |
| `--out` | checkpoints/best.pt | 模型保存路径 |

### infer.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--ckpt` | checkpoints/best.pt | 模型路径 |
| `--out` | submission.csv | 输出 CSV 路径 |
| `--bs` | 256 | batch size |
| `--workers` | 8 | DataLoader 进程数 |
| `--cache-dir` | cache | mel 缓存目录，留空则不用缓存 |

### precache.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--subset` | all | train/dev_seen/dev_unseen/eval_seen/eval_unseen/all |
| `--save-cache` | False | 生成缓存文件 |
| `--cache-dir` | cache | 缓存目录 |

## 缓存机制

- 缓存文件为 `.npy`，键为 `zip路径:wav文件名` 的 SHA256
- 缓存缺失时自动回退原始读取/特征计算路径
- 缓存仅用于加速，不影响训练逻辑与结果

## NPU 适配说明

- 自动选择 `npu`，不可用时回退 `cpu`
- 默认初始化单卡环境变量，避免多进程 te_fusion 初始化问题
- 模型结构与原始 baseline 保持一致，仅 transport 层适配 NPU
