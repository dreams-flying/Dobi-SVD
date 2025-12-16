# Matryoshka SVD 评测指南

## 概述

`evaluate_matryoshka_svdllm.py` 用于评测从 SVD-LLM 训练的 Matryoshka SVD 模型。

## 功能

1. **困惑度 (Perplexity) 评测**
   - 支持数据集: wikitext2, c4, ptb
   - 支持自适应 rank 或固定 rank

2. **多 Rank 评测**
   - 自动测试 r_min, r_mid, r_max, adaptive
   - 对比不同压缩率的性能

3. **压缩统计**
   - 显示每层的压缩比
   - 计算总体参数削减

## 基础用法

### 1. 单 Rank 评测（自适应）

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_matryoshka_svdllm.py \
    --checkpoint ./matryoshka_output/final \
    --eval_dataset wikitext2 \
    --eval_rank adaptive \
    --n_eval_samples 256
```

**输出示例:**
```
Compression Statistics
================================================================================
Overall Compression
  Total layers:                    224
  Total original:        3,670,016,000
  Total compressed:        918,528,000
  Overall ratio:                25.03%
  Parameter reduction:          74.97%
  Memory saved:              2751.5M params

Evaluation Results
================================================================================
Dataset:     wikitext2
Rank:        adaptive
Perplexity:  12.3456
Avg rank:    128.45
```

### 2. 固定 Rank 评测

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_matryoshka_svdllm.py \
    --checkpoint ./matryoshka_output/final \
    --eval_dataset wikitext2 \
    --eval_rank 64 \
    --n_eval_samples 256
```

### 3. 多 Rank 评测（推荐）

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_matryoshka_svdllm.py \
    --checkpoint ./matryoshka_output/final \
    --eval_dataset wikitext2 \
    --multi_rank_eval \
    --n_eval_samples 256 \
    --save_results
```

**输出示例:**
```
Multi-Rank Evaluation Summary
================================================================================
Dataset: wikitext2

adaptive:
  Perplexity:    12.3456
  Avg rank:     128.45

r_max=256:
  Perplexity:    11.8901
  Avg rank:     256.00

r_mid=128:
  Perplexity:    12.5678
  Avg rank:     128.00

r_min=64:
  Perplexity:    14.2345
  Avg rank:      64.00
```

## 参数说明

### 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--checkpoint` | (必需) | 模型检查点路径 (HuggingFace 格式目录) |
| `--eval_dataset` | wikitext2 | 评测数据集: wikitext2, c4, ptb |
| `--eval_rank` | adaptive | 评测 rank: "adaptive" 或整数 (如 64, 128) |
| `--multi_rank_eval` | False | 启用多 rank 评测 |

### 数据集参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--n_eval_samples` | 256 | 评测样本数 |
| `--seq_len` | 2048 | 序列长度 |
| `--path_head_folder` | ./ | 数据缓存路径 |

### 系统参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--gpu` | 0 | GPU 设备 ID |
| `--seed` | 0 | 随机种子 |
| `--save_results` | False | 保存结果到 JSON |

## 评测流程

### 1. 训练模型

```bash
CUDA_VISIBLE_DEVICES=3 python train_matryoshka_from_svdllm.py \
    --model meta-llama/Llama-2-7b-hf \
    --svdllm_model /path/to/svdllm/model.pt \
    --r_max 256 --r_min 64 \
    --output_dir ./matryoshka_output \
    --num_train_epochs 1
```

### 2. 评测模型

```bash
# 快速评测 (adaptive)
CUDA_VISIBLE_DEVICES=0 python evaluate_matryoshka_svdllm.py \
    --checkpoint ./matryoshka_output/final \
    --eval_dataset wikitext2 \
    --eval_rank adaptive

# 完整评测 (多 rank)
CUDA_VISIBLE_DEVICES=0 python evaluate_matryoshka_svdllm.py \
    --checkpoint ./matryoshka_output/final \
    --eval_dataset wikitext2 \
    --multi_rank_eval \
    --save_results
```

### 3. 分析结果

```bash
# 查看保存的结果
cat ./matryoshka_output/eval_wikitext2_multi_rank.json
```

## 评测指标

### Perplexity (困惑度)

**含义**: 模型预测下一个词的平均不确定性
- 越低越好
- PPL = exp(平均交叉熵损失)

**参考值** (Llama-2-7B on wikitext2):
- 原始模型: ~10-11
- SVD-LLM (rank=256): ~12-13
- Matryoshka (adaptive): ~12-14
- Matryoshka (r=64): ~15-18

### Average Rank (平均 Rank)

**含义**: 自适应模式下，模型实际使用的平均 rank
- 反映模型的动态压缩行为
- 介于 r_min 和 r_max 之间
- 越低说明压缩越激进

**示例分析**:
```
r_max=256, r_min=64
avg_rank=128.45
→ 模型平均使用约 50% 的最大 rank
→ 在 PPL 和压缩之间找到了平衡
```

## 典型评测场景

### 场景 1: 快速验证训练效果

```bash
# 只需要知道模型大致性能
python evaluate_matryoshka_svdllm.py \
    --checkpoint ./output/checkpoint-1000 \
    --eval_rank adaptive \
    --n_eval_samples 128  # 少量样本，快速评测
```

### 场景 2: 论文实验（完整评测）

```bash
# 需要完整的多 rank 对比数据
for dataset in wikitext2 c4 ptb; do
    python evaluate_matryoshka_svdllm.py \
        --checkpoint ./matryoshka_output/final \
        --eval_dataset $dataset \
        --multi_rank_eval \
        --n_eval_samples 512 \
        --save_results
done
```

### 场景 3: 部署前性能测试

```bash
# 测试目标 rank (如 rank=64)
python evaluate_matryoshka_svdllm.py \
    --checkpoint ./matryoshka_output/final \
    --eval_rank 64 \
    --n_eval_samples 256

# 对比自适应模式
python evaluate_matryoshka_svdllm.py \
    --checkpoint ./matryoshka_output/final \
    --eval_rank adaptive \
    --n_eval_samples 256
```

## 输出文件

### 单 Rank 评测

文件: `eval_{dataset}_rank{rank}.json`

```json
{
  "dataset": "wikitext2",
  "eval_rank": "adaptive",
  "perplexity": 12.3456,
  "avg_rank": 128.45
}
```

### 多 Rank 评测

文件: `eval_{dataset}_multi_rank.json`

```json
{
  "adaptive": {
    "perplexity": 12.3456,
    "avg_rank": 128.45,
    "target_rank": "adaptive"
  },
  "r_max=256": {
    "perplexity": 11.8901,
    "avg_rank": 256.0,
    "target_rank": 256
  },
  "r_mid=128": {
    "perplexity": 12.5678,
    "avg_rank": 128.0,
    "target_rank": 128
  },
  "r_min=64": {
    "perplexity": 14.2345,
    "avg_rank": 64.0,
    "target_rank": 64
  }
}
```

## 性能优化

### 内存不足

```bash
# 减少评测样本数
--n_eval_samples 64

# 减少序列长度
--seq_len 512
```

### 评测太慢

```bash
# 使用少量样本快速测试
--n_eval_samples 32

# 只评测关键 rank
--eval_rank adaptive  # 不使用 --multi_rank_eval
```

## 常见问题

### Q1: 提示 "No MatryoshkaSVDLayer found"

**原因**: 加载的模型不是 Matryoshka SVD 模型

**解决**:
1. 检查 checkpoint 路径是否正确
2. 确认模型是用 `train_matryoshka_from_svdllm.py` 训练的
3. 查看模型架构: `print(model)`

### Q2: Perplexity 是 inf

**原因**: 模型输出了 NaN 或极大值

**解决**:
1. 检查模型是否正确加载权重
2. 尝试使用固定 rank 评测
3. 检查训练是否收敛

### Q3: 不同 rank 的 PPL 差异很小

**原因**: 可能的情况:
1. Rank predictor 训练不充分
2. 数据集太简单，不需要高 rank
3. r_min 设置得太高

**分析**:
```bash
# 测试极端情况
--eval_rank 16  # 很低的 rank
--eval_rank 512 # 很高的 rank
```

## 对比基线

### 与 SVD-LLM 对比

```bash
# 1. 评测 SVD-LLM (固定 rank)
python evaluate_svdllm.py --model svdllm_output/model.pt

# 2. 评测 Matryoshka (固定相同 rank)
python evaluate_matryoshka_svdllm.py \
    --checkpoint matryoshka_output/final \
    --eval_rank 256

# 3. 评测 Matryoshka (自适应)
python evaluate_matryoshka_svdllm.py \
    --checkpoint matryoshka_output/final \
    --eval_rank adaptive
```

**预期结果**:
- SVD-LLM (r=256): PPL ≈ 12.5
- Matryoshka (r=256): PPL ≈ 12.5-12.8 (略高，因为训练了 predictor)
- Matryoshka (adaptive): PPL ≈ 13.0-13.5，但 avg_rank ≈ 128 (更高效)

## 总结

**推荐评测流程**:

1. **训练后立即测试**: 用 `--eval_rank adaptive --n_eval_samples 128` 快速验证
2. **完整评测**: 用 `--multi_rank_eval --save_results` 获取完整数据
3. **对比分析**: 比较不同 rank 的 PPL vs 压缩率权衡

**重点指标**:
- **Perplexity**: 模型质量
- **Average Rank**: 实际压缩率
- **Compression Ratio**: 参数削减百分比

**论文实验建议**:
- 至少在 3 个数据集上评测 (wikitext2, c4, ptb)
- 包含多 rank 对比 (r_min, r_mid, r_max, adaptive)
- 报告压缩率和性能的 trade-off 曲线
