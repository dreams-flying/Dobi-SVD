# Matryoshka SVD Implementation Guide

## Overview

This implementation combines **SVD-LLM's whitened SVD decomposition** with **Matryoshka's per-token dynamic rank** to create an activation-aware, dynamically compressed LLM.

### Architecture Pipeline

```
Step 1: Whitened SVD (using SVD-LLM)
  Original Model -> Calibration Data -> Whitening -> SVD -> U, V matrices

Step 2: Matryoshka Training (this implementation)
  U, V matrices -> MatryoshkaSVDLayer -> Rank Predictor Training -> Dynamic Model
```

## Prerequisites

1. **SVD-LLM repository** (already cloned)
   ```bash
   cd /home/user/SVD-LLM
   ```

2. **Dependencies**
   ```bash
   pip install transformers==4.35.2 torch datasets accelerate
   ```

## Step-by-Step Guide

### Step 1: Apply SVD-LLM Whitening (Offline, ~2-4 hours for 7B model)

SVD-LLM will compute activation-aware U and V matrices.

```bash
cd /home/user/SVD-LLM

# Run SVD-LLM compression
python SVDLLM.py \
    --model meta-llama/Llama-2-7b-hf \
    --whitening_nsamples 256 \
    --dataset c4 \
    --ratio 0.5 \
    --save_dir ./svd_llm_output \
    --skip_lora  # We'll train Matryoshka instead
```

**What this does:**
- Loads calibration data (256 samples from C4)
- Computes input covariance matrices for each layer
- Applies whitening transformation: `W_white = W @ Σ_x^{-1/2}`
- Performs SVD on whitened weights
- Saves U and V projections

**Output:**
```
svd_llm_output/
├── model.pth              # Compressed model with U,V projections
├── profiling_mat.pth      # Whitening matrices (optional)
└── config.json
```

### Step 2: Extract U, V Matrices for Matryoshka

```bash
cd /home/user/Dobi-SVD

# Extract SVD factors from SVD-LLM output
python scripts/extract_svd_factors.py \
    --svdllm_model /home/user/SVD-LLM/svd_llm_output/model.pth \
    --output_dir ./svd_factors
```

This extracts U and V matrices from each layer into separate files.

### Step 3: Train Matryoshka (Per-token Dynamic Rank)

```bash
# Train rank predictor with multi-scale training
python train_matryoshka_from_svdllm.py \
    --model meta-llama/Llama-2-7b-hf \
    --svd_factors_dir ./svd_factors \
    --r_max 256 \
    --r_min 64 \
    --output_dir ./matryoshka_output \
    --dataset wikitext2 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-4 \
    --freeze_uv  # Only train rank predictor
```

**Training objectives:**
1. **Task loss**: Language modeling loss
2. **Rank regularization**: Encourages lower average rank
3. **Multi-scale training**: Samples different fixed ranks during training

### Step 4: Evaluation

```bash
# Evaluate at different compression levels
for rank in 64 128 192 256; do
    python evaluate_matryoshka.py \
        --checkpoint ./matryoshka_output/checkpoint-best \
        --target_rank $rank \
        --dataset wikitext2
done
```

## Key Advantages of This Approach

### 1. **Activation-Aware** (from SVD-LLM)
- Whitening transformation: `W_white = W @ Σ_x^{-1/2}`
- Ensures weight Frobenius norm = activation L2 error
- Better than pure weight SVD at low ranks

### 2. **Per-Token Dynamic** (from Matryoshka)
- Each token gets its own rank: `r_i = RankPredictor(x_i)`
- Simple tokens use low rank (fast)
- Complex tokens use high rank (accurate)
- Average computational cost: `O(r_avg)` where `r_avg < r_max`

### 3. **Nested Structure** (from Matryoshka)
- Single set of U, V supports all ranks `[r_min, r_max]`
- No need to retrain for different compression levels
- Smooth quality-speed tradeoff curve

## Theoretical Foundation

### Whitening Correctness

Standard SVD minimizes weight error:
```
min ||W - W_k||_F²
```

Whitened SVD minimizes activation error:
```
min E[||Wx - W_k x||²] where E[xx^T] ≠ I
```

By whitening with `Σ_x^{-1/2}`, we make these equivalent:
```
||W_white - W_white,k||_F² = E[||Wx - W_k x||²]
```

### Soft Gating

For predicted rank `r_i` and singular value index `k`:
```python
gate_k = σ((r_i - k) / τ)
z_gated[k] = z[k] * gate_k
```

This provides smooth transitions between ranks, enabling:
- Gradient flow during training
- Continuous rank values (not discrete)

## Configuration Details

### `r_max` Selection

Choose `r_max` based on SVD-LLM's compression ratio:

| SVD-LLM ratio | Effective rank | Recommended r_max |
|---------------|----------------|-------------------|
| 0.3           | ~153           | 128-192           |
| 0.5           | ~256           | 256-320           |
| 0.7           | ~358           | 320-384           |

### `r_min` Selection

Choose `r_min` for minimum acceptable quality:
- `r_min = 64`: Aggressive compression
- `r_min = 128`: Balanced
- `r_min = 192`: Conservative

### Multi-Scale Training

Training samples ranks from `[r_min, r_max]`:
```python
# Sample 3 ranks per batch
sampled_ranks = [r_min, r_max, random.randint(r_min, r_max)]

# Forward at each rank and average loss
loss = mean([forward(x, rank=r) for r in sampled_ranks])
```

## Expected Results

Based on theory and similar methods:

| Metric | r=64 | r=128 | r=256 |
|--------|------|-------|-------|
| PPL (vs baseline) | +30% | +10% | +2% |
| Speedup | 3.5x | 2.2x | 1.3x |
| Memory | -60% | -40% | -20% |

**Note**: Whitened SVD should give ~5-10% better PPL than pure weight SVD at low ranks.

## Troubleshooting

### Q: SVD-LLM fails with "ill-conditioned matrix"
**A**: Increase regularization in whitening:
```python
# In SVDLLM.py, line where Cholesky is computed
Sigma_x = Sigma_x + 1e-5 * torch.eye(Sigma_x.shape[0])
```

### Q: Rank predictor outputs constant values
**A**: Check:
1. Learning rate (try 1e-4 to 1e-3)
2. Rank regularization weight (try 0.001 to 0.01)
3. Gating temperature (try 0.05 to 0.2)

### Q: Training loss diverges
**A**:
1. Reduce learning rate
2. Add gradient clipping: `--max_grad_norm 1.0`
3. Check if U, V were loaded correctly

## Next Steps

1. **Quantization**: Combine with GPTQ/AWQ for further compression
2. **Knowledge Distillation**: Use original model as teacher
3. **Structured Pruning**: Combine with N:M sparsity

## References

1. SVD-LLM: https://github.com/AIoT-MLSys-Lab/SVD-LLM
2. Matryoshka Representation Learning: https://arxiv.org/abs/2205.13147
3. Dobi-SVD: Original codebase in this repo

## File Structure

```
Dobi-SVD/
├── modules/
│   ├── matryoshka_svd_layer.py      # Core Matryoshka implementation
│   └── stable_svd.py                # SVD utilities (if needed)
├── scripts/
│   └── extract_svd_factors.py       # Extract from SVD-LLM
├── train_matryoshka_from_svdllm.py  # Main training script
├── evaluate_matryoshka.py           # Evaluation script
└── MATRYOSHKA_SVD_GUIDE.md         # This file
```
