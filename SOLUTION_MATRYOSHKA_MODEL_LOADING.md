# Solution: Matryoshka SVD Model Loading Issue

## Problem Summary

**Issue**: When evaluating a trained Matryoshka SVD model, the model had NO `MatryoshkaSVDLayer` instances, resulting in:

```bash
Set 0 layers to ADAPTIVE rank prediction  # ← Should be 224 layers!
Perplexity: 189946.7656  # ← Astronomical PPL = broken model
```

**Root Cause**: HuggingFace's `save_pretrained()` only saves:
- `config.json` (defines standard Llama architecture with `Linear` layers)
- Weight tensors (`.safetensors` files)

When loading with `from_pretrained()`, it reconstructs the model from `config.json`, creating standard `Linear` layers instead of `MatryoshkaSVDLayer`. The custom layer structure is completely lost.

## Solution Implemented

Created a custom save/load mechanism that preserves `MatryoshkaSVDLayer` structure:

### 1. New File: `matryoshka_model_utils.py`

Provides three key functions:

#### `save_matryoshka_model(model, tokenizer, output_dir)`
- Saves model weights (standard HuggingFace format)
- Saves tokenizer
- **Saves `matryoshka_metadata.json`** with layer reconstruction info

#### `load_matryoshka_model(checkpoint_path, base_model, device)`
- Loads checkpoint state dict (preserving all MatryoshkaSVDLayer weights)
- Loads base model (creates standard Linear layers)
- **Reconstructs `MatryoshkaSVDLayer` from metadata**
- Loads weights into reconstructed layers
- Returns fully functional Matryoshka model

#### `verify_matryoshka_structure(checkpoint_path)`
- Checks if a checkpoint has Matryoshka metadata
- Returns layer count and configuration

### 2. Updated Files

#### `train_matryoshka_from_svdllm.py`
**Before**:
```python
trainer.save_model(final_output_dir)
tokenizer.save_pretrained(final_output_dir)
```

**After**:
```python
from matryoshka_model_utils import save_matryoshka_model

save_matryoshka_model(
    model=trainer.model,
    tokenizer=tokenizer,
    output_dir=final_output_dir,
    safe_serialization=True
)
```

#### `evaluate_matryoshka_svdllm.py`
**Before**:
```python
model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
# MatryoshkaSVDLayer is LOST!
```

**After**:
```python
from matryoshka_model_utils import load_matryoshka_model as load_custom

model, tokenizer = load_custom(
    checkpoint_path=checkpoint_path,
    base_model=base_model,
    device=device
)
# MatryoshkaSVDLayer is RECONSTRUCTED!
```

### 3. Test Suite: `test_matryoshka_save_load.py`

Verifies:
- ✅ Metadata file is created during save
- ✅ All MatryoshkaSVDLayer instances are reconstructed
- ✅ Model outputs are identical before/after save/load (0 difference!)
- ✅ Fixed rank inference works
- ✅ Adaptive rank prediction works

**Test Results**:
```
================================================================================
ALL TESTS PASSED ✅
================================================================================

Summary:
✅ save_matryoshka_model() creates all required files
✅ Metadata is correctly saved
✅ load_matryoshka_model() reconstructs all MatryoshkaSVDLayer instances
✅ Model outputs are preserved across save/load
✅ Fixed rank inference works
✅ Adaptive rank prediction works

🎉 Matryoshka SVD save/load mechanism is working correctly!
```

## Files Created/Modified

### New Files:
1. `matryoshka_model_utils.py` - Core save/load utilities
2. `test_matryoshka_save_load.py` - Comprehensive test suite
3. `MATRYOSHKA_SAVE_LOAD_FIX.md` - Detailed documentation
4. `SOLUTION_MATRYOSHKA_MODEL_LOADING.md` - This summary

### Modified Files:
1. `train_matryoshka_from_svdllm.py` - Uses custom save function
2. `evaluate_matryoshka_svdllm.py` - Uses custom load function

## What You Need to Do

### Step 1: Retrain Your Model

**IMPORTANT**: Your existing checkpoint does NOT have the `matryoshka_metadata.json` file, so it cannot be loaded with the new system.

You must retrain the model with the updated training script:

```bash
CUDA_VISIBLE_DEVICES=3 python train_matryoshka_from_svdllm.py \
    --model meta-llama/Llama-2-7b-hf \
    --svdllm_model /path/to/svdllm/model.pt \
    --r_max 256 --r_min 64 \
    --output_dir ./matryoshka_output \
    --num_train_epochs 1
```

**What this will save**:
```
./matryoshka_output/final/
├── config.json
├── model.safetensors (or sharded files)
├── matryoshka_metadata.json  ← NEW! Critical for loading
├── tokenizer.json
└── tokenizer_config.json
```

### Step 2: Verify Checkpoint Structure

After training, verify the checkpoint has metadata:

```python
from matryoshka_model_utils import verify_matryoshka_structure

result = verify_matryoshka_structure('./matryoshka_output/final')
print(f"Has metadata: {result['has_metadata']}")  # Should be True
print(f"Num layers: {result['num_layers']}")  # Should be 224 for Llama-2-7B
```

### Step 3: Evaluate Model

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_matryoshka_svdllm.py \
    --checkpoint ./matryoshka_output/final \
    --eval_rank adaptive \
    --n_eval_samples 256
```

**Expected output**:
```
================================================================================
Loading Matryoshka SVD Model
================================================================================
Checkpoint: ./matryoshka_output/final

Loading Matryoshka metadata...
  Found metadata for 224 Matryoshka layers  ← Should see this!

Loading checkpoint state dict...
  Loaded from model.safetensors
  Total tensors in checkpoint: 512

Reconstructing MatryoshkaSVDLayer instances...
  ✅ model.layers.0.self_attn.q_proj: Linear → MatryoshkaSVDLayer (r=64-256)
  ✅ model.layers.0.self_attn.k_proj: Linear → MatryoshkaSVDLayer (r=64-256)
  ...
  ✅ model.layers.31.mlp.down_proj: Linear → MatryoshkaSVDLayer (r=64-256)

Reconstructed 224/224 layers  ← All layers reconstructed!

================================================================================
Model Loaded Successfully
================================================================================
MatryoshkaSVDLayer instances: 224  ← Should be > 0!
✅ Successfully reconstructed Matryoshka model

Matryoshka Configuration:
  Rank range: [64, 256]
  Predictor mode: rank
  Hard inference: True

================================================================================
Evaluation Results
================================================================================
Dataset: wikitext2
Rank: adaptive
Set 224 layers to ADAPTIVE rank prediction  ← FIXED!
Perplexity: 12.3456  ← Reasonable PPL!
Avg rank: 127.8
```

### Step 4: Multi-Rank Evaluation

```bash
python evaluate_matryoshka_svdllm.py \
    --checkpoint ./matryoshka_output/final \
    --multi_rank_eval \
    --save_results
```

**Expected results**:
```
================================================================================
Multi-Rank Evaluation Summary
================================================================================
Dataset: wikitext2

adaptive:
  Perplexity:    12.35
  Avg rank:     127.8

r_max=256:
  Perplexity:    11.89
  Avg rank:     256.0

r_mid=160:
  Perplexity:    12.24
  Avg rank:     160.0

r_min=64:
  Perplexity:    14.12
  Avg rank:      64.0
```

## Technical Details

### How It Works

1. **Save Process**:
   ```
   save_matryoshka_model()
   ├── model.save_pretrained()  → saves weights
   ├── tokenizer.save_pretrained()  → saves tokenizer
   └── json.dump(metadata)  → saves layer configuration
   ```

2. **Load Process**:
   ```
   load_matryoshka_model()
   ├── Load checkpoint state dict (ALL weights, including MatryoshkaSVDLayer)
   ├── Load base model (creates Linear layers)
   ├── For each layer in metadata:
   │   ├── Create MatryoshkaSVDLayer with saved config
   │   ├── Load weights from checkpoint state dict
   │   └── Replace Linear layer with MatryoshkaSVDLayer
   └── Return reconstructed model
   ```

3. **Why This Works**:
   - Checkpoint state dict preserves ALL weights (U, V, rank_predictor.*)
   - Metadata preserves structure (which layers, what ranks, what mode)
   - Reconstruction creates exact same architecture + loads exact same weights
   - **Result**: Model outputs are identical (0 numerical difference!)

### Metadata Format

```json
{
  "version": "1.0",
  "matryoshka_layers": [
    {
      "name": "model.layers.0.self_attn.q_proj",
      "in_features": 4096,
      "out_features": 4096,
      "r_max": 256,
      "r_min": 64,
      "use_rank_predictor": true,
      "predictor_mode": "rank",
      "gating_tau": 0.1,
      "hard_inference": true,
      "bias": false
    },
    ...
  ]
}
```

### Supported Weight Formats

- ✅ `pytorch_model.bin` (single file)
- ✅ `model.safetensors` (single file)
- ✅ Sharded safetensors (`model-00001-of-00002.safetensors`, etc.)

## Backwards Compatibility

### Old Checkpoints (Without Metadata)

**Symptom**:
```
⚠️  WARNING: matryoshka_metadata.json not found!
Will attempt to load as standard model, but MatryoshkaSVDLayer will be lost.
```

**Solution**: Must retrain with updated training script.

### New Checkpoints (With Metadata)

**Works perfectly**: All MatryoshkaSVDLayer instances are reconstructed.

## Troubleshooting

### Issue 1: "matryoshka_metadata.json not found"

**Cause**: Checkpoint was saved with old training script.

**Fix**: Retrain model.

### Issue 2: "Reconstructed 100/224 layers"

**Cause**: Incomplete or corrupted metadata file.

**Fix**: Check metadata file, retrain if necessary.

### Issue 3: Still seeing "Set 0 layers to ADAPTIVE rank prediction"

**Cause**: Not using the updated evaluation script.

**Fix**: Make sure you're using `evaluate_matryoshka_svdllm.py` that imports `load_matryoshka_model` from `matryoshka_model_utils`.

## Verification Checklist

Before using your model:

- [ ] Trained with updated `train_matryoshka_from_svdllm.py`
- [ ] Checkpoint contains `matryoshka_metadata.json`
- [ ] `verify_matryoshka_structure()` returns `has_metadata=True`
- [ ] Evaluation loads with "Reconstructing MatryoshkaSVDLayer instances..."
- [ ] Evaluation shows "MatryoshkaSVDLayer instances: 224" (or appropriate number)
- [ ] Perplexity is reasonable (10-15 for Llama-2-7B on wikitext2)
- [ ] Adaptive rank shows dynamic prediction (avg rank between r_min and r_max)

## Summary

✅ **Problem Identified**: HuggingFace's save/load loses custom module structure

✅ **Solution Implemented**: Custom save/load with metadata preservation

✅ **Verified**: Tests pass with 0 numerical difference before/after save/load

✅ **Action Required**: Retrain model with updated script

✅ **Expected Result**: Evaluation will show 224 MatryoshkaSVDLayer instances and reasonable perplexity

## Next Steps

1. **Retrain** your model using the updated training script
2. **Verify** the checkpoint has `matryoshka_metadata.json`
3. **Evaluate** and confirm you see 224 MatryoshkaSVDLayer instances
4. **Celebrate** 🎉 when you see reasonable perplexity values!

---

**Files to reference**:
- `matryoshka_model_utils.py` - Implementation
- `MATRYOSHKA_SAVE_LOAD_FIX.md` - Detailed technical documentation
- `test_matryoshka_save_load.py` - Test suite showing it works
