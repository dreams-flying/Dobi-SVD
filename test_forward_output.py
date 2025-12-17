#!/usr/bin/env python3
"""
Test if MatryoshkaSVDLayer forward pass produces reasonable outputs.
"""

import torch
from matryoshka_model_utils import load_matryoshka_model

print("="*80)
print("Testing MatryoshkaSVDLayer Forward Pass")
print("="*80)

# Load model
print("\nLoading model...")
model, tokenizer = load_matryoshka_model(
    checkpoint_path="matryoshka_output0/final",
    device='cuda',
    torch_dtype=torch.float32  # Use float32 for better numerical stability
)

model.eval()

# Create test input
batch_size = 1
seq_len = 4
hidden_size = 4096

x = torch.randn(batch_size, seq_len, hidden_size, device='cuda', dtype=torch.float32)

print(f"\nTest input shape: {x.shape}")
print(f"Test input mean: {x.mean().item():.6f}, std: {x.std().item():.6f}")

# Test first layer's q_matryoshka
print(f"\n{'='*80}")
print("Testing model.layers.0.self_attn.q_matryoshka")
print("="*80)

q_layer = model.model.layers[0].self_attn.q_matryoshka

# Set to fixed rank for testing
q_layer.set_fixed_rank(512)  # Use maximum rank

with torch.no_grad():
    output = q_layer(x)

print(f"\nOutput shape: {output.shape}")
print(f"Output mean: {output.mean().item():.6f}")
print(f"Output std: {output.std().item():.6f}")
print(f"Output min: {output.min().item():.6f}")
print(f"Output max: {output.max().item():.6f}")

# Check for problems
has_nan = torch.isnan(output).any().item()
has_inf = torch.isinf(output).any().item()
is_all_zeros = (output.abs() < 1e-6).all().item()

print(f"\nSanity checks:")
print(f"  Has NaN: {has_nan}")
print(f"  Has Inf: {has_inf}")
print(f"  All zeros: {is_all_zeros}")

if has_nan:
    print("  ❌ PROBLEM: Output contains NaN!")
elif has_inf:
    print("  ❌ PROBLEM: Output contains Inf!")
elif is_all_zeros:
    print("  ❌ PROBLEM: Output is all zeros!")
else:
    print("  ✅ Output looks reasonable")

# Test adaptive rank
print(f"\n{'='*80}")
print("Testing with ADAPTIVE rank prediction")
print("="*80)

q_layer.set_fixed_rank(None)  # Enable adaptive

with torch.no_grad():
    output_adaptive = q_layer(x)

    # Get predicted rank
    if hasattr(q_layer, 'rank_predictor'):
        predicted_rank = q_layer.rank_predictor(x)
        print(f"Predicted rank: {predicted_rank.mean().item():.2f}")

print(f"\nOutput shape: {output_adaptive.shape}")
print(f"Output mean: {output_adaptive.mean().item():.6f}")
print(f"Output std: {output_adaptive.std().item():.6f}")

# Check if adaptive output is different from fixed
diff = (output_adaptive - output).abs().mean().item()
print(f"\nDifference from fixed rank=512: {diff:.6f}")
if diff < 1e-6:
    print("  ⚠️  WARNING: Adaptive and fixed outputs are identical!")
    print("  This suggests rank predictor might not be working")

# Test full model forward
print(f"\n{'='*80}")
print("Testing FULL MODEL forward pass")
print("="*80)

# Create realistic input (token IDs)
input_ids = torch.randint(0, 32000, (1, 10), device='cuda')
print(f"Input IDs shape: {input_ids.shape}")

with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs.logits

print(f"\nLogits shape: {logits.shape}")
print(f"Logits mean: {logits.mean().item():.6f}")
print(f"Logits std: {logits.std().item():.6f}")
print(f"Logits min: {logits.min().item():.6f}")
print(f"Logits max: {logits.max().item():.6f}")

# Check for problems
has_nan = torch.isnan(logits).any().item()
has_inf = torch.isinf(logits).any().item()

print(f"\nSanity checks:")
print(f"  Has NaN: {has_nan}")
print(f"  Has Inf: {has_inf}")

if has_nan or has_inf:
    print("  ❌ PROBLEM: Model output has NaN/Inf!")
else:
    print("  ✅ Model output looks reasonable")

    # Compute perplexity for this single sample
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, logits.size(-1)), shift_labels.view(-1))
    ppl = torch.exp(loss).item()

    print(f"\nSingle sample perplexity: {ppl:.4f}")
    if ppl > 1000:
        print("  ❌ PROBLEM: PPL is extremely high!")
        print("  This suggests the model is producing random/wrong predictions")
    elif ppl > 100:
        print("  ⚠️  WARNING: PPL is quite high")
    else:
        print("  ✅ PPL is reasonable")

print(f"\n{'='*80}")
print("Diagnostic Complete")
print("="*80)
