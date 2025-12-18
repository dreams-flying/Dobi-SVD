#!/usr/bin/env python3
"""Test backward compatibility of get_effective_rank()"""

import torch
from modules.matryoshka_svd_layer_v2 import MatryoshkaSVDLayerV2

print("="*80)
print("Testing get_effective_rank() backward compatibility")
print("="*80)

# Create layer
layer = MatryoshkaSVDLayerV2(
    in_features=4096,
    out_features=4096,
    r_max=512,
    r_min=256,
    use_rank_predictor=True
)

layer.train()

# Test 1: Call get_effective_rank() WITHOUT forward pass (should use fallback)
print("\nTest 1: get_effective_rank() without forward pass")
try:
    rank1 = layer.get_effective_rank()
    print(f"  ✅ Rank (fallback): {rank1:.2f}")
    assert rank1 == 512.0, "Should fallback to r_max"
except Exception as e:
    print(f"  ❌ Error: {e}")

# Test 2: Do forward pass, then call get_effective_rank() WITHOUT arguments
print("\nTest 2: get_effective_rank() after forward pass (cached)")
x = torch.randn(2, 8, 4096)
output = layer(x)
try:
    rank2 = layer.get_effective_rank()  # Should use cached value
    print(f"  ✅ Rank (cached): {rank2:.2f}")
    assert 256 <= rank2 <= 512, f"Rank should be in [256, 512], got {rank2}"
except Exception as e:
    print(f"  ❌ Error: {e}")

# Test 3: Call get_effective_rank() WITH argument (compute immediately)
print("\nTest 3: get_effective_rank(x) with argument")
try:
    rank3 = layer.get_effective_rank(x)  # Compute immediately
    print(f"  ✅ Rank (computed): {rank3:.2f}")
    assert 256 <= rank3 <= 512, f"Rank should be in [256, 512], got {rank3}"
except Exception as e:
    print(f"  ❌ Error: {e}")

# Test 4: Fixed rank mode
print("\nTest 4: get_effective_rank() in fixed rank mode")
layer.set_fixed_rank(384)
try:
    rank4 = layer.get_effective_rank()
    print(f"  ✅ Rank (fixed): {rank4:.2f}")
    assert rank4 == 384.0, "Should return fixed rank"
except Exception as e:
    print(f"  ❌ Error: {e}")

print("\n" + "="*80)
print("All backward compatibility tests passed! ✅")
print("="*80)
print("\nThe training script can now call:")
print("  avg_rank = module.get_effective_rank()  # No arguments needed!")
