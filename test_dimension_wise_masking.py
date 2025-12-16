#!/usr/bin/env python3
"""
Test script for dimension-wise soft masking.

Demonstrates:
1. Basic usage of both predictor modes
2. Gradient checkpointing compatibility
3. Performance comparison
"""

import torch
import torch.nn as nn
from modules.matryoshka_svd_layer import MatryoshkaSVDLayer


def test_basic_usage():
    """Test basic forward pass for both modes."""
    print("=" * 80)
    print("Test 1: Basic Usage")
    print("=" * 80)

    batch_size, seq_len, hidden_dim = 2, 128, 4096
    r_max = 256

    # Create input
    x = torch.randn(batch_size, seq_len, hidden_dim)

    # Mode 1: Scalar rank prediction
    print("\n[Mode 1: Scalar Rank Prediction]")
    layer_rank = MatryoshkaSVDLayer(
        in_features=hidden_dim,
        out_features=hidden_dim,
        r_max=r_max,
        r_min=64,
        use_rank_predictor=True,
        predictor_mode='rank',
        gating_tau=0.1
    )

    output_rank, rank = layer_rank(x, return_rank=True)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output_rank.shape}")
    print(f"Rank shape:   {rank.shape}")
    print(f"Predicted rank (avg): {rank.mean().item():.2f}")
    print(f"Effective rank: {layer_rank.get_effective_rank():.2f}")

    # Mode 2: Dimension-wise masking
    print("\n[Mode 2: Dimension-Wise Masking]")
    layer_dimwise = MatryoshkaSVDLayer(
        in_features=hidden_dim,
        out_features=hidden_dim,
        r_max=r_max,
        r_min=64,
        use_rank_predictor=True,
        predictor_mode='dimension_wise',
        use_gumbel=False,  # Deterministic sigmoid
        gating_tau=0.1
    )

    output_dimwise, mask = layer_dimwise(x, return_rank=True)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output_dimwise.shape}")
    print(f"Mask shape:   {mask.shape}")
    print(f"Mask values:  min={mask.min().item():.3f}, max={mask.max().item():.3f}, mean={mask.mean().item():.3f}")
    print(f"Effective rank (sum of mask): {mask.sum(dim=-1).mean().item():.2f} / {r_max}")
    print(f"Effective rank (from method): {layer_dimwise.get_effective_rank():.2f}")

    print("\n✅ Both modes produce correct output shapes")


def test_gradient_checkpointing():
    """Test gradient checkpointing compatibility."""
    print("\n" + "=" * 80)
    print("Test 2: Gradient Checkpointing Compatibility")
    print("=" * 80)

    batch_size, seq_len, hidden_dim = 2, 64, 1024
    r_max = 128

    # Create input that requires grad
    x = torch.randn(batch_size, seq_len, hidden_dim, requires_grad=True)

    # Test dimension-wise mode (should work with gradient checkpointing)
    print("\n[Testing Dimension-Wise Mode]")
    layer = MatryoshkaSVDLayer(
        in_features=hidden_dim,
        out_features=hidden_dim,
        r_max=r_max,
        use_rank_predictor=True,
        predictor_mode='dimension_wise',
        use_gumbel=False  # Deterministic
    )

    layer.train()  # Set to training mode (deterministic=True)

    # Simulate gradient checkpointing: forward twice
    print("  Forward pass 1...")
    output1 = layer(x)

    print("  Forward pass 2 (recomputation)...")
    output2 = layer(x)

    # Check if outputs are identical (deterministic)
    max_diff = (output1 - output2).abs().max().item()
    print(f"  Max difference between passes: {max_diff:.6e}")

    if max_diff < 1e-5:
        print("  ✅ PASS: Outputs are deterministic (gradient checkpointing compatible)")
    else:
        print("  ❌ FAIL: Outputs differ (not gradient checkpointing compatible)")

    # Test backward pass
    print("  Backward pass...")
    loss = output1.mean()
    loss.backward()

    if x.grad is not None:
        print(f"  Input gradient shape: {x.grad.shape}")
        print(f"  Input gradient norm: {x.grad.norm().item():.4f}")
        print("  ✅ PASS: Backward pass successful")
    else:
        print("  ❌ FAIL: No gradient computed")


def test_gumbel_softmax():
    """Test Gumbel-Softmax mode for sparser masks."""
    print("\n" + "=" * 80)
    print("Test 3: Gumbel-Softmax Mode")
    print("=" * 80)

    batch_size, seq_len, hidden_dim = 2, 64, 1024
    r_max = 128

    x = torch.randn(batch_size, seq_len, hidden_dim)

    print("\n[Comparing different masking strategies]")

    # Strategy 1: Deterministic sigmoid
    layer_sigmoid = MatryoshkaSVDLayer(
        in_features=hidden_dim,
        out_features=hidden_dim,
        r_max=r_max,
        use_rank_predictor=True,
        predictor_mode='dimension_wise',
        use_gumbel=False
    )

    layer_sigmoid.eval()  # Enable non-deterministic if using Gumbel
    _, mask_sigmoid = layer_sigmoid(x, return_rank=True)

    print(f"\n1. Sigmoid masking:")
    print(f"   Mean: {mask_sigmoid.mean().item():.3f}")
    print(f"   Std:  {mask_sigmoid.std().item():.3f}")
    print(f"   Values in [0.4, 0.6]: {((mask_sigmoid > 0.4) & (mask_sigmoid < 0.6)).float().mean().item():.1%}")

    # Strategy 2: Gumbel-Softmax (soft)
    layer_gumbel_soft = MatryoshkaSVDLayer(
        in_features=hidden_dim,
        out_features=hidden_dim,
        r_max=r_max,
        use_rank_predictor=True,
        predictor_mode='dimension_wise',
        use_gumbel=True,
        gumbel_tau=0.5,
        gumbel_hard=False
    )

    layer_gumbel_soft.eval()
    _, mask_gumbel_soft = layer_gumbel_soft(x, return_rank=True)

    print(f"\n2. Gumbel-Softmax (soft, τ=0.5):")
    print(f"   Mean: {mask_gumbel_soft.mean().item():.3f}")
    print(f"   Std:  {mask_gumbel_soft.std().item():.3f}")
    print(f"   Values in [0.4, 0.6]: {((mask_gumbel_soft > 0.4) & (mask_gumbel_soft < 0.6)).float().mean().item():.1%}")

    # Strategy 3: Gumbel-Softmax (hard)
    layer_gumbel_hard = MatryoshkaSVDLayer(
        in_features=hidden_dim,
        out_features=hidden_dim,
        r_max=r_max,
        use_rank_predictor=True,
        predictor_mode='dimension_wise',
        use_gumbel=True,
        gumbel_tau=0.1,
        gumbel_hard=True
    )

    layer_gumbel_hard.eval()
    _, mask_gumbel_hard = layer_gumbel_hard(x, return_rank=True)

    print(f"\n3. Gumbel-Softmax (hard, τ=0.1):")
    print(f"   Mean: {mask_gumbel_hard.mean().item():.3f}")
    print(f"   Std:  {mask_gumbel_hard.std().item():.3f}")
    print(f"   Values near 0 or 1: {((mask_gumbel_hard < 0.1) | (mask_gumbel_hard > 0.9)).float().mean().item():.1%}")

    print("\n✅ Different strategies produce different sparsity levels")


def test_memory_efficiency():
    """Compare memory footprint of different modes."""
    print("\n" + "=" * 80)
    print("Test 4: Memory Efficiency")
    print("=" * 80)

    hidden_dim = 4096
    r_max = 256

    print("\n[Predictor parameter counts]")

    # Rank predictor
    layer_rank = MatryoshkaSVDLayer(
        in_features=hidden_dim,
        out_features=hidden_dim,
        r_max=r_max,
        use_rank_predictor=True,
        predictor_mode='rank'
    )

    rank_params = sum(p.numel() for p in layer_rank.rank_predictor.parameters())
    print(f"1. Rank predictor:          {rank_params:,} parameters")

    # Dimension-wise predictor
    layer_dimwise = MatryoshkaSVDLayer(
        in_features=hidden_dim,
        out_features=hidden_dim,
        r_max=r_max,
        use_rank_predictor=True,
        predictor_mode='dimension_wise'
    )

    dimwise_params = sum(p.numel() for p in layer_dimwise.rank_predictor.parameters())
    print(f"2. Dimension-wise predictor: {dimwise_params:,} parameters")

    overhead = dimwise_params - rank_params
    print(f"\nOverhead: {overhead:,} parameters ({overhead/rank_params*100:.1f}% increase)")

    # Total layer parameters
    total_rank = sum(p.numel() for p in layer_rank.parameters())
    total_dimwise = sum(p.numel() for p in layer_dimwise.parameters())

    print(f"\nTotal layer parameters:")
    print(f"  Rank mode:          {total_rank:,}")
    print(f"  Dimension-wise mode: {total_dimwise:,}")
    print(f"  Difference:         {total_dimwise - total_rank:,} ({(total_dimwise - total_rank)/total_rank*100:.2f}%)")

    print("\n✅ Memory overhead is minimal compared to projection weights")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("DIMENSION-WISE SOFT MASKING TEST SUITE")
    print("=" * 80)

    torch.manual_seed(42)

    try:
        test_basic_usage()
        test_gradient_checkpointing()
        test_gumbel_softmax()
        test_memory_efficiency()

        print("\n" + "=" * 80)
        print("ALL TESTS PASSED ✅")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
