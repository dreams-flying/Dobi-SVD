"""
Test script for Matryoshka SVD implementation.

Validates:
1. SVD decomposition correctness
2. Soft truncation behavior
3. Forward pass numerical stability
4. Gradient flow
5. Compression ratio calculation

Usage:
    python test_matryoshka_svd.py
"""

import torch
import torch.nn as nn
from modules.matryoshka_svd import MatryoshkaSVDLayer, AdaptiveRankPredictor


def test_adaptive_rank_predictor():
    """Test AdaptiveRankPredictor with different strategies."""
    print("\n" + "="*80)
    print("TEST 1: AdaptiveRankPredictor")
    print("="*80)

    hidden_size = 128
    batch_size = 2
    seq_len = 16

    x = torch.randn(batch_size, seq_len, hidden_size)

    # Test norm-based
    print("\n[Test 1.1] Norm-based importance")
    predictor_norm = AdaptiveRankPredictor(hidden_size, strategy='norm')
    importance_norm = predictor_norm(x)

    print(f"  Input shape: {x.shape}")
    print(f"  Importance shape: {importance_norm.shape}")
    print(f"  Importance range: [{importance_norm.min():.4f}, {importance_norm.max():.4f}]")
    print(f"  Importance mean: {importance_norm.mean():.4f}")

    assert importance_norm.shape == (batch_size, seq_len), "Wrong shape for norm importance"
    assert (importance_norm >= 0).all() and (importance_norm <= 1).all(), "Importance not in [0,1]"
    print("  ✓ Passed")

    # Test learned predictor
    print("\n[Test 1.2] Learned importance predictor")
    predictor_learned = AdaptiveRankPredictor(hidden_size, strategy='learned')
    importance_learned = predictor_learned(x)

    print(f"  Importance shape: {importance_learned.shape}")
    print(f"  Importance range: [{importance_learned.min():.4f}, {importance_learned.max():.4f}]")

    assert importance_learned.shape == (batch_size, seq_len), "Wrong shape for learned importance"
    assert (importance_learned >= 0).all() and (importance_learned <= 1).all(), "Importance not in [0,1]"
    print("  ✓ Passed")

    # Test gradient flow
    print("\n[Test 1.3] Gradient flow through learned predictor")
    loss = importance_learned.sum()
    loss.backward()

    grad_exists = False
    for param in predictor_learned.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            grad_exists = True
            break

    assert grad_exists, "No gradients in learned predictor"
    print("  ✓ Gradients exist")


def test_soft_truncation():
    """Test soft truncation function."""
    print("\n" + "="*80)
    print("TEST 2: Soft Truncation")
    print("="*80)

    # Create a simple layer
    weight = torch.randn(64, 128)
    layer = MatryoshkaSVDLayer(
        input_size=128,
        output_size=64,
        weight=weight,
        r_max=32,
        r_min=8,
        temperature=0.1
    )

    # Test truncation for different ranks
    print("\n[Test 2.1] Truncation behavior")
    batch_size = 1
    seq_len = 1

    for test_rank in [8, 16, 24, 32]:
        adaptive_rank = torch.tensor([[test_rank]], dtype=torch.float32)
        truncation = layer.compute_soft_truncation(adaptive_rank)  # [1, 1, 32]

        # Check which components are kept
        truncation_vals = truncation[0, 0].tolist()
        high_vals = sum(1 for v in truncation_vals[:test_rank] if v > 0.9)
        low_vals = sum(1 for v in truncation_vals[test_rank:] if v < 0.1)

        print(f"  Rank={test_rank}:")
        print(f"    High values (>0.9) in first {test_rank}: {high_vals}/{test_rank}")
        print(f"    Low values (<0.1) after {test_rank}: {low_vals}/{32-test_rank}")

        # Most values before rank should be high, most after should be low
        assert high_vals >= test_rank * 0.7, f"Too few high values for rank={test_rank}"
        if test_rank < 32:
            assert low_vals >= (32 - test_rank) * 0.5, f"Too few low values for rank={test_rank}"

    print("  ✓ Passed")


def test_forward_pass():
    """Test forward pass and numerical stability."""
    print("\n" + "="*80)
    print("TEST 3: Forward Pass")
    print("="*80)

    # Create layer
    input_size = 256
    output_size = 128
    batch_size = 4
    seq_len = 32

    weight = torch.randn(output_size, input_size) * 0.01
    layer = MatryoshkaSVDLayer(
        input_size=input_size,
        output_size=output_size,
        weight=weight,
        r_max=64,
        r_min=16,
        importance_strategy='norm',
        temperature=0.1
    )

    # Forward pass
    print("\n[Test 3.1] Basic forward pass")
    x = torch.randn(batch_size, seq_len, input_size)
    output = layer(x)

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")

    assert output.shape == (batch_size, seq_len, output_size), "Wrong output shape"
    assert not torch.isnan(output).any(), "NaN in output"
    assert not torch.isinf(output).any(), "Inf in output"
    print("  ✓ Passed")

    # Test with rank info
    print("\n[Test 3.2] Forward pass with rank info")
    output, rank_info = layer(x, return_rank_info=True)

    print(f"  Avg rank: {rank_info['avg_rank']:.2f}")
    print(f"  Rank range: [{rank_info['min_rank']:.2f}, {rank_info['max_rank']:.2f}]")
    print(f"  Rank std: {rank_info['rank_std']:.2f}")

    assert 16 <= rank_info['avg_rank'] <= 64, "Avg rank out of bounds"
    assert rank_info['min_rank'] >= 16, "Min rank below r_min"
    assert rank_info['max_rank'] <= 64, "Max rank above r_max"
    print("  ✓ Passed")


def test_gradient_flow():
    """Test gradient flow through the layer."""
    print("\n" + "="*80)
    print("TEST 4: Gradient Flow")
    print("="*80)

    # Create layer with learned predictor
    weight = torch.randn(64, 128)
    layer = MatryoshkaSVDLayer(
        input_size=128,
        output_size=64,
        weight=weight,
        r_max=32,
        r_min=8,
        importance_strategy='learned',  # Learned predictor has parameters
        temperature=0.1
    )

    print("\n[Test 4.1] Gradient flow through learned predictor")
    x = torch.randn(2, 16, 128, requires_grad=True)
    output = layer(x)
    loss = output.sum()
    loss.backward()

    # Check gradients exist for predictor parameters
    grad_exists = False
    total_grad_norm = 0.0

    for name, param in layer.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norm = param.grad.norm().item()
            total_grad_norm += grad_norm
            if grad_norm > 0:
                grad_exists = True
                print(f"  {name}: grad_norm={grad_norm:.6f}")

    assert grad_exists, "No gradients in layer"
    assert total_grad_norm > 0, "Zero total gradient norm"
    print("  ✓ Passed")


def test_compression_ratio():
    """Test compression ratio calculation."""
    print("\n" + "="*80)
    print("TEST 5: Compression Ratio")
    print("="*80)

    input_size = 4096
    output_size = 4096

    weight = torch.randn(output_size, input_size) * 0.01

    print("\n[Test 5.1] Different rank configurations")
    for r_max, r_min in [(256, 64), (128, 32), (64, 16)]:
        layer = MatryoshkaSVDLayer(
            input_size=input_size,
            output_size=output_size,
            weight=weight,
            r_max=r_max,
            r_min=r_min,
            importance_strategy='norm'
        )

        # Simulate some forwards to get avg rank
        x = torch.randn(1, 10, input_size)
        for _ in range(5):
            _ = layer(x)

        compression = layer.get_compression_ratio()
        original_params = input_size * output_size
        compressed_params = int(compression * original_params)

        print(f"  r_max={r_max}, r_min={r_min}:")
        print(f"    Avg rank: {layer.avg_rank_tracker.item():.1f}")
        print(f"    Compression: {compression:.1%}")
        print(f"    Params: {compressed_params:,} / {original_params:,}")

        assert 0 < compression < 1, "Compression ratio not in (0, 1)"

    print("  ✓ Passed")


def test_nested_structure():
    """Test Matryoshka nested property: using more rank should not hurt."""
    print("\n" + "="*80)
    print("TEST 6: Nested Structure (Matryoshka Property)")
    print("="*80)

    weight = torch.randn(128, 256) * 0.01
    layer = MatryoshkaSVDLayer(
        input_size=256,
        output_size=128,
        weight=weight,
        r_max=64,
        r_min=16,
        temperature=0.01  # Very hard truncation to approximate discrete ranks
    )

    x = torch.randn(1, 8, 256)

    print("\n[Test 6.1] Reconstruction error decreases with rank")
    # Get original weight approximation using full SVD
    U = layer.U
    S = layer.S
    V = layer.V
    W_full = U @ torch.diag(S) @ V

    errors = []
    ranks_to_test = [16, 24, 32, 48, 64]

    for rank in ranks_to_test:
        # Reconstruct with fixed rank
        W_k = U[:, :rank] @ torch.diag(S[:rank]) @ V[:rank, :]
        error = (weight - W_k).norm() / weight.norm()
        errors.append(error.item())
        print(f"  Rank={rank}: reconstruction_error={error.item():.6f}")

    # Check monotonic decrease (nested property)
    for i in range(len(errors) - 1):
        assert errors[i] >= errors[i+1], f"Error increased from rank {ranks_to_test[i]} to {ranks_to_test[i+1]}"

    print("  ✓ Nested property verified")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*80)
    print("MATRYOSHKA SVD VALIDATION TESTS")
    print("="*80)

    try:
        test_adaptive_rank_predictor()
        test_soft_truncation()
        test_forward_pass()
        test_gradient_flow()
        test_compression_ratio()
        test_nested_structure()

        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED")
        print("="*80)
        return True

    except Exception as e:
        print("\n" + "="*80)
        print("❌ TEST FAILED")
        print("="*80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)
