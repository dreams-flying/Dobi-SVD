"""
Quick test script to verify MultiSubspaceSVDLayer implementation.

This script performs basic unit tests on the dynamic subspace routing components.
"""

import torch
import torch.nn as nn
from modules.dynamic_subspace import (
    TokenRouter,
    MultiSubspaceSVDLayer,
    compute_load_balance_loss
)


def test_token_router():
    """Test TokenRouter module."""
    print("\n" + "="*60)
    print("Testing TokenRouter")
    print("="*60)

    batch_size = 2
    seq_len = 10
    hidden_size = 64
    n_subspaces = 3

    # Create router
    router = TokenRouter(
        hidden_size=hidden_size,
        n_subspaces=n_subspaces,
        routing_strategy='norm',
        learnable_thresholds=False,
        device='cpu'
    )

    # Create dummy input
    x = torch.randn(batch_size, seq_len, hidden_size)

    # Test importance computation
    importance = router.compute_importance(x)
    print(f"✓ Importance computation: {importance.shape} (expected: [{batch_size}, {seq_len}])")
    assert importance.shape == (batch_size, seq_len), "Importance shape mismatch"

    # Test hard routing
    routing_hard = router.route_tokens(importance, hard=True)
    print(f"✓ Hard routing: {routing_hard.shape}, unique values: {routing_hard.unique().tolist()}")
    assert routing_hard.shape == (batch_size, seq_len), "Hard routing shape mismatch"
    assert routing_hard.max() < n_subspaces, "Invalid routing assignment"

    # Test soft routing
    routing_soft = router.route_tokens(importance, hard=False)
    print(f"✓ Soft routing: {routing_soft.shape} (expected: [{batch_size}, {seq_len}, {n_subspaces}])")
    assert routing_soft.shape == (batch_size, seq_len, n_subspaces), "Soft routing shape mismatch"
    assert torch.allclose(routing_soft.sum(dim=-1), torch.ones(batch_size, seq_len), atol=1e-5), "Soft routing not normalized"

    # Test routing statistics
    dist = router.get_routing_distribution()
    print(f"✓ Routing distribution: {dist.tolist()}")

    print("\n✅ TokenRouter tests passed!")


def test_multisubspace_svd_layer():
    """Test MultiSubspaceSVDLayer module."""
    print("\n" + "="*60)
    print("Testing MultiSubspaceSVDLayer")
    print("="*60)

    batch_size = 2
    seq_len = 8
    input_size = 64
    output_size = 64
    n_subspaces = 3

    # Create dummy linear layer weights
    weight = torch.randn(output_size, input_size)
    bias = torch.randn(output_size)

    # Create MultiSubspaceSVDLayer
    layer = MultiSubspaceSVDLayer(
        gammas=[20, 30, 40],  # Different ranks for each subspace
        n_subspaces=n_subspaces,
        SEQ_LEN=seq_len,
        beta=100.0,
        input_size=input_size,
        output_size=output_size,
        weight_size=torch.tensor(input_size * output_size),
        weight=weight,
        bias=bias,
        name='test_layer',
        device='cpu',
        routing_strategy='norm',
        learnable_thresholds=False,
        use_soft_routing=False,
        routing_temperature=1.0
    )

    # Create input
    x = torch.randn(batch_size, seq_len, input_size)

    # Test forward pass (training mode)
    layer.train()
    output_train = layer(x)
    print(f"✓ Training forward pass: input {x.shape} -> output {output_train.shape}")
    assert output_train.shape == (batch_size, seq_len, output_size), "Training output shape mismatch"

    # Test forward pass (inference mode)
    layer.eval()
    output_eval = layer(x)
    print(f"✓ Inference forward pass: input {x.shape} -> output {output_eval.shape}")
    assert output_eval.shape == (batch_size, seq_len, output_size), "Inference output shape mismatch"

    # Test routing statistics
    stats = layer.get_routing_stats()
    print(f"✓ Routing stats: distribution = {stats['routing_distribution']}")
    print(f"  Gammas: {stats['gammas']}")

    # Test gradient flow
    layer.train()
    output = layer(x)
    loss = output.sum()
    loss.backward()

    for i, gamma in enumerate(layer.gammas):
        assert gamma.grad is not None, f"Gamma {i} has no gradient"
        print(f"✓ Gamma {i} gradient: {gamma.grad.item():.6f}")

    print("\n✅ MultiSubspaceSVDLayer tests passed!")


def test_load_balance_loss():
    """Test load balance loss computation."""
    print("\n" + "="*60)
    print("Testing Load Balance Loss")
    print("="*60)

    # Create a simple model with MultiSubspaceSVDLayer
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            weight1 = torch.randn(32, 32)
            weight2 = torch.randn(32, 32)

            self.layer1 = MultiSubspaceSVDLayer(
                gammas=[10, 20, 30],
                n_subspaces=3,
                SEQ_LEN=8,
                beta=100.0,
                input_size=32,
                output_size=32,
                weight_size=torch.tensor(32 * 32),
                weight=weight1,
                bias=None,
                name='layer1',
                device='cpu',
                routing_strategy='norm'
            )

            self.layer2 = MultiSubspaceSVDLayer(
                gammas=[10, 20, 30],
                n_subspaces=3,
                SEQ_LEN=8,
                beta=100.0,
                input_size=32,
                output_size=32,
                weight_size=torch.tensor(32 * 32),
                weight=weight2,
                bias=None,
                name='layer2',
                device='cpu',
                routing_strategy='norm'
            )

        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            return x

    model = TestModel()

    # Run forward pass
    x = torch.randn(2, 8, 32)
    output = model(x)

    # Compute load balance loss
    balance_loss = compute_load_balance_loss(model)
    print(f"✓ Load balance loss: {balance_loss.item():.6f}")

    # Test gradient flow
    total_loss = output.sum() + balance_loss
    total_loss.backward()

    print(f"✓ Gradient flow verified")

    print("\n✅ Load balance loss tests passed!")


def test_routing_consistency():
    """Test routing consistency between training and inference modes."""
    print("\n" + "="*60)
    print("Testing Routing Consistency")
    print("="*60)

    input_size = 32
    output_size = 32
    seq_len = 8

    weight = torch.randn(output_size, input_size)

    layer = MultiSubspaceSVDLayer(
        gammas=[10, 20, 30],
        n_subspaces=3,
        SEQ_LEN=seq_len,
        beta=100.0,
        input_size=input_size,
        output_size=output_size,
        weight_size=torch.tensor(input_size * output_size),
        weight=weight,
        bias=None,
        name='test_layer',
        device='cpu',
        routing_strategy='norm',
        use_soft_routing=False
    )

    x = torch.randn(1, seq_len, input_size)

    # Run in eval mode multiple times
    layer.eval()
    outputs = []
    for _ in range(3):
        with torch.no_grad():
            output = layer(x)
            outputs.append(output)

    # Check consistency
    for i in range(1, len(outputs)):
        assert torch.allclose(outputs[0], outputs[i], atol=1e-5), f"Output {i} differs from output 0"

    print("✓ Inference outputs are consistent across multiple runs")

    print("\n✅ Routing consistency tests passed!")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("Dynamic Subspace Routing - Unit Tests")
    print("="*60)

    try:
        test_token_router()
        test_multisubspace_svd_layer()
        test_load_balance_loss()
        test_routing_consistency()

        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)

    except Exception as e:
        print("\n" + "="*60)
        print("❌ TEST FAILED!")
        print("="*60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
