#!/usr/bin/env python3
"""
Quick integration test for advanced routing strategies.

Tests that the advanced routing is properly integrated into the training pipeline.
"""

import torch
import sys
from modules.dynamic_subspace import TokenRouter, MultiSubspaceSVDLayer

def test_token_router_advanced():
    """Test TokenRouter with advanced routing."""
    print("Testing TokenRouter with advanced routing...")

    # Test parameters
    hidden_size = 768
    n_subspaces = 3
    batch_size = 2
    seq_len = 10

    strategies = ['topk', 'expert_choice', 'sinkhorn', 'gating', 'adaptive']

    for strategy in strategies:
        print(f"\n  Testing {strategy}...")

        # Prepare kwargs
        kwargs = {}
        if strategy in ['topk', 'expert_choice']:
            kwargs['top_k'] = 1
            kwargs['capacity_factor'] = 1.25
        elif strategy == 'sinkhorn':
            kwargs['sinkhorn_iters'] = 3

        try:
            # Create router
            router = TokenRouter(
                hidden_size=hidden_size,
                n_subspaces=n_subspaces,
                routing_strategy='norm',  # importance calculation strategy
                advanced_routing=strategy,  # routing assignment strategy
                advanced_routing_kwargs=kwargs,
                device='cpu'
            )

            # Create fake input
            x = torch.randn(batch_size, seq_len, hidden_size)

            # Compute importance
            importance = router.compute_importance(x)
            assert importance.shape == (batch_size, seq_len), f"Wrong importance shape: {importance.shape}"

            # Test hard routing
            routing_hard = router.route_tokens(importance, hard=True)
            assert routing_hard.shape == (batch_size, seq_len), f"Wrong hard routing shape: {routing_hard.shape}"
            assert routing_hard.dtype == torch.long, f"Wrong hard routing dtype: {routing_hard.dtype}"

            # Test soft routing
            routing_soft = router.route_tokens(importance, hard=False)
            if routing_soft.dim() == 3:
                assert routing_soft.shape == (batch_size, seq_len, n_subspaces), \
                    f"Wrong soft routing shape: {routing_soft.shape}"

            print(f"    ✓ {strategy} routing works!")
            print(f"      Hard routing distribution: {[(routing_hard == i).sum().item() for i in range(n_subspaces)]}")

        except Exception as e:
            print(f"    ✗ {strategy} routing failed: {e}")
            return False

    print("\n✓ All TokenRouter advanced routing tests passed!")
    return True


def test_multisubspace_layer_advanced():
    """Test MultiSubspaceSVDLayer with advanced routing."""
    print("\nTesting MultiSubspaceSVDLayer with advanced routing...")

    # Test parameters
    input_size = 256
    output_size = 512
    seq_len = 128
    n_subspaces = 3

    # Create fake weight
    weight = torch.randn(output_size, input_size)

    # Test with expert_choice (best performing)
    try:
        layer = MultiSubspaceSVDLayer(
            gammas=[64, 128, 192],
            n_subspaces=n_subspaces,
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
            advanced_routing='expert_choice',
            advanced_routing_kwargs={'capacity_factor': 1.25}
        )

        # Test forward pass
        x = torch.randn(1, seq_len, input_size)
        output = layer(x)

        print(f"  ✓ MultiSubspaceSVDLayer with expert_choice works!")
        print(f"    Input shape: {x.shape}")
        print(f"    Output shape: {output.shape}")

        return True

    except Exception as e:
        print(f"  ✗ MultiSubspaceSVDLayer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backward_compatibility():
    """Test that default behavior (no advanced routing) still works."""
    print("\nTesting backward compatibility (no advanced routing)...")

    try:
        router = TokenRouter(
            hidden_size=768,
            n_subspaces=3,
            routing_strategy='norm',
            device='cpu'
        )

        x = torch.randn(2, 10, 768)
        importance = router.compute_importance(x)
        routing = router.route_tokens(importance, hard=True)

        print(f"  ✓ Default threshold-based routing still works!")
        print(f"    Routing distribution: {[(routing == i).sum().item() for i in range(3)]}")

        return True

    except Exception as e:
        print(f"  ✗ Backward compatibility test failed: {e}")
        return False


if __name__ == '__main__':
    print("="*70)
    print("Advanced Routing Integration Test")
    print("="*70)

    all_passed = True

    # Run tests
    all_passed &= test_token_router_advanced()
    all_passed &= test_multisubspace_layer_advanced()
    all_passed &= test_backward_compatibility()

    print("\n" + "="*70)
    if all_passed:
        print("✓ All integration tests PASSED!")
        print("="*70)
        sys.exit(0)
    else:
        print("✗ Some integration tests FAILED")
        print("="*70)
        sys.exit(1)
