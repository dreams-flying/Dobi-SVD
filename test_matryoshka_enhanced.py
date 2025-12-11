"""
Test script for enhanced Matryoshka SVD features.

Tests:
1. Activation-aware SVD initialization
2. Learnable singular values
3. Auto rank range selection
"""

import torch
import torch.nn as nn
from modules.matryoshka_svd_enhanced import (
    MatryoshkaSVDConfig,
    MatryoshkaSVDLayer,
    replace_linear_with_matryoshka_svd
)


def test_activation_aware_svd():
    """Test 1: Activation-aware SVD initialization."""
    print("\n" + "="*80)
    print("Test 1: Activation-Aware SVD Initialization")
    print("="*80)

    # Create a simple weight matrix
    input_size, output_size = 512, 1024
    weight = torch.randn(output_size, input_size) * 0.02

    # Generate calibration data (simulating real activations)
    num_samples = 100
    calibration_data = torch.randn(num_samples, input_size)

    # Standard SVD
    config_standard = MatryoshkaSVDConfig(
        r_max=128,
        r_min=32,
        activation_aware_init=False
    )

    layer_standard = MatryoshkaSVDLayer(
        input_size=input_size,
        output_size=output_size,
        weight=weight,
        config=config_standard,
        name="test_standard"
    )

    # Activation-aware SVD
    config_aware = MatryoshkaSVDConfig(
        r_max=128,
        r_min=32,
        activation_aware_init=True
    )

    layer_aware = MatryoshkaSVDLayer(
        input_size=input_size,
        output_size=output_size,
        weight=weight.clone(),
        config=config_aware,
        calibration_data=calibration_data,
        name="test_activation_aware"
    )

    # Test forward pass
    x = torch.randn(2, 16, input_size)

    out_standard = layer_standard(x)
    out_aware = layer_aware(x)

    print(f"\nStandard SVD output shape: {out_standard.shape}")
    print(f"Activation-aware SVD output shape: {out_aware.shape}")
    print(f"Output difference norm: {(out_standard - out_aware).norm().item():.6f}")

    print("\n✓ Test 1 passed!")


def test_learnable_singular_values():
    """Test 2: Learnable singular values."""
    print("\n" + "="*80)
    print("Test 2: Learnable Singular Values")
    print("="*80)

    input_size, output_size = 256, 512
    weight = torch.randn(output_size, input_size) * 0.02

    # With learnable S
    config = MatryoshkaSVDConfig(
        r_max=64,
        r_min=16,
        learnable_singular_values=True,
        spectral_regularization_weight=0.001
    )

    layer = MatryoshkaSVDLayer(
        input_size=input_size,
        output_size=output_size,
        weight=weight,
        config=config,
        name="test_learnable_s"
    )

    # Check that log_S is a parameter
    print(f"\nLearnable parameters:")
    for name, param in layer.named_parameters():
        print(f"  {name}: {param.shape}, requires_grad={param.requires_grad}")

    # Get initial singular values
    S_init = layer.S.clone().detach()
    print(f"\nInitial S range: [{S_init.min():.6f}, {S_init.max():.6f}]")
    print(f"Initial S shape: {S_init.shape}")

    # Simulate training
    x = torch.randn(2, 16, input_size)
    target = torch.randn(2, 16, output_size)

    optimizer = torch.optim.Adam(layer.parameters(), lr=1e-3)

    print(f"\nSimulating 10 training steps...")
    for step in range(10):
        optimizer.zero_grad()

        output = layer(x)

        # Simple MSE loss
        loss = ((output - target) ** 2).mean()

        # Add regularization losses
        reg_losses = layer.get_regularization_losses()
        total_loss = loss + reg_losses['rank_reg'] + reg_losses['spectral_reg']

        total_loss.backward()
        optimizer.step()

        if step % 3 == 0:
            print(f"  Step {step}: loss={loss.item():.6f}, "
                  f"rank_reg={reg_losses['rank_reg'].item():.6f}, "
                  f"spectral_reg={reg_losses['spectral_reg'].item():.6f}")

    # Check that S changed
    S_final = layer.S.clone().detach()
    S_change = (S_final - S_init).abs().mean().item()

    print(f"\nFinal S range: [{S_final.min():.6f}, {S_final.max():.6f}]")
    print(f"Average S change: {S_change:.6f}")

    assert S_change > 0, "Singular values should have changed during training"

    print("\n✓ Test 2 passed!")


def test_auto_rank_range():
    """Test 3: Auto rank range selection."""
    print("\n" + "="*80)
    print("Test 3: Auto Rank Range Selection")
    print("="*80)

    input_size, output_size = 512, 1024

    # Create weight with specific spectral structure
    # First half of singular values are large, second half are small
    U = torch.randn(output_size, 512)
    U, _ = torch.linalg.qr(U)

    V = torch.randn(512, input_size)
    V, _ = torch.linalg.qr(V.T)
    V = V.T

    # Create singular values with clear energy drop
    S = torch.cat([
        torch.linspace(10.0, 5.0, 256),  # High energy
        torch.linspace(1.0, 0.1, 256)     # Low energy
    ])

    weight = U @ torch.diag(S) @ V

    # Manual rank range
    config_manual = MatryoshkaSVDConfig(
        r_max=256,
        r_min=64,
        auto_rank_range=False
    )

    layer_manual = MatryoshkaSVDLayer(
        input_size=input_size,
        output_size=output_size,
        weight=weight,
        config=config_manual,
        name="test_manual_range"
    )

    # Auto rank range
    config_auto = MatryoshkaSVDConfig(
        r_max=512,  # Will be adjusted
        r_min=16,   # Will be adjusted
        auto_rank_range=True,
        energy_threshold_low=0.90,   # 90% energy for r_min
        energy_threshold_high=0.99   # 99% energy for r_max
    )

    layer_auto = MatryoshkaSVDLayer(
        input_size=input_size,
        output_size=output_size,
        weight=weight.clone(),
        config=config_auto,
        name="test_auto_range"
    )

    print(f"\nManual rank range: [{layer_manual.r_min}, {layer_manual.r_max}]")
    print(f"Auto rank range:   [{layer_auto.r_min}, {layer_auto.r_max}]")

    # Verify auto range makes sense
    assert layer_auto.r_min >= 1
    assert layer_auto.r_max <= 512
    assert layer_auto.r_min < layer_auto.r_max

    print("\n✓ Test 3 passed!")


def test_model_replacement():
    """Test 4: Model replacement with enhanced features."""
    print("\n" + "="*80)
    print("Test 4: Model Replacement with Enhanced Features")
    print("="*80)

    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(256, 512)
            self.layer2 = nn.Linear(512, 256)
            self.layer3 = nn.Linear(256, 128)

        def forward(self, x):
            x = self.layer1(x)
            x = torch.relu(x)
            x = self.layer2(x)
            x = torch.relu(x)
            x = self.layer3(x)
            return x

    model = SimpleModel()

    # Replace with enhanced Matryoshka SVD
    config = MatryoshkaSVDConfig(
        r_max=128,
        r_min=32,
        learnable_singular_values=True,
        activation_aware_init=False,
        auto_rank_range=False
    )

    model = replace_linear_with_matryoshka_svd(
        model,
        target_layers=None,  # Replace all
        config=config,
        verbose=True
    )

    # Test forward pass
    x = torch.randn(2, 10, 256)
    output = model(x)

    print(f"\nModel output shape: {output.shape}")
    print(f"Expected shape: torch.Size([2, 10, 128])")

    assert output.shape == torch.Size([2, 10, 128])

    # Check that all layers were replaced
    from modules.matryoshka_svd_enhanced import MatryoshkaSVDLayer

    svd_layer_count = 0
    for module in model.modules():
        if isinstance(module, MatryoshkaSVDLayer):
            svd_layer_count += 1

    print(f"\nMatryoshkaSVD layers found: {svd_layer_count}")
    assert svd_layer_count == 3, "Should have 3 MatryoshkaSVD layers"

    print("\n✓ Test 4 passed!")


def test_gradient_flow():
    """Test 5: Gradient flow with learnable S."""
    print("\n" + "="*80)
    print("Test 5: Gradient Flow with Learnable S")
    print("="*80)

    config = MatryoshkaSVDConfig(
        r_max=64,
        r_min=16,
        learnable_singular_values=True,
        importance_strategy='learned'
    )

    layer = MatryoshkaSVDLayer(
        input_size=128,
        output_size=256,
        weight=torch.randn(256, 128) * 0.02,
        config=config,
        name="test_gradient"
    )

    # Forward pass
    x = torch.randn(2, 10, 128, requires_grad=True)
    output = layer(x)

    # Backward pass
    loss = output.sum()
    loss.backward()

    # Check gradients
    print(f"\nGradients computed:")
    print(f"  x.grad is not None: {x.grad is not None}")
    print(f"  layer.log_S.grad is not None: {layer.log_S.grad is not None}")

    for name, param in layer.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"  {name}: grad_norm={grad_norm:.6f}")

    assert x.grad is not None, "Input should have gradient"
    assert layer.log_S.grad is not None, "log_S should have gradient"

    print("\n✓ Test 5 passed!")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("Testing Enhanced Matryoshka SVD Features")
    print("="*80)

    try:
        test_activation_aware_svd()
        test_learnable_singular_values()
        test_auto_rank_range()
        test_model_replacement()
        test_gradient_flow()

        print("\n" + "="*80)
        print("✓ ALL TESTS PASSED!")
        print("="*80)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
