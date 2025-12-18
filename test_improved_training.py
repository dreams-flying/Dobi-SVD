#!/usr/bin/env python3
"""
Quick test to verify improved training strategy works before full training.

This script:
1. Tests parameter group creation
2. Tests progressive rank trainer
3. Tests distillation loss (if teacher available)
4. Runs 1 training step to verify everything works

Run this BEFORE starting full training to catch issues early.
"""

import torch
import torch.nn as nn
from improved_training_strategy import (
    create_optimizer_with_param_groups,
    ProgressiveRankTrainer
)


def test_parameter_groups():
    """Test parameter group creation."""
    print("="*80)
    print("Test 1: Parameter Group Creation")
    print("="*80)

    # Create a simple model with matryoshka-like structure
    class SimpleMatryoshkaModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.rank_predictor = nn.Linear(10, 1)
            self.v_proj = nn.Linear(10, 5)
            self.u_proj = nn.Linear(5, 10)
            self.other_layer = nn.Linear(10, 10)

        def forward(self, x):
            return self.other_layer(self.u_proj(self.v_proj(x)))

    model = SimpleMatryoshkaModel()

    # Create optimizer with parameter groups
    try:
        optimizer = create_optimizer_with_param_groups(
            model,
            rank_predictor_lr=1e-4,
            uv_lr=1e-6,
            other_lr=5e-5
        )

        # Check param groups
        print(f"\n✅ Created optimizer with {len(optimizer.param_groups)} parameter groups:")
        for i, group in enumerate(optimizer.param_groups):
            name = group.get('name', f'group_{i}')
            lr = group['lr']
            num_params = len(group['params'])
            print(f"   {i+1}. {name}: lr={lr:.0e}, {num_params} param tensors")

        # Verify learning rates
        rank_pred_lr = optimizer.param_groups[0]['lr']
        uv_lr = optimizer.param_groups[1]['lr']
        other_lr = optimizer.param_groups[2]['lr']

        assert rank_pred_lr == 1e-4, f"Expected rank_predictor lr=1e-4, got {rank_pred_lr}"
        assert uv_lr == 1e-6, f"Expected uv lr=1e-6, got {uv_lr}"
        assert other_lr == 5e-5, f"Expected other lr=5e-5, got {other_lr}"

        print(f"\n✅ Learning rates verified:")
        print(f"   Rank predictor: {rank_pred_lr:.0e} (large - learn fast)")
        print(f"   U/V projections: {uv_lr:.0e} (small - fine-tune)")
        print(f"   Other params: {other_lr:.0e} (medium)")

        # Test one step
        x = torch.randn(2, 10)
        optimizer.zero_grad()
        output = model(x)
        loss = output.mean()
        loss.backward()
        optimizer.step()

        print(f"\n✅ Test training step successful")
        print(f"   Loss: {loss.item():.6f}")

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


def test_progressive_rank():
    """Test progressive rank trainer."""
    print("\n" + "="*80)
    print("Test 2: Progressive Rank Trainer")
    print("="*80)

    # Create a mock model
    class MockMatryoshkaLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.r_min = 256
            self.r_max = 512
            self.rank_predictor = nn.Module()
            self.rank_predictor.r_min = self.r_min
            self.rank_predictor.r_max = self.r_max

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([MockMatryoshkaLayer() for _ in range(3)])

    model = MockModel()

    try:
        progressive_trainer = ProgressiveRankTrainer(
            model=model,
            r_max=512,
            r_min=256,
            num_epochs=5
        )

        print(f"\n✅ Created progressive rank trainer")
        print(f"   Target: [{progressive_trainer.r_min}, {progressive_trainer.r_max}]")
        print(f"   Num epochs: {progressive_trainer.num_epochs}")

        # Test schedule
        print(f"\n📊 Rank schedule:")
        for epoch in range(5):
            r_min, r_max = progressive_trainer.get_current_rank_range(epoch)
            print(f"   Epoch {epoch}: rank range [{r_min}, {r_max}]")

        # Verify curriculum (should start high, end low)
        r_min_start, _ = progressive_trainer.get_current_rank_range(0)
        r_min_end, _ = progressive_trainer.get_current_rank_range(4)

        assert r_min_start > r_min_end, "Should start with higher r_min (easier)"
        assert r_min_end == 256, "Should end at target r_min"

        print(f"\n✅ Curriculum verified:")
        print(f"   Starts easy: r_min={r_min_start} (high rank)")
        print(f"   Ends hard: r_min={r_min_end} (low rank)")

        # Test update
        progressive_trainer.update_model_rank_range(0)
        print(f"\n✅ Model rank range updated successfully")
        print(f"   Layer 0 r_min: {model.layers[0].r_min}")

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_distillation_loss():
    """Test distillation loss (basic check)."""
    print("\n" + "="*80)
    print("Test 3: Distillation Loss (Optional)")
    print("="*80)

    try:
        from distillation_loss import DistillationLoss

        distill_loss_fn = DistillationLoss(
            temperature=2.0,
            alpha=1.0,
            beta=0.5
        )

        # Create dummy logits (with requires_grad for backward test)
        batch_size = 2
        seq_len = 10
        vocab_size = 100

        student_logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Compute loss
        total_loss, loss_dict = distill_loss_fn(student_logits, teacher_logits, labels)

        print(f"\n✅ Distillation loss computed successfully:")
        print(f"   Total loss: {total_loss.item():.6f}")
        print(f"   LM loss: {loss_dict['loss_lm']:.6f}")
        print(f"   KD loss: {loss_dict['loss_kd']:.6f}")

        # Verify loss is differentiable
        total_loss.backward()
        print(f"\n✅ Loss is differentiable (backward pass successful)")
        print(f"   Student logits grad shape: {student_logits.grad.shape}")

        return True

    except ImportError:
        print(f"\n⚠️  distillation_loss.py not found - skipping")
        print(f"   (This is optional, not needed for parameter groups + progressive rank)")
        return None

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*80)
    print("Testing Improved Training Strategy")
    print("="*80)
    print("\nThis will verify all components before full training")
    print("="*80 + "\n")

    results = {}

    # Test 1: Parameter groups
    results['param_groups'] = test_parameter_groups()

    # Test 2: Progressive rank
    results['progressive_rank'] = test_progressive_rank()

    # Test 3: Distillation (optional)
    results['distillation'] = test_distillation_loss()

    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80 + "\n")

    all_passed = True
    for name, result in results.items():
        if result is True:
            print(f"✅ {name}: PASSED")
        elif result is None:
            print(f"⚠️  {name}: SKIPPED (optional)")
        else:
            print(f"❌ {name}: FAILED")
            all_passed = False

    print("\n" + "="*80)

    if all_passed:
        print("✅ All tests passed!")
        print("\nYou are ready to run full training:")
        print("  ./run_improved_training.sh")
        print("\nOr:")
        print("  python train_with_improved_strategy.py --matryoshka_model_path <path>")
    else:
        print("❌ Some tests failed!")
        print("\nPlease fix errors before running full training.")

    print("="*80 + "\n")

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
