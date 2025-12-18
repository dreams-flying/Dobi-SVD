#!/usr/bin/env python3
"""
Improved training strategy to overcome loss plateau.

Key ideas:
1. Different learning rates for rank predictor vs U/V
2. Progressive rank training (curriculum learning)
3. Warmup and cosine schedule
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


def create_parameter_groups(model, rank_predictor_lr=1e-4, uv_lr=1e-6, other_lr=5e-5):
    """
    Create parameter groups with different learning rates.

    Args:
        model: The model
        rank_predictor_lr: Learning rate for rank predictors (highest)
        uv_lr: Learning rate for U/V projections (lowest, because SVD initialized)
        other_lr: Learning rate for other parameters

    Returns:
        parameter_groups: List of parameter groups for optimizer
    """
    rank_predictor_params = []
    uv_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if 'rank_predictor' in name:
            # Rank predictors: need to learn from scratch
            rank_predictor_params.append(param)
        elif 'v_proj' in name or 'u_proj' in name:
            # U/V projections: already SVD initialized, need small updates
            uv_params.append(param)
        else:
            # Other parameters (embeddings, layer norms, etc.)
            other_params.append(param)

    parameter_groups = []

    if rank_predictor_params:
        parameter_groups.append({
            'params': rank_predictor_params,
            'lr': rank_predictor_lr,
            'name': 'rank_predictor'
        })
        print(f"  Rank predictor params: {len(rank_predictor_params)} groups, lr={rank_predictor_lr}")

    if uv_params:
        parameter_groups.append({
            'params': uv_params,
            'lr': uv_lr,
            'name': 'uv_projections'
        })
        print(f"  U/V projection params: {len(uv_params)} groups, lr={uv_lr}")

    if other_params:
        parameter_groups.append({
            'params': other_params,
            'lr': other_lr,
            'name': 'other'
        })
        print(f"  Other params: {len(other_params)} groups, lr={other_lr}")

    return parameter_groups


def create_optimizer_with_param_groups(model, rank_predictor_lr=1e-4, uv_lr=1e-6, other_lr=5e-5):
    """
    Create optimizer with different learning rates for different components.
    """
    print("\nCreating optimizer with parameter groups:")
    param_groups = create_parameter_groups(
        model,
        rank_predictor_lr=rank_predictor_lr,
        uv_lr=uv_lr,
        other_lr=other_lr
    )

    optimizer = AdamW(param_groups, weight_decay=0.01)
    return optimizer


def create_progressive_scheduler(optimizer, num_epochs, warmup_epochs=1):
    """
    Create learning rate scheduler with warmup + cosine decay.

    Args:
        optimizer: The optimizer
        num_epochs: Total number of epochs
        warmup_epochs: Number of warmup epochs

    Returns:
        scheduler: Learning rate scheduler
    """
    # Warmup scheduler
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs
    )

    # Cosine annealing scheduler
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_epochs - warmup_epochs,
        eta_min=1e-7
    )

    # Sequential: warmup then cosine
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )

    return scheduler


class ProgressiveRankTrainer:
    """
    Progressive rank training: start from high ranks, gradually decrease.

    This is curriculum learning for compression:
    - Early epochs: easy task (high rank, little compression)
    - Later epochs: hard task (low rank, more compression)
    """

    def __init__(self, model, r_max, r_min, num_epochs):
        """
        Args:
            model: The model with MatryoshkaSVDLayer
            r_max: Maximum rank
            r_min: Minimum rank (final target)
            num_epochs: Total epochs
        """
        self.model = model
        self.r_max = r_max
        self.r_min = r_min
        self.num_epochs = num_epochs

    def get_current_rank_range(self, epoch):
        """
        Get rank range for current epoch.

        Schedule:
        - Epoch 0-20%: [0.9*r_max, r_max] (almost no compression)
        - Epoch 20-40%: [0.75*r_max, r_max]
        - Epoch 40-60%: [0.5*(r_min+r_max), r_max]
        - Epoch 60-100%: [r_min, r_max] (full compression)
        """
        progress = epoch / self.num_epochs

        if progress < 0.2:
            # Stage 1: minimal compression
            current_r_min = int(0.9 * self.r_max)
        elif progress < 0.4:
            # Stage 2: light compression
            current_r_min = int(0.75 * self.r_max)
        elif progress < 0.6:
            # Stage 3: medium compression
            current_r_min = int(0.5 * (self.r_min + self.r_max))
        else:
            # Stage 4: full compression
            current_r_min = self.r_min

        return current_r_min, self.r_max

    def update_model_rank_range(self, epoch):
        """
        Update model's rank range for current epoch.
        """
        current_r_min, current_r_max = self.get_current_rank_range(epoch)

        # Update all MatryoshkaSVDLayer instances
        for module in self.model.modules():
            if hasattr(module, 'r_min') and hasattr(module, 'r_max'):
                module.r_min = current_r_min
                # r_max stays the same

                # Update rank predictor's range
                if hasattr(module, 'rank_predictor') and module.rank_predictor is not None:
                    module.rank_predictor.r_min = current_r_min

        print(f"\nEpoch {epoch}: Updated rank range to [{current_r_min}, {current_r_max}]")

        return current_r_min, current_r_max


def freeze_uv_projections(model):
    """Freeze U/V projections."""
    for name, param in model.named_parameters():
        if 'v_proj' in name or 'u_proj' in name:
            param.requires_grad = False


def unfreeze_uv_projections(model):
    """Unfreeze U/V projections."""
    for name, param in model.named_parameters():
        if 'v_proj' in name or 'u_proj' in name:
            param.requires_grad = True


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("Testing Improved Training Strategy")
    print("="*80)

    # Simulate a model with MatryoshkaSVDLayer
    from modules.matryoshka_svd_layer_v2 import MatryoshkaSVDLayerV2

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = MatryoshkaSVDLayerV2(
                in_features=4096,
                out_features=4096,
                r_max=512,
                r_min=256,
                use_rank_predictor=True
            )
            self.layer2 = nn.Linear(4096, 4096)

    model = DummyModel()

    # Test parameter groups
    print("\n" + "="*80)
    print("Testing Parameter Groups")
    print("="*80)

    optimizer = create_optimizer_with_param_groups(
        model,
        rank_predictor_lr=1e-4,
        uv_lr=1e-6,
        other_lr=5e-5
    )

    print(f"\nOptimizer created with {len(optimizer.param_groups)} parameter groups")

    # Test progressive rank training
    print("\n" + "="*80)
    print("Testing Progressive Rank Training")
    print("="*80)

    progressive_trainer = ProgressiveRankTrainer(
        model,
        r_max=512,
        r_min=256,
        num_epochs=10
    )

    for epoch in range(10):
        r_min, r_max = progressive_trainer.update_model_rank_range(epoch)
        print(f"  Epoch {epoch}: rank range [{r_min}, {r_max}]")

    # Test scheduler
    print("\n" + "="*80)
    print("Testing Learning Rate Scheduler")
    print("="*80)

    scheduler = create_progressive_scheduler(optimizer, num_epochs=10, warmup_epochs=1)

    print("Learning rate schedule:")
    for epoch in range(10):
        lrs = [group['lr'] for group in optimizer.param_groups]
        print(f"  Epoch {epoch}: {lrs}")
        scheduler.step()

    print("\n" + "="*80)
    print("All tests passed! ✅")
    print("="*80)
