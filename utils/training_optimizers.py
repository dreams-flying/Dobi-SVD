"""
Training Optimizers and Utilities for Dynamic Subspace SVD

This module provides advanced training utilities including:
- Adaptive temperature scheduling
- Gamma regularization losses
- Learning rate schedulers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class TemperatureScheduler:
    """
    Adaptive temperature scheduler for routing.

    High temperature (5.0) at start → Soft routing, exploration
    Low temperature (0.5) at end → Hard routing, exploitation

    Expected benefit: +2-3% compression rate, faster convergence
    """

    def __init__(self,
                 initial_temp: float = 5.0,
                 final_temp: float = 0.5,
                 decay_steps: int = 5000,
                 decay_type: str = 'exponential'):
        """
        Args:
            initial_temp: Starting temperature (high for exploration)
            final_temp: Ending temperature (low for exploitation)
            decay_steps: Number of steps to decay from initial to final
            decay_type: 'exponential', 'linear', or 'cosine'
        """
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.decay_steps = decay_steps
        self.decay_type = decay_type

        print(f"[TemperatureScheduler] {decay_type} decay: "
              f"{initial_temp:.2f} → {final_temp:.2f} over {decay_steps} steps")

    def get_temperature(self, step: int) -> float:
        """Get temperature for current training step."""
        if step >= self.decay_steps:
            return self.final_temp

        progress = step / self.decay_steps

        if self.decay_type == 'exponential':
            # Exponential decay: smooth transition
            temperature = self.initial_temp * (self.final_temp / self.initial_temp) ** progress
        elif self.decay_type == 'linear':
            # Linear decay
            temperature = self.initial_temp + (self.final_temp - self.initial_temp) * progress
        elif self.decay_type == 'cosine':
            # Cosine decay: smooth start and end
            temperature = self.final_temp + 0.5 * (self.initial_temp - self.final_temp) * \
                         (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
        else:
            raise ValueError(f"Unknown decay_type: {self.decay_type}")

        return temperature

    def state_dict(self) -> dict:
        """Save scheduler state."""
        return {
            'initial_temp': self.initial_temp,
            'final_temp': self.final_temp,
            'decay_steps': self.decay_steps,
            'decay_type': self.decay_type
        }


class GammaRegularizer:
    """
    Regularization for gamma parameters to improve compression.

    Combines two losses:
    1. L1 loss: Encourages smaller gamma values (more compression)
    2. Diversity loss: Encourages different gamma values across subspaces

    Expected benefit: +3-5% compression rate
    """

    def __init__(self,
                 l1_weight: float = 0.01,
                 diversity_weight: float = 0.001):
        """
        Args:
            l1_weight: Weight for L1 regularization (sparsity)
            diversity_weight: Weight for diversity regularization
        """
        self.l1_weight = l1_weight
        self.diversity_weight = diversity_weight

        print(f"[GammaRegularizer] L1={l1_weight}, Diversity={diversity_weight}")

    def compute_loss(self, model: nn.Module) -> Tuple[torch.Tensor, dict]:
        """
        Compute regularization loss for all gamma parameters.

        Returns:
            loss: Total regularization loss
            stats: Dictionary with individual loss components
        """
        from modules.dynamic_subspace import (
            MultiSubspaceSVDLayer,
            SharedParamMultiSubspaceSVDLayer
        )

        l1_loss = 0.0
        diversity_loss = 0.0
        n_layers = 0

        for module in model.modules():
            if isinstance(module, (MultiSubspaceSVDLayer, SharedParamMultiSubspaceSVDLayer)):
                if not hasattr(module, 'gammas') or len(module.gammas) == 0:
                    continue

                # Stack gammas for this layer
                gammas = torch.stack([g for g in module.gammas])

                # L1 loss: Encourage low gamma values (more compression)
                l1_loss += gammas.abs().mean()

                # Diversity loss: Encourage gamma differentiation
                # Penalize small differences between gammas
                for i in range(len(gammas)):
                    for j in range(i + 1, len(gammas)):
                        diff = (gammas[i] - gammas[j]).abs()
                        # Negative sign: maximize difference
                        diversity_loss -= diff

                n_layers += 1

        if n_layers > 0:
            l1_loss = l1_loss / n_layers
            diversity_loss = diversity_loss / n_layers
        else:
            # No SVD layers found, return zero loss
            device = next(model.parameters()).device
            return torch.tensor(0.0, device=device), {'gamma_l1': 0.0, 'gamma_diversity': 0.0, 'gamma_reg_total': 0.0, 'n_layers': 0}

        # Combine losses
        total_loss = self.l1_weight * l1_loss + self.diversity_weight * diversity_loss

        # Ensure losses are tensors for proper device handling
        if not isinstance(total_loss, torch.Tensor):
            device = next(model.parameters()).device
            total_loss = torch.tensor(total_loss, device=device)

        stats = {
            'gamma_l1': l1_loss.item() if isinstance(l1_loss, torch.Tensor) else 0.0,
            'gamma_diversity': diversity_loss.item() if isinstance(diversity_loss, torch.Tensor) else 0.0,
            'gamma_reg_total': total_loss.item() if isinstance(total_loss, torch.Tensor) else 0.0,
            'n_layers': n_layers
        }

        return total_loss, stats


def setup_differentiated_optimizer(model: nn.Module,
                                   gamma_lr: float = 1e-3,
                                   other_lr: float = 1e-4,
                                   weight_decay: float = 1e-5) -> torch.optim.Optimizer:
    """
    Setup optimizer with differentiated learning rates.

    Gamma parameters get higher learning rate (10x) because:
    - They are low-dimensional (few per layer)
    - They control compression directly
    - Need faster adaptation

    Expected benefit: +30-50% faster convergence

    Args:
        model: The model to optimize
        gamma_lr: Learning rate for gamma parameters
        other_lr: Learning rate for other trainable parameters
        weight_decay: Weight decay for regularization

    Returns:
        Configured Adam optimizer
    """
    from modules.dynamic_subspace import (
        MultiSubspaceSVDLayer,
        SharedParamMultiSubspaceSVDLayer
    )

    gamma_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if 'gamma' in name.lower():
            gamma_params.append(param)
        else:
            other_params.append(param)

    print(f"[Optimizer] Gamma params: {len(gamma_params)} (lr={gamma_lr})")
    print(f"[Optimizer] Other params: {len(other_params)} (lr={other_lr})")

    # Create parameter groups with different learning rates
    param_groups = []

    if len(gamma_params) > 0:
        param_groups.append({
            'params': gamma_params,
            'lr': gamma_lr,
            'weight_decay': 0.0  # No weight decay for gamma
        })

    if len(other_params) > 0:
        param_groups.append({
            'params': other_params,
            'lr': other_lr,
            'weight_decay': weight_decay
        })

    if len(param_groups) == 0:
        raise ValueError("No trainable parameters found!")

    optimizer = torch.optim.Adam(param_groups)

    return optimizer


def get_layer_importance_weight(layer_name: str) -> float:
    """
    Get importance weight for a layer based on its position/type.

    Strategy:
    - Embedding/LM head: High importance (1.5x)
    - Attention: Medium-high importance (1.2x)
    - MLP/FC: Lower importance, more aggressive compression (0.8x)

    Expected benefit: +2-3% compression rate

    Args:
        layer_name: Name of the layer

    Returns:
        Importance weight multiplier
    """
    name_lower = layer_name.lower()

    # Embedding layers and LM head - preserve quality
    if 'embed' in name_lower or 'lm_head' in name_lower:
        return 1.5

    # Attention layers - important for model quality
    elif any(x in name_lower for x in ['attn', 'attention', 'q_proj', 'k_proj', 'v_proj', 'out_proj']):
        return 1.2

    # MLP/FC layers - can be compressed more aggressively
    elif any(x in name_lower for x in ['mlp', 'fc', 'ffn', 'feed_forward']):
        return 0.8

    # Default
    else:
        return 1.0


def compute_adaptive_gamma_multipliers(layer_name: str,
                                       base_multipliers: List[float] = [0.5, 1.0, 1.5],
                                       n_subspaces: int = 3) -> List[float]:
    """
    Compute layer-specific gamma multipliers based on importance.

    Args:
        layer_name: Name of the layer
        base_multipliers: Base multipliers for subspaces
        n_subspaces: Number of subspaces

    Returns:
        Adjusted multipliers for this layer
    """
    importance_weight = get_layer_importance_weight(layer_name)

    # Adjust base multipliers
    if importance_weight > 1.0:
        # Important layers: use higher gammas (less compression)
        adjusted = [m * 1.2 for m in base_multipliers[:n_subspaces]]
    elif importance_weight < 1.0:
        # Less important layers: use lower gammas (more compression)
        adjusted = [m * 0.8 for m in base_multipliers[:n_subspaces]]
    else:
        adjusted = base_multipliers[:n_subspaces]

    return adjusted


class GammaConvergenceMonitor:
    """
    Monitor gamma convergence and suggest early stopping.

    Tracks gamma changes over time and detects convergence.
    """

    def __init__(self,
                 patience: int = 500,
                 threshold: float = 0.01):
        """
        Args:
            patience: Number of steps to check for convergence
            threshold: Threshold for total gamma change
        """
        self.patience = patience
        self.threshold = threshold
        self.history = []

    def update(self, model: nn.Module) -> bool:
        """
        Update history and check convergence.

        Returns:
            True if converged, False otherwise
        """
        from modules.dynamic_subspace import (
            MultiSubspaceSVDLayer,
            SharedParamMultiSubspaceSVDLayer
        )

        current_gammas = []
        for module in model.modules():
            if isinstance(module, (MultiSubspaceSVDLayer, SharedParamMultiSubspaceSVDLayer)):
                if hasattr(module, 'gammas'):
                    current_gammas.extend([g.item() for g in module.gammas])

        self.history.append(current_gammas)

        # Check convergence
        if len(self.history) > self.patience:
            old_gammas = self.history[-self.patience]
            total_change = sum(abs(a - b) for a, b in zip(current_gammas, old_gammas))

            if total_change < self.threshold:
                return True  # Converged

        return False

    def get_stats(self) -> dict:
        """Get convergence statistics."""
        if len(self.history) < 2:
            return {'avg_change': float('inf'), 'history_length': len(self.history)}

        recent = self.history[-1]
        previous = self.history[-2]
        avg_change = sum(abs(a - b) for a, b in zip(recent, previous)) / len(recent)

        return {
            'avg_change': avg_change,
            'history_length': len(self.history),
            'convergence_progress': 1.0 - min(1.0, avg_change / self.threshold)
        }
