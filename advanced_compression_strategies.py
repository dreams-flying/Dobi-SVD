#!/usr/bin/env python3
"""
Advanced compression strategies WITHOUT knowledge distillation.

These methods improve Matryoshka SVD compression performance:
1. Self-distillation (no teacher needed)
2. Two-stage training
3. Layer-wise adaptive rank
4. Improved SVD initialization
5. Reconstruction loss
6. Learned temperature scheduling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from transformers import Trainer


# ============================================================================
# Strategy 1: Self-Distillation (Full Rank as Teacher)
# ============================================================================

class SelfDistillationLoss(nn.Module):
    """
    Use model's own full-rank output as teacher.
    No extra memory needed - just do two forward passes.

    Concept:
    - Forward with full rank (r=r_max) → teacher logits
    - Forward with predicted rank → student logits
    - Minimize KL divergence

    Benefits:
    - No teacher model needed
    - Teaches predictor when to compress
    - Better than plain LM loss
    """

    def __init__(
        self,
        temperature: float = 2.0,
        alpha: float = 1.0,  # LM loss weight
        beta: float = 0.3,   # Self-distill loss weight
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta

    def forward(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            model: The Matryoshka model
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels for LM loss

        Returns:
            total_loss: Combined loss
            loss_dict: Individual loss components
        """
        # 1. Forward with full rank (teacher)
        self._set_model_rank(model, mode='full')
        with torch.no_grad():
            teacher_outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            teacher_logits = teacher_outputs.logits

        # 2. Forward with predicted rank (student)
        self._set_model_rank(model, mode='adaptive')
        student_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        student_logits = student_outputs.logits

        # 3. Compute losses
        # 3a. LM loss (from student outputs)
        if labels is not None:
            loss_lm = student_outputs.loss
        else:
            loss_lm = torch.tensor(0.0, device=student_logits.device)

        # 3b. Self-distillation loss (KL divergence)
        student_log_probs = F.log_softmax(
            student_logits / self.temperature, dim=-1
        )
        teacher_probs = F.softmax(
            teacher_logits / self.temperature, dim=-1
        )

        loss_kd = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='batchmean'
        ) * (self.temperature ** 2)

        # 4. Combine
        total_loss = self.alpha * loss_lm + self.beta * loss_kd

        loss_dict = {
            'loss_lm': loss_lm.item(),
            'loss_kd': loss_kd.item(),
            'loss_total': total_loss.item(),
        }

        return total_loss, loss_dict

    def _set_model_rank(self, model: nn.Module, mode: str):
        """Set all matryoshka layers to full or adaptive rank."""
        for module in model.modules():
            if hasattr(module, 'fixed_rank'):
                if mode == 'full':
                    # Force full rank
                    module.fixed_rank = module.r_max
                elif mode == 'adaptive':
                    # Use rank predictor
                    module.fixed_rank = None


# ============================================================================
# Strategy 2: Two-Stage Training
# ============================================================================

class TwoStageTrainer:
    """
    Stage 1: Train rank predictor only (freeze U/V)
    Stage 2: Joint training with small U/V learning rate

    Benefits:
    - Predictor learns good rank selection first
    - U/V adapts slowly to predictor's choices
    - Prevents U/V from fighting predictor
    """

    def __init__(
        self,
        model: nn.Module,
        stage1_epochs: int = 2,
        stage2_epochs: int = 3,
        stage1_predictor_lr: float = 2e-4,
        stage2_predictor_lr: float = 1e-4,
        stage2_uv_lr: float = 1e-6,
        stage2_other_lr: float = 5e-5,
    ):
        self.model = model
        self.stage1_epochs = stage1_epochs
        self.stage2_epochs = stage2_epochs
        self.stage1_predictor_lr = stage1_predictor_lr
        self.stage2_predictor_lr = stage2_predictor_lr
        self.stage2_uv_lr = stage2_uv_lr
        self.stage2_other_lr = stage2_other_lr

    def get_stage1_optimizer(self) -> torch.optim.Optimizer:
        """Stage 1: Only train rank predictor."""
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze rank predictor
        predictor_params = []
        for name, param in self.model.named_parameters():
            if 'rank_predictor' in name:
                param.requires_grad = True
                predictor_params.append(param)

        print(f"Stage 1: Training {len(predictor_params)} rank predictor parameters")

        optimizer = torch.optim.AdamW(
            predictor_params,
            lr=self.stage1_predictor_lr,
            weight_decay=0.01
        )

        return optimizer

    def get_stage2_optimizer(self) -> torch.optim.Optimizer:
        """Stage 2: Train all parameters with different LRs."""
        # Unfreeze all parameters
        for param in self.model.parameters():
            param.requires_grad = True

        # Create parameter groups
        rank_predictor_params = []
        uv_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if 'rank_predictor' in name:
                rank_predictor_params.append(param)
            elif 'v_proj' in name or 'u_proj' in name:
                uv_params.append(param)
            else:
                other_params.append(param)

        print(f"Stage 2: Training all parameters")
        print(f"  Rank predictor: {len(rank_predictor_params)} params, lr={self.stage2_predictor_lr:.0e}")
        print(f"  U/V projections: {len(uv_params)} params, lr={self.stage2_uv_lr:.0e}")
        print(f"  Other: {len(other_params)} params, lr={self.stage2_other_lr:.0e}")

        optimizer = torch.optim.AdamW([
            {'params': rank_predictor_params, 'lr': self.stage2_predictor_lr, 'name': 'rank_predictor'},
            {'params': uv_params, 'lr': self.stage2_uv_lr, 'name': 'uv_projections'},
            {'params': other_params, 'lr': self.stage2_other_lr, 'name': 'other'},
        ], weight_decay=0.01)

        return optimizer


# ============================================================================
# Strategy 3: Layer-wise Adaptive Rank
# ============================================================================

class LayerWiseRankConfig:
    """
    Different layers have different importance.
    Use larger ranks for important layers.

    Heuristic:
    - Early layers (input): Medium rank (need to preserve input info)
    - Middle layers (reasoning): High rank (most important)
    - Late layers (output): Medium rank (output is simpler)
    """

    def __init__(
        self,
        num_layers: int,
        global_r_min: int = 256,
        global_r_max: int = 512,
    ):
        self.num_layers = num_layers
        self.global_r_min = global_r_min
        self.global_r_max = global_r_max

    def get_layer_rank_range(self, layer_idx: int) -> Tuple[int, int]:
        """Get r_min, r_max for a specific layer."""
        progress = layer_idx / self.num_layers

        # Importance curve (higher in middle)
        if progress < 0.2:
            # Early layers: 75% capacity
            importance = 0.75
        elif progress < 0.4:
            # Early-middle: 90% capacity
            importance = 0.90
        elif progress < 0.7:
            # Middle layers: 100% capacity (most important)
            importance = 1.0
        elif progress < 0.85:
            # Late-middle: 90% capacity
            importance = 0.90
        else:
            # Late layers: 80% capacity
            importance = 0.80

        # Scale r_min, r_max based on importance
        r_min = int(self.global_r_min * importance)
        r_max = int(self.global_r_max * importance)

        # Ensure valid range
        r_min = max(32, r_min)
        r_max = max(r_min + 32, r_max)

        return r_min, r_max

    def apply_to_model(self, model: nn.Module):
        """Apply layer-wise rank configuration to model."""
        layer_idx = 0

        for name, module in model.named_modules():
            # Find matryoshka layers
            if hasattr(module, 'rank_predictor') and hasattr(module, 'r_min'):
                r_min, r_max = self.get_layer_rank_range(layer_idx)

                module.r_min = r_min
                module.r_max = r_max

                if hasattr(module.rank_predictor, 'r_min'):
                    module.rank_predictor.r_min = r_min
                    module.rank_predictor.r_max = r_max

                print(f"Layer {layer_idx} ({name}): rank range [{r_min}, {r_max}]")

                layer_idx += 1

        print(f"\n✅ Applied layer-wise rank config to {layer_idx} layers")


# ============================================================================
# Strategy 4: Reconstruction Loss (Activation Matching)
# ============================================================================

class ActivationReconstructionLoss(nn.Module):
    """
    Match intermediate activations between full-rank and compressed.

    This is like distillation but only on hidden states, not logits.
    Uses less memory than full distillation.
    """

    def __init__(
        self,
        reconstruction_weight: float = 0.1,
        match_layers: Optional[List[int]] = None,  # Which layers to match
    ):
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.match_layers = match_layers  # e.g., [8, 16, 24] for 32-layer model

        # Storage for activations
        self.teacher_activations = {}
        self.student_activations = {}

    def _register_hooks(self, model: nn.Module, mode: str):
        """Register forward hooks to capture activations."""
        hooks = []
        layer_idx = 0

        for name, module in model.named_modules():
            # Hook on attention output or MLP output
            if 'self_attn' in name or 'mlp' in name:
                if self.match_layers is None or layer_idx in self.match_layers:

                    def hook_fn(module, input, output, name=name, mode=mode):
                        if mode == 'teacher':
                            self.teacher_activations[name] = output.detach()
                        else:
                            self.student_activations[name] = output

                    hook = module.register_forward_hook(hook_fn)
                    hooks.append(hook)

                layer_idx += 1

        return hooks

    def forward(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute reconstruction loss.
        """
        self.teacher_activations = {}
        self.student_activations = {}

        # 1. Forward with full rank (teacher)
        self._set_model_rank(model, mode='full')
        teacher_hooks = self._register_hooks(model, mode='teacher')

        with torch.no_grad():
            teacher_outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # Remove hooks
        for hook in teacher_hooks:
            hook.remove()

        # 2. Forward with predicted rank (student)
        self._set_model_rank(model, mode='adaptive')
        student_hooks = self._register_hooks(model, mode='student')

        student_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        # Remove hooks
        for hook in student_hooks:
            hook.remove()

        # 3. Compute reconstruction loss
        recon_loss = 0.0
        num_matches = 0

        for name in self.teacher_activations.keys():
            if name in self.student_activations:
                teacher_act = self.teacher_activations[name]
                student_act = self.student_activations[name]

                # MSE loss on activations
                recon_loss += F.mse_loss(student_act, teacher_act)
                num_matches += 1

        if num_matches > 0:
            recon_loss = recon_loss / num_matches

        # 4. Combine with LM loss
        lm_loss = student_outputs.loss if labels is not None else torch.tensor(0.0)
        total_loss = lm_loss + self.reconstruction_weight * recon_loss

        loss_dict = {
            'loss_lm': lm_loss.item() if isinstance(lm_loss, torch.Tensor) else lm_loss,
            'loss_recon': recon_loss.item() if isinstance(recon_loss, torch.Tensor) else recon_loss,
            'loss_total': total_loss.item(),
            'num_matched_layers': num_matches,
        }

        return total_loss, loss_dict

    def _set_model_rank(self, model: nn.Module, mode: str):
        """Set all matryoshka layers to full or adaptive rank."""
        for module in model.modules():
            if hasattr(module, 'fixed_rank'):
                if mode == 'full':
                    module.fixed_rank = module.r_max
                elif mode == 'adaptive':
                    module.fixed_rank = None


# ============================================================================
# Strategy 5: Learned Temperature Scheduling
# ============================================================================

class LearnedTemperatureScheduler:
    """
    Instead of fixed gating_tau, learn it during training.

    Start with high tau (soft gating) → gradually decrease (harder gating)
    This gives better gradients early, sharper decisions later.
    """

    def __init__(
        self,
        initial_tau: float = 5.0,
        final_tau: float = 0.5,
        num_epochs: int = 5,
        schedule_type: str = 'cosine',  # 'linear', 'cosine', 'exponential'
    ):
        self.initial_tau = initial_tau
        self.final_tau = final_tau
        self.num_epochs = num_epochs
        self.schedule_type = schedule_type

    def get_tau(self, epoch: float) -> float:
        """Get temperature for current epoch."""
        progress = epoch / self.num_epochs
        progress = min(1.0, max(0.0, progress))

        if self.schedule_type == 'linear':
            tau = self.initial_tau + (self.final_tau - self.initial_tau) * progress

        elif self.schedule_type == 'cosine':
            import math
            tau = self.final_tau + (self.initial_tau - self.final_tau) * \
                  (1 + math.cos(math.pi * progress)) / 2

        elif self.schedule_type == 'exponential':
            import math
            tau = self.initial_tau * (self.final_tau / self.initial_tau) ** progress

        else:
            tau = self.initial_tau

        return tau

    def update_model_temperature(self, model: nn.Module, epoch: float):
        """Update gating_tau for all matryoshka layers."""
        tau = self.get_tau(epoch)

        updated = 0
        for module in model.modules():
            if hasattr(module, 'gating_tau'):
                module.gating_tau = tau
                updated += 1

        return tau, updated


# ============================================================================
# Helper: Create Integrated Trainer
# ============================================================================

def create_advanced_trainer(
    base_trainer_class,
    strategy: str = 'self_distill',  # 'self_distill', 'recon', or 'both'
    **strategy_kwargs
):
    """
    Create trainer with advanced compression strategies.

    Args:
        base_trainer_class: HuggingFace Trainer class
        strategy: 'self_distill', 'recon', or 'both'
        strategy_kwargs: Arguments for the chosen strategy

    Returns:
        Advanced trainer class
    """

    class AdvancedCompressionTrainer(base_trainer_class):
        def __init__(self, *args, compression_strategy=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.compression_strategy = compression_strategy

        def compute_loss(self, model, inputs, return_outputs=False):
            """Override compute_loss to use advanced strategy."""

            if self.compression_strategy is None:
                # Standard training
                outputs = model(**inputs)
                loss = outputs.loss
                return (loss, outputs) if return_outputs else loss

            # Use advanced strategy
            input_ids = inputs.get('input_ids')
            attention_mask = inputs.get('attention_mask')
            labels = inputs.get('labels')

            loss, loss_dict = self.compression_strategy(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            # Log individual losses
            if self.state.global_step % self.args.logging_steps == 0:
                for k, v in loss_dict.items():
                    self.log({k: v})

            return loss

    return AdvancedCompressionTrainer


# ============================================================================
# Main: Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Advanced Compression Strategies (No Distillation)")
    print("=" * 80)
    print("\nAvailable strategies:")
    print("1. Self-Distillation: Use full-rank as teacher (no extra model)")
    print("2. Two-Stage Training: Train predictor first, then joint")
    print("3. Layer-wise Adaptive Rank: Important layers get more capacity")
    print("4. Reconstruction Loss: Match activations (lighter than distillation)")
    print("5. Learned Temperature: Better gradient flow")
    print("\nSee documentation in each class for details.")
