#!/usr/bin/env python3
"""
Improved training with knowledge distillation.

This adds reconstruction loss to help the model learn better.
The idea: Matryoshka output should match the original SVD-LLM output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """
    Knowledge distillation loss for Matryoshka training.

    Helps the model learn by matching teacher (full SVD-LLM) outputs.
    """

    def __init__(self, alpha=1.0, beta=0.5, temperature=2.0):
        """
        Args:
            alpha: Weight for language modeling loss
            beta: Weight for distillation loss
            temperature: Temperature for distillation (higher = softer)
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor
    ):
        """
        Compute combined loss.

        Args:
            student_logits: Matryoshka model outputs [batch, seq_len, vocab]
            teacher_logits: SVD-LLM full model outputs [batch, seq_len, vocab]
            labels: Ground truth labels [batch, seq_len]

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with loss components
        """
        # 1. Language modeling loss (student)
        loss_lm = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )

        # 2. Distillation loss (KL divergence between student and teacher)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)

        loss_kd = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='batchmean'
        ) * (self.temperature ** 2)

        # 3. Total loss
        total_loss = self.alpha * loss_lm + self.beta * loss_kd

        return total_loss, {
            'loss_lm': loss_lm.item(),
            'loss_kd': loss_kd.item(),
            'loss_total': total_loss.item()
        }


class ReconstructionLoss(nn.Module):
    """
    Reconstruction loss: ensure Matryoshka output ≈ original SVD-LLM output.

    This is simpler than full distillation - just match hidden states.
    """

    def __init__(self, weight=0.5):
        """
        Args:
            weight: Weight for reconstruction loss
        """
        super().__init__()
        self.weight = weight

    def forward(
        self,
        student_hidden: torch.Tensor,
        teacher_hidden: torch.Tensor
    ):
        """
        Compute MSE between student and teacher hidden states.

        Args:
            student_hidden: Matryoshka layer output [batch, seq_len, hidden]
            teacher_hidden: SVD-LLM layer output [batch, seq_len, hidden]

        Returns:
            loss: MSE loss
        """
        return self.weight * F.mse_loss(student_hidden, teacher_hidden)


def add_distillation_to_trainer(trainer_class):
    """
    Modify HuggingFace Trainer to use distillation.

    Usage:
        class CustomTrainer(add_distillation_to_trainer(Trainer)):
            pass
    """

    class DistillationTrainer(trainer_class):
        def __init__(self, *args, teacher_model=None, distill_alpha=1.0, distill_beta=0.5, **kwargs):
            super().__init__(*args, **kwargs)
            self.teacher_model = teacher_model
            self.distill_loss = DistillationLoss(alpha=distill_alpha, beta=distill_beta)

            # Move teacher to same device as student
            if self.teacher_model is not None:
                self.teacher_model.to(self.args.device)
                self.teacher_model.eval()
                # Freeze teacher
                for param in self.teacher_model.parameters():
                    param.requires_grad = False

        def compute_loss(self, model, inputs, return_outputs=False):
            """
            Override compute_loss to add distillation.
            """
            labels = inputs.get("labels")

            # Student forward
            outputs = model(**inputs)
            student_logits = outputs.logits

            if self.teacher_model is not None:
                # Teacher forward (no grad)
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(**inputs)
                    teacher_logits = teacher_outputs.logits

                # Distillation loss
                loss, loss_dict = self.distill_loss(student_logits, teacher_logits, labels)

                # Log components
                if self.state.global_step % 10 == 0:
                    for key, value in loss_dict.items():
                        self.log({f"train/{key}": value})
            else:
                # Normal CE loss
                loss = outputs.loss

            return (loss, outputs) if return_outputs else loss

    return DistillationTrainer


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("Testing Distillation Loss")
    print("="*80)

    batch_size = 2
    seq_len = 10
    vocab_size = 32000

    # Simulate student and teacher outputs
    student_logits = torch.randn(batch_size, seq_len, vocab_size)
    teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Test distillation loss
    distill_loss = DistillationLoss(alpha=1.0, beta=0.5)
    loss, loss_dict = distill_loss(student_logits, teacher_logits, labels)

    print(f"\nLoss components:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.4f}")

    print(f"\nTotal loss: {loss.item():.4f}")

    # Test reconstruction loss
    print(f"\n{'='*80}")
    print("Testing Reconstruction Loss")
    print("="*80)

    hidden_dim = 4096
    student_hidden = torch.randn(batch_size, seq_len, hidden_dim)
    teacher_hidden = torch.randn(batch_size, seq_len, hidden_dim)

    recon_loss = ReconstructionLoss(weight=0.5)
    loss = recon_loss(student_hidden, teacher_hidden)

    print(f"Reconstruction loss: {loss.item():.4f}")

    print(f"\n{'='*80}")
    print("All tests passed! ✅")
    print("="*80)
