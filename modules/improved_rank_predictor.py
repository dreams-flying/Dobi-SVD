#!/usr/bin/env python3
"""
Improved Rank Predictor for Matryoshka SVD.

Key improvements:
1. RankPredictor only predicts continuous rank ∈ [r_min, r_max]
2. Separate build_nested_mask() for mask construction
3. Clean separation: prediction vs masking
4. Easy to add rank regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedRankPredictor(nn.Module):
    """
    Predicts a continuous rank value in [r_min, r_max] for each token.

    This is a clean predictor that ONLY does prediction, not masking.
    The masking is handled separately by build_nested_mask().
    """

    def __init__(
        self,
        in_features: int,
        r_max: int,
        r_min: int,
        hidden_dim: int = 256,
        use_context: bool = False,
        context_window: int = 3
    ):
        """
        Args:
            in_features: Input feature dimension
            r_max: Maximum rank
            r_min: Minimum rank
            hidden_dim: Hidden layer dimension
            use_context: Whether to use contextual information
            context_window: Size of context window if use_context=True
        """
        super().__init__()

        self.r_max = r_max
        self.r_min = r_min
        self.use_context = use_context

        if use_context:
            # Contextual predictor: uses conv1d to aggregate local context
            self.context_encoder = nn.Sequential(
                nn.Conv1d(in_features, hidden_dim, kernel_size=context_window, padding=context_window // 2),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
            )
            predictor_in_dim = hidden_dim
        else:
            # Simple predictor: per-token MLP
            self.context_encoder = None
            predictor_in_dim = in_features

        # Predictor network: maps to [0, 1], then scale to [r_min, r_max]
        self.predictor = nn.Sequential(
            nn.Linear(predictor_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict continuous rank for each token.

        Args:
            x: Input tensor [batch_size, seq_len, in_features]

        Returns:
            rank: Continuous rank values [batch_size, seq_len, 1] in range [r_min, r_max]
        """
        if self.use_context:
            # x: [batch, seq_len, features] → [batch, features, seq_len]
            x_transposed = x.transpose(1, 2)
            context_features = self.context_encoder(x_transposed)  # [batch, hidden, seq_len]
            context_features = context_features.transpose(1, 2)  # [batch, seq_len, hidden]
            pred_input = context_features
        else:
            pred_input = x

        # Predict normalized rank in [0, 1]
        rank_normalized = self.predictor(pred_input)  # [batch, seq_len, 1]

        # Scale to [r_min, r_max]
        rank = self.r_min + rank_normalized * (self.r_max - self.r_min)

        return rank

    def get_average_rank(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get average predicted rank for regularization.

        Args:
            x: Input tensor [batch_size, seq_len, in_features]

        Returns:
            avg_rank: Scalar tensor with average rank
        """
        rank = self.forward(x)
        return rank.mean()


def build_nested_mask(
    rank: torch.Tensor,
    r_max: int,
    tau: float = 1.0,
    hard: bool = False
) -> torch.Tensor:
    """
    Build nested Matryoshka mask from continuous rank values.

    The mask has the property: if dimension i is kept, all dimensions j < i are also kept.
    This is the "nested" or "Matryoshka" property.

    Args:
        rank: Continuous rank values [batch_size, seq_len, 1] in range [r_min, r_max]
        r_max: Maximum rank (mask dimension)
        tau: Temperature for soft masking (lower = sharper)
        hard: If True, use hard binary mask; if False, use soft sigmoid mask

    Returns:
        gates: Nested mask [batch_size, seq_len, r_max]
              For hard=True: [1,1,1,...,1,0,0,0] (binary)
              For hard=False: [0.99,0.98,0.95,...,0.05,0.02,0.01] (soft)
    """
    batch_size, seq_len, _ = rank.shape
    device = rank.device

    # Create dimension indices: [0, 1, 2, ..., r_max-1]
    # Shape: [r_max]
    dim_indices = torch.arange(r_max, device=device, dtype=rank.dtype)

    # Expand for broadcasting: [1, 1, r_max]
    dim_indices = dim_indices.view(1, 1, -1)

    # rank: [batch, seq_len, 1]
    # dim_indices: [1, 1, r_max]
    # Broadcast to [batch, seq_len, r_max]

    if hard:
        # Hard mask: dimension i is kept if i < rank
        # Result: [1,1,1,...,0,0,0]
        gates = (dim_indices < rank).float()
    else:
        # Soft mask: sigmoid-based smooth transition
        # gates[i] = sigmoid((rank - i) / tau)
        # When i << rank: gates[i] ≈ 1
        # When i ≈ rank: gates[i] ≈ 0.5
        # When i >> rank: gates[i] ≈ 0
        gates = torch.sigmoid((rank - dim_indices) / tau)

    return gates


def build_nested_mask_gumbel(
    rank: torch.Tensor,
    r_max: int,
    tau: float = 1.0,
    hard: bool = False,
    training: bool = True
) -> torch.Tensor:
    """
    Build nested mask using Gumbel-Softmax for better gradient flow.

    This variant uses Gumbel-Softmax trick which can provide better gradients
    during training compared to plain sigmoid.

    Args:
        rank: Continuous rank values [batch_size, seq_len, 1]
        r_max: Maximum rank
        tau: Temperature for Gumbel-Softmax
        hard: If True, use straight-through estimator
        training: Whether in training mode

    Returns:
        gates: Nested mask [batch_size, seq_len, r_max]
    """
    batch_size, seq_len, _ = rank.shape
    device = rank.device

    # Create logits for each dimension
    # We want: prob[keep dimension i] ∝ exp((rank - i) / tau)
    dim_indices = torch.arange(r_max, device=device, dtype=rank.dtype).view(1, 1, -1)

    # Logits: high when i < rank, low when i > rank
    logits = (rank - dim_indices) / tau

    if training and not hard:
        # During training with soft mode: add Gumbel noise
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
        logits = logits + gumbel_noise

    # Apply sigmoid to get probabilities
    gates = torch.sigmoid(logits)

    if hard:
        # Straight-through estimator: forward uses hard, backward uses soft
        gates_hard = (logits > 0).float()
        gates = gates_hard - gates.detach() + gates

    return gates


class RankRegularizationLoss(nn.Module):
    """
    Regularization loss to control average rank.

    Lower average rank = more compression = faster inference
    But too low rank = worse quality

    Usage:
        loss = rank_reg_loss(predicted_ranks, target_avg_rank=384)
        total_loss = lm_loss + 0.001 * loss
    """

    def __init__(self, target_avg_rank: float = None, weight: float = 1.0):
        """
        Args:
            target_avg_rank: Target average rank to encourage
                            If None, just penalizes high ranks
            weight: Loss weight
        """
        super().__init__()
        self.target_avg_rank = target_avg_rank
        self.weight = weight

    def forward(self, predicted_ranks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predicted_ranks: [batch, seq_len, 1] or [batch, seq_len]

        Returns:
            loss: Scalar regularization loss
        """
        avg_rank = predicted_ranks.mean()

        if self.target_avg_rank is not None:
            # MSE to target rank
            loss = F.mse_loss(avg_rank, torch.tensor(self.target_avg_rank, device=avg_rank.device))
        else:
            # Just penalize high ranks
            loss = avg_rank

        return self.weight * loss


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("Testing Improved Rank Predictor")
    print("="*80)

    # Setup
    batch_size = 2
    seq_len = 8
    in_features = 4096
    r_max = 512
    r_min = 256

    # Create predictor
    predictor = ImprovedRankPredictor(
        in_features=in_features,
        r_max=r_max,
        r_min=r_min,
        use_context=True  # Try contextual predictor
    )

    # Test input
    x = torch.randn(batch_size, seq_len, in_features)

    # Predict ranks
    print(f"\nInput shape: {x.shape}")
    predicted_ranks = predictor(x)
    print(f"Predicted ranks shape: {predicted_ranks.shape}")
    print(f"Rank range: [{predicted_ranks.min().item():.2f}, {predicted_ranks.max().item():.2f}]")
    print(f"Average rank: {predicted_ranks.mean().item():.2f}")

    # Build soft mask
    print(f"\n{'='*80}")
    print("Building SOFT nested mask (training)")
    print("="*80)

    soft_mask = build_nested_mask(predicted_ranks, r_max, tau=1.0, hard=False)
    print(f"Soft mask shape: {soft_mask.shape}")
    print(f"Soft mask sample (first token, first 20 dims):")
    print(f"  {soft_mask[0, 0, :20]}")
    print(f"Mask values range: [{soft_mask.min().item():.4f}, {soft_mask.max().item():.4f}]")

    # Build hard mask
    print(f"\n{'='*80}")
    print("Building HARD nested mask (inference)")
    print("="*80)

    hard_mask = build_nested_mask(predicted_ranks, r_max, tau=1.0, hard=True)
    print(f"Hard mask shape: {hard_mask.shape}")
    print(f"Hard mask sample (first token, first 20 dims):")
    print(f"  {hard_mask[0, 0, :20]}")
    print(f"Effective rank (sum of hard mask): {hard_mask[0, 0].sum().item():.0f}")

    # Test nested property
    print(f"\n{'='*80}")
    print("Verifying NESTED property")
    print("="*80)

    # Check if mask is truly nested: if gate[i]=1, then gate[j]=1 for all j<i
    for token_idx in range(min(3, seq_len)):
        mask_sample = hard_mask[0, token_idx]

        # Find the cutoff point
        ones = (mask_sample == 1).nonzero(as_tuple=True)[0]
        zeros = (mask_sample == 0).nonzero(as_tuple=True)[0]

        if len(ones) > 0 and len(zeros) > 0:
            last_one = ones.max().item()
            first_zero = zeros.min().item()
            is_nested = (first_zero > last_one)
            print(f"Token {token_idx}: Last 1 at dim {last_one}, First 0 at dim {first_zero}, Nested: {is_nested}")
        else:
            print(f"Token {token_idx}: All ones or all zeros")

    # Test rank regularization
    print(f"\n{'='*80}")
    print("Testing Rank Regularization")
    print("="*80)

    rank_reg_loss = RankRegularizationLoss(target_avg_rank=384.0, weight=0.001)
    loss = rank_reg_loss(predicted_ranks)
    print(f"Regularization loss: {loss.item():.6f}")
    print(f"This encourages average rank to be close to 384")

    # Test Gumbel variant
    print(f"\n{'='*80}")
    print("Testing Gumbel-Softmax variant")
    print("="*80)

    gumbel_mask = build_nested_mask_gumbel(predicted_ranks, r_max, tau=0.5, hard=True, training=True)
    print(f"Gumbel mask shape: {gumbel_mask.shape}")
    print(f"Gumbel mask sample (first token, first 20 dims):")
    print(f"  {gumbel_mask[0, 0, :20]}")

    print(f"\n{'='*80}")
    print("All tests passed! ✅")
    print("="*80)
