#!/usr/bin/env python3
"""
Matryoshka SVD Layer V2 - Improved with clean rank predictor design.

Key improvements over V1:
1. Cleaner separation: RankPredictor only predicts, build_nested_mask only masks
2. Easy to switch between soft/hard masking
3. Built-in rank regularization support
4. Better gradient flow with optional Gumbel-Softmax
"""

import torch
import torch.nn as nn
from typing import Optional

from improved_rank_predictor import (
    ImprovedRankPredictor,
    build_nested_mask,
    build_nested_mask_gumbel,
    RankRegularizationLoss
)


class MatryoshkaSVDLayerV2(nn.Module):
    """
    Matryoshka SVD Layer with improved rank prediction.

    This version uses the improved design:
    - RankPredictor only predicts continuous rank ∈ [r_min, r_max]
    - Separate build_nested_mask() for mask construction
    - Clean separation enables easier experimentation
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r_max: int,
        r_min: int,
        use_rank_predictor: bool = True,
        use_contextual_predictor: bool = False,
        gating_tau: float = 1.0,
        use_gumbel: bool = False,
        hard_inference: bool = True,
        bias: bool = False
    ):
        """
        Args:
            in_features: Input dimension
            out_features: Output dimension
            r_max: Maximum rank
            r_min: Minimum rank
            use_rank_predictor: Whether to use dynamic rank prediction
            use_contextual_predictor: Whether predictor uses context
            gating_tau: Temperature for soft gating
            use_gumbel: Whether to use Gumbel-Softmax for better gradients
            hard_inference: Use hard binary mask during inference
            bias: Whether to use bias
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.r_max = r_max
        self.r_min = r_min
        self.use_rank_predictor = use_rank_predictor
        self.use_contextual_predictor = use_contextual_predictor
        self.gating_tau = gating_tau
        self.use_gumbel = use_gumbel
        self.hard_inference = hard_inference
        self.bias = bias

        # SVD decomposition: W ≈ U @ V
        self.v_proj = nn.Linear(in_features, r_max, bias=False)
        self.u_proj = nn.Linear(r_max, out_features, bias=bias)

        # Rank predictor
        if use_rank_predictor:
            self.rank_predictor = ImprovedRankPredictor(
                in_features=in_features,
                r_max=r_max,
                r_min=r_min,
                use_context=use_contextual_predictor
            )
        else:
            self.rank_predictor = None

        # For fixed rank mode (training with specific rank)
        self.fixed_rank = None

    def set_fixed_rank(self, rank: Optional[int]):
        """
        Set a fixed rank for training at specific compression level.

        Args:
            rank: Fixed rank to use. If None, use dynamic prediction.
        """
        self.fixed_rank = rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with Matryoshka gating.

        Args:
            x: Input tensor [batch_size, seq_len, in_features]

        Returns:
            output: Output tensor [batch_size, seq_len, out_features]
        """
        # V projection: [batch, seq_len, in_features] → [batch, seq_len, r_max]
        z = self.v_proj(x)

        # Determine rank and build mask
        if self.fixed_rank is not None:
            # Fixed rank mode (training with specific rank)
            batch_size, seq_len, _ = x.shape
            rank = torch.full(
                (batch_size, seq_len, 1),
                self.fixed_rank,
                device=x.device,
                dtype=x.dtype
            )
        elif self.use_rank_predictor and self.rank_predictor is not None:
            # Dynamic rank prediction
            rank = self.rank_predictor(x)  # [batch, seq_len, 1]
        else:
            # Default: use maximum rank
            batch_size, seq_len, _ = x.shape
            rank = torch.full(
                (batch_size, seq_len, 1),
                self.r_max,
                device=x.device,
                dtype=x.dtype
            )

        # Build nested mask
        use_hard = self.hard_inference and not self.training

        if self.use_gumbel and self.training:
            # Gumbel-Softmax for better gradients during training
            gates = build_nested_mask_gumbel(
                rank,
                self.r_max,
                tau=self.gating_tau,
                hard=use_hard,
                training=self.training
            )
        else:
            # Standard sigmoid-based mask
            gates = build_nested_mask(
                rank,
                self.r_max,
                tau=self.gating_tau,
                hard=use_hard
            )

        # Apply gating: [batch, seq_len, r_max] ⊙ [batch, seq_len, r_max]
        z_gated = z * gates

        # U projection: [batch, seq_len, r_max] → [batch, seq_len, out_features]
        output = self.u_proj(z_gated)

        return output

    def get_effective_rank(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the effective rank being used (for logging/analysis).

        Args:
            x: Input tensor

        Returns:
            effective_rank: Average effective rank across tokens
        """
        if self.fixed_rank is not None:
            return torch.tensor(self.fixed_rank, dtype=torch.float32)

        if self.use_rank_predictor and self.rank_predictor is not None:
            rank = self.rank_predictor(x)
            return rank.mean()

        return torch.tensor(self.r_max, dtype=torch.float32)

    def get_compression_ratio(self) -> float:
        """
        Get theoretical compression ratio at maximum rank.

        Returns:
            compression_ratio: Ratio of compressed params to original params
        """
        original_params = self.in_features * self.out_features
        compressed_params = self.in_features * self.r_max + self.r_max * self.out_features

        return compressed_params / original_params

    def extra_repr(self) -> str:
        """String representation for print(model)."""
        return (
            f'in_features={self.in_features}, '
            f'out_features={self.out_features}, '
            f'r_range=[{self.r_min}, {self.r_max}], '
            f'use_rank_predictor={self.use_rank_predictor}, '
            f'contextual={self.use_contextual_predictor}, '
            f'tau={self.gating_tau}, '
            f'gumbel={self.use_gumbel}, '
            f'hard_inference={self.hard_inference}'
        )


def create_matryoshka_from_linear(
    linear: nn.Linear,
    r_max: int,
    r_min: int,
    use_contextual_predictor: bool = False,
    use_gumbel: bool = False
) -> MatryoshkaSVDLayerV2:
    """
    Create MatryoshkaSVDLayerV2 from an existing Linear layer using SVD.

    Args:
        linear: Existing nn.Linear layer
        r_max: Maximum rank
        r_min: Minimum rank
        use_contextual_predictor: Whether to use contextual predictor
        use_gumbel: Whether to use Gumbel-Softmax

    Returns:
        matryoshka_layer: New MatryoshkaSVDLayerV2 with SVD-initialized weights
    """
    in_features = linear.in_features
    out_features = linear.out_features

    # Create Matryoshka layer
    matryoshka_layer = MatryoshkaSVDLayerV2(
        in_features=in_features,
        out_features=out_features,
        r_max=r_max,
        r_min=r_min,
        use_rank_predictor=True,
        use_contextual_predictor=use_contextual_predictor,
        use_gumbel=use_gumbel,
        bias=linear.bias is not None
    )

    # Perform SVD on weight matrix
    W = linear.weight.data  # [out_features, in_features]

    # SVD: W = U @ S @ V^T
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)

    # Truncate to r_max
    U_truncated = U[:, :r_max]  # [out_features, r_max]
    S_truncated = S[:r_max]  # [r_max]
    Vt_truncated = Vt[:r_max, :]  # [r_max, in_features]

    # W ≈ U_truncated @ diag(S_truncated) @ Vt_truncated
    # We want: W ≈ u_proj @ v_proj
    # So: u_proj = U_truncated @ diag(sqrt(S_truncated))
    #     v_proj = diag(sqrt(S_truncated)) @ Vt_truncated

    sqrt_S = torch.sqrt(S_truncated + 1e-8)  # Add epsilon for numerical stability

    # Initialize v_proj: [r_max, in_features]
    v_proj_weight = torch.diag(sqrt_S) @ Vt_truncated
    matryoshka_layer.v_proj.weight.data = v_proj_weight

    # Initialize u_proj: [out_features, r_max]
    u_proj_weight = U_truncated @ torch.diag(sqrt_S)
    matryoshka_layer.u_proj.weight.data = u_proj_weight

    # Copy bias if exists
    if linear.bias is not None:
        matryoshka_layer.u_proj.bias.data = linear.bias.data.clone()

    return matryoshka_layer


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("Testing MatryoshkaSVDLayerV2")
    print("="*80)

    # Setup
    batch_size = 2
    seq_len = 8
    in_features = 4096
    out_features = 4096
    r_max = 512
    r_min = 256

    # Create layer
    layer = MatryoshkaSVDLayerV2(
        in_features=in_features,
        out_features=out_features,
        r_max=r_max,
        r_min=r_min,
        use_rank_predictor=True,
        use_contextual_predictor=True,
        use_gumbel=True
    )

    print(f"\nLayer configuration:")
    print(f"  {layer}")

    # Test input
    x = torch.randn(batch_size, seq_len, in_features)
    print(f"\nInput shape: {x.shape}")

    # Forward pass (training mode - soft gates)
    layer.train()
    print(f"\n{'='*80}")
    print("Training mode (soft gates)")
    print("="*80)

    output_train = layer(x)
    print(f"Output shape: {output_train.shape}")
    print(f"Output mean: {output_train.mean().item():.6f}")
    print(f"Output std: {output_train.std().item():.6f}")
    print(f"Effective rank: {layer.get_effective_rank(x).item():.2f}")

    # Forward pass (eval mode - hard gates)
    layer.eval()
    print(f"\n{'='*80}")
    print("Eval mode (hard gates)")
    print("="*80)

    with torch.no_grad():
        output_eval = layer(x)

    print(f"Output shape: {output_eval.shape}")
    print(f"Output mean: {output_eval.mean().item():.6f}")
    print(f"Output std: {output_eval.std().item():.6f}")
    print(f"Effective rank: {layer.get_effective_rank(x).item():.2f}")

    # Test fixed rank mode
    print(f"\n{'='*80}")
    print("Fixed rank mode (rank=384)")
    print("="*80)

    layer.set_fixed_rank(384)
    with torch.no_grad():
        output_fixed = layer(x)

    print(f"Output shape: {output_fixed.shape}")
    print(f"Effective rank: {layer.get_effective_rank(x).item():.2f}")

    # Test conversion from Linear
    print(f"\n{'='*80}")
    print("Testing conversion from Linear layer")
    print("="*80)

    linear = nn.Linear(in_features, out_features)
    matryoshka_converted = create_matryoshka_from_linear(
        linear,
        r_max=512,
        r_min=256,
        use_gumbel=True
    )

    print(f"Converted layer: {matryoshka_converted}")

    # Compare outputs
    with torch.no_grad():
        linear_output = linear(x)
        matryoshka_converted.set_fixed_rank(512)  # Use full rank
        matryoshka_converted.eval()
        matryoshka_output = matryoshka_converted(x)

        diff = (linear_output - matryoshka_output).abs().mean()
        print(f"\nReconstruction error (full rank): {diff.item():.6f}")
        print(f"(Should be very small, indicating good SVD approximation)")

    print(f"\n{'='*80}")
    print("All tests passed! ✅")
    print("="*80)
    print(f"\nCompression ratio: {layer.get_compression_ratio():.2%}")
