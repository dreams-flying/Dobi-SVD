"""
Matryoshka SVD Layer

Builds on SVD-LLM's whitened SVD decomposition to add per-token dynamic rank.

Key features:
- Uses whitened SVD from SVD-LLM (activation-aware weight decomposition)
- Per-token rank prediction
- Soft gating for smooth rank transitions
- Nested rank structure (supports any rank from r_min to r_max)

Architecture:
  Input x -> Rank Predictor -> r_token
                            |
  Input x -> V_proj -> Low-rank features
                    |
                    -> Soft Gating (based on r_token) -> Gated features
                                                       |
                                                       -> U_proj -> Output

Author: Claude
Date: 2025-12-15
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class RankPredictor(nn.Module):
    """
    Predicts per-token rank for dynamic compression.

    Architecture:
        hidden -> MLP -> sigmoid -> rank

    The predictor learns which tokens need high fidelity (complex patterns)
    vs which can use lower rank (simple patterns).
    """

    def __init__(
        self,
        hidden_dim: int,
        r_max: int,
        r_min: int = 0,
        predictor_hidden: int = 128,
        temperature: float = 1.0
    ):
        super().__init__()
        self.r_max = r_max
        self.r_min = r_min
        self.temperature = temperature

        # Lightweight MLP predictor
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, predictor_hidden),
            nn.ReLU(),
            nn.Linear(predictor_hidden, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict rank for each token.

        Args:
            x: Input features [batch, seq_len, hidden_dim]

        Returns:
            Predicted ranks [batch, seq_len, 1] in range [r_min, r_max]
        """
        # Get rank ratio [0, 1]
        rank_ratio = self.predictor(x)  # [batch, seq_len, 1]

        # Scale to [r_min, r_max]
        rank = self.r_min + rank_ratio * (self.r_max - self.r_min)

        return rank / self.temperature


class MatryoshkaSVDLayer(nn.Module):
    """
    Matryoshka SVD layer with per-token dynamic rank.

    This layer wraps SVD-LLM's factorized weights (U, V) and adds:
    1. Per-token rank prediction
    2. Soft gating based on predicted rank
    3. Support for nested rank structure

    Forward pass:
        1. Predict rank r_i for each token i
        2. Apply V projection: z = x @ V
        3. Apply soft gating: z_gated = z * gate(r_i)
        4. Apply U projection: y = z_gated @ U^T
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r_max: int,
        r_min: int = 8,
        use_rank_predictor: bool = True,
        gating_tau: float = 0.1,
        bias: bool = False
    ):
        """
        Args:
            in_features: Input dimension
            out_features: Output dimension
            r_max: Maximum rank (size of low-rank bottleneck)
            r_min: Minimum rank
            use_rank_predictor: If True, use dynamic per-token rank
            gating_tau: Temperature for soft gating (smaller = sharper cutoff)
            bias: Whether to use bias (typically False for LLMs)
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.r_max = r_max
        self.r_min = r_min
        self.gating_tau = gating_tau
        self.use_rank_predictor = use_rank_predictor

        # V projection: [in_features, r_max] - reduces dimension
        self.v_proj = nn.Linear(in_features, r_max, bias=False)

        # U projection: [r_max, out_features] - expands dimension
        self.u_proj = nn.Linear(r_max, out_features, bias=False)

        # Optional bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # Rank predictor (optional)
        if use_rank_predictor:
            self.rank_predictor = RankPredictor(
                hidden_dim=in_features,
                r_max=r_max,
                r_min=r_min
            )
        else:
            self.rank_predictor = None

        # Fixed rank override (for multi-scale training)
        self.fixed_rank = None

    def compute_soft_gating(
        self,
        rank: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """
        Compute soft gating weights based on predicted rank.

        Uses sigmoid function centered at each position k:
            gate_k = σ((r - k) / τ)

        When r = k, gate ≈ 0.5
        When r >> k, gate ≈ 1 (keep this dimension)
        When r << k, gate ≈ 0 (truncate this dimension)

        Args:
            rank: Predicted ranks [batch, seq_len, 1]
            device: Device to create gates on

        Returns:
            Gating weights [batch, seq_len, r_max]
        """
        # Create position indices [1, 2, ..., r_max]
        positions = torch.arange(
            1, self.r_max + 1,
            device=device,
            dtype=rank.dtype
        )  # [r_max]

        # Compute (r - k) for all positions
        # rank: [batch, seq_len, 1]
        # positions: [r_max]
        # -> diff: [batch, seq_len, r_max]
        diff = rank - positions.unsqueeze(0).unsqueeze(0)

        # Apply sigmoid gating
        gates = torch.sigmoid(diff / self.gating_tau)

        return gates

    def forward(
        self,
        x: torch.Tensor,
        return_rank: bool = False
    ) -> torch.Tensor:
        """
        Forward pass with per-token dynamic rank.

        Args:
            x: Input tensor [batch, seq_len, in_features]
            return_rank: If True, also return predicted ranks

        Returns:
            Output tensor [batch, seq_len, out_features]
            (and optionally predicted ranks)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Step 1: Predict rank (if using rank predictor)
        if self.use_rank_predictor and self.fixed_rank is None:
            # Dynamic rank per token (works in both training and inference!)
            rank = self.rank_predictor(x)  # [batch, seq_len, 1]
        elif self.fixed_rank is not None:
            # Fixed rank (for multi-scale training)
            rank = torch.full(
                (batch_size, seq_len, 1),
                self.fixed_rank,
                device=device,
                dtype=x.dtype
            )
        else:
            # Use maximum rank (fallback when no predictor)
            rank = torch.full(
                (batch_size, seq_len, 1),
                self.r_max,
                device=device,
                dtype=x.dtype
            )

        # Step 2: Apply V projection (dimension reduction)
        z = self.v_proj(x)  # [batch, seq_len, r_max]

        # Step 3: Apply soft gating based on predicted rank
        gates = self.compute_soft_gating(rank, device)  # [batch, seq_len, r_max]
        z_gated = z * gates  # Element-wise gating

        # Step 4: Apply U projection (dimension expansion)
        output = self.u_proj(z_gated)  # [batch, seq_len, out_features]

        # Add bias if present
        if self.bias is not None:
            output = output + self.bias

        # Cache average predicted rank for monitoring (used in get_effective_rank)
        if self.use_rank_predictor and self.fixed_rank is None:
            with torch.no_grad():
                self._last_avg_rank = rank.mean().item()
        else:
            self._last_avg_rank = None

        if return_rank:
            return output, rank
        return output

    def set_fixed_rank(self, rank: Optional[float]):
        """Set fixed rank for multi-scale training."""
        self.fixed_rank = rank

    def get_effective_rank(self) -> float:
        """
        Get the effective rank being used.

        Useful for monitoring compression during training.
        Returns the actual average predicted rank if available.
        """
        if self.fixed_rank is not None:
            return self.fixed_rank
        if hasattr(self, '_last_avg_rank') and self._last_avg_rank is not None:
            return self._last_avg_rank  # Return actual predicted average rank
        return self.r_max  # Fallback

    @classmethod
    def from_svdllm_weights(
        cls,
        u_weight: torch.Tensor,
        v_weight: torch.Tensor,
        r_max: int,
        r_min: int = 8,
        use_rank_predictor: bool = True
    ):
        """
        Create MatryoshkaSVDLayer from SVD-LLM's U and V weights.

        Args:
            u_weight: U projection weight from SVD-LLM [out_features, r]
            v_weight: V projection weight from SVD-LLM [r, in_features]
            r_max: Maximum rank to use
            r_min: Minimum rank
            use_rank_predictor: Whether to use dynamic rank prediction

        Returns:
            MatryoshkaSVDLayer instance
        """
        # Determine dimensions from weights
        out_features = u_weight.shape[0]
        in_features = v_weight.shape[1]
        current_r = min(u_weight.shape[1], v_weight.shape[0])

        # Check rank compatibility
        if r_max > current_r:
            print(f"Warning: r_max ({r_max}) > current rank ({current_r})")
            print(f"Using current rank as r_max")
            r_max = current_r

        # Create layer
        layer = cls(
            in_features=in_features,
            out_features=out_features,
            r_max=r_max,
            r_min=r_min,
            use_rank_predictor=use_rank_predictor
        )

        # Load weights (truncated to r_max if needed)
        with torch.no_grad():
            layer.u_proj.weight.copy_(u_weight[:, :r_max])
            layer.v_proj.weight.copy_(v_weight[:r_max, :])

        return layer


def replace_with_matryoshka_svd(
    model: nn.Module,
    r_max: int,
    r_min: int = 8,
    target_modules: Optional[list] = None
):
    """
    Replace linear layers with MatryoshkaSVDLayer.

    This function assumes the model already has SVD-LLM structure
    (with u_proj and v_proj layers).

    Args:
        model: Model with SVD-LLM layers
        r_max: Maximum rank
        r_min: Minimum rank
        target_modules: List of module names to replace (e.g., ['q_proj', 'k_proj'])
    """
    # This will be implemented based on specific model structure
    # For now, just a placeholder
    raise NotImplementedError("Model replacement to be implemented based on specific architecture")
