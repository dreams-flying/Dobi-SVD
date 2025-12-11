"""
Matryoshka SVD: Unified Adaptive Rank Compression (Enhanced Version)

This module implements a theoretically sound SVD compression approach with:
1. Single SVD decomposition per layer (with activation-aware option)
2. Adaptive per-token rank selection (no routing overhead)
3. Nested Matryoshka structure for mathematical rigor
4. Learnable singular values for fine-tuning
5. Layer-wise adaptive rank ranges

Key advantages:
- No routing collapse (no discrete decisions)
- O(n×d×r) complexity (vs O(n²×d) for value-aware routing)
- Numerically stable (single SVD, smooth gradients)
- Flexible (continuous rank range, not discrete subspaces)

New features in this enhanced version:
- Activation-aware SVD initialization (借鉴 Dobi-SVD)
- Learnable singular values with spectral regularization
- Auto rank range based on spectral energy
- Improved importance predictor with running statistics

Author: Enhanced Implementation
Date: 2025-12-11
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal, Dict, Tuple, Union
from dataclasses import dataclass
import math


# ============================================================================
# Configuration Class
# ============================================================================

@dataclass
class MatryoshkaSVDConfig:
    """Configuration for MatryoshkaSVD layers."""
    # Basic SVD parameters
    r_max: int = 256
    r_min: int = 32
    importance_strategy: Literal['norm', 'learned', 'attention'] = 'norm'
    temperature: float = 1.0
    soft_truncation_margin: float = 2.0

    # Training parameters
    enable_multiscale_loss: bool = True
    rank_regularization_weight: float = 0.001

    # New features
    learnable_singular_values: bool = False
    activation_aware_init: bool = False
    # CRITICAL: Reduced from 0.0001 to allow S to learn while maintaining stability
    spectral_regularization_weight: float = 0.00001

    # Auto rank range
    auto_rank_range: bool = False
    energy_threshold_low: float = 0.90   # r_min: 90% energy
    energy_threshold_high: float = 0.99  # r_max: 99% energy

    # Bucketed inference for dynamic computation
    use_bucketed_inference: bool = True   # Enable bucketed inference
    num_inference_buckets: int = 4        # Number of rank buckets


# ============================================================================
# Enhanced Adaptive Rank Predictor
# ============================================================================

class AdaptiveRankPredictor(nn.Module):
    """
    Enhanced token importance predictor with running statistics.

    Supports multiple strategies:
    - 'norm': L2 norm of activation (fast, no parameters)
    - 'learned': Small MLP predictor (best quality)
    - 'attention': Uses attention scores (requires hooks)
    """

    def __init__(
        self,
        hidden_size: int,
        strategy: Literal['norm', 'learned', 'attention'] = 'norm',
        predictor_hidden_dim: int = 128,
        use_running_stats: bool = True,
        device: Optional[torch.device] = None
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.strategy = strategy
        self.device = device
        self.use_running_stats = use_running_stats

        # Running statistics for stable normalization
        if use_running_stats:
            self.register_buffer('running_mean', torch.tensor(0.0))
            self.register_buffer('running_std', torch.tensor(1.0))
            self.register_buffer('num_batches_tracked', torch.tensor(0))

        if strategy == 'learned':
            # Enhanced MLP with LayerNorm and GELU
            self.predictor = nn.Sequential(
                nn.Linear(hidden_size, predictor_hidden_dim),
                nn.LayerNorm(predictor_hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(predictor_hidden_dim, predictor_hidden_dim // 2),
                nn.LayerNorm(predictor_hidden_dim // 2),
                nn.GELU(),
                nn.Linear(predictor_hidden_dim // 2, 1)
            )

            # Better initialization
            self._init_weights()

            if device is not None:
                self.predictor = self.predictor.to(device)

        elif strategy == 'attention':
            # Placeholder for attention-based importance
            self.attention_cache = None

    def _init_weights(self):
        """Initialize weights for stable training."""
        for module in self.predictor.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Initialize last layer to output ~0 (middle importance after sigmoid)
        last_linear = list(self.predictor.modules())[-1]
        if isinstance(last_linear, nn.Linear):
            nn.init.zeros_(last_linear.weight)
            if last_linear.bias is not None:
                nn.init.zeros_(last_linear.bias)

    def _compute_norm_importance(self, x: torch.Tensor) -> torch.Tensor:
        """Compute L2 norm-based importance with stable normalization."""
        # L2 norm normalized by sqrt(hidden_size)
        importance = x.norm(dim=-1, p=2) / math.sqrt(self.hidden_size)

        if self.use_running_stats and self.training:
            # Update running statistics
            batch_mean = importance.mean()
            batch_std = importance.std() + 1e-6

            # CRITICAL: Use smaller momentum for more stable running average
            # 0.01 gives ~100-batch averaging, more stable than 0.1 (10-batch)
            momentum = 0.01
            self.running_mean = (1 - momentum) * self.running_mean + momentum * batch_mean
            self.running_std = (1 - momentum) * self.running_std + momentum * batch_std
            self.num_batches_tracked += 1

        # Normalize using running stats (more stable)
        if self.use_running_stats and self.num_batches_tracked > 0:
            importance = (importance - self.running_mean) / (self.running_std + 1e-6)
        else:
            # Fallback to batch normalization
            importance = (importance - importance.mean()) / (importance.std() + 1e-6)

        # Map to [0, 1] using sigmoid
        importance = torch.sigmoid(importance)

        return importance

    def forward(
        self,
        x: torch.Tensor,
        attention_scores: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute token importance scores.

        Args:
            x: Input tensor [batch, seq_len, hidden_size]
            attention_scores: Optional attention scores

        Returns:
            importance: Importance scores [batch, seq_len] in range [0, 1]
        """
        if self.strategy == 'norm':
            return self._compute_norm_importance(x)

        elif self.strategy == 'learned':
            logits = self.predictor(x).squeeze(-1)  # [batch, seq_len]
            return torch.sigmoid(logits)

        elif self.strategy == 'attention':
            if attention_scores is None:
                raise ValueError("attention_scores required for 'attention' strategy")

            # Average attention received by each token
            importance = attention_scores.mean(dim=1).mean(dim=-2)  # [batch, seq_len]

            # Normalize to [0, 1]
            batch_size = attention_scores.size(0)
            for b in range(batch_size):
                imp_b = importance[b]
                imp_min = imp_b.min()
                imp_max = imp_b.max()
                if (imp_max - imp_min) > 1e-6:
                    importance[b] = (imp_b - imp_min) / (imp_max - imp_min + 1e-10)
                else:
                    importance[b] = torch.ones_like(imp_b) * 0.5

            return importance

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")


# ============================================================================
# Enhanced Matryoshka SVD Layer
# ============================================================================

class MatryoshkaSVDLayer(nn.Module):
    """
    Enhanced Matryoshka SVD Layer with advanced features:

    1. Activation-aware SVD initialization (借鉴 Dobi-SVD)
    2. Learnable singular values for fine-tuning
    3. Auto rank range based on spectral energy

    Mathematical formulation:
        W ≈ U @ diag(S) @ V^T  (W is [out, in], V is [r, in])

        For each token i with importance s_i ∈ [0, 1]:
            r_i = r_min + (r_max - r_min) × s_i
            truncation_i(k) = σ((r_i - k + margin) / τ)
            S_adaptive[i, k] = S[k] × truncation_i(k)

        Output: x @ V^T @ diag(S_adaptive) @ U^T
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        config: Optional[MatryoshkaSVDConfig] = None,
        calibration_data: Optional[torch.Tensor] = None,
        name: Optional[str] = None,
        device: Optional[torch.device] = None
    ):
        super().__init__()

        self.config = config or MatryoshkaSVDConfig()
        self.input_size = input_size
        self.output_size = output_size
        self.name = name or "matryoshka_svd"
        self.device = device or weight.device

        # Auto-adjust r_max if needed
        max_possible_rank = min(input_size, output_size)
        self.r_max = min(self.config.r_max, max_possible_rank)
        self.r_min = min(self.config.r_min, self.r_max - 1)

        # Validate
        assert self.r_min > 0, "r_min must be positive"
        assert self.r_max > self.r_min, "r_max must be greater than r_min"

        print(f"[{self.name}] Initializing Matryoshka SVD layer")
        print(f"  Matrix size: {output_size} x {input_size}")
        print(f"  Activation-aware: {self.config.activation_aware_init}")
        print(f"  Learnable S: {self.config.learnable_singular_values}")
        print(f"  Auto rank range: {self.config.auto_rank_range}")

        # Compute SVD (potentially activation-aware)
        U, S, V = self._compute_svd(weight, calibration_data)

        # Auto-adjust rank range based on spectral energy (if enabled)
        if self.config.auto_rank_range:
            self.r_min, self.r_max = self._auto_rank_range(S)
            print(f"  Auto-adjusted rank range: [{self.r_min}, {self.r_max}]")

        # Register SVD components
        self.register_buffer('U', U.to(self.device))
        self.register_buffer('V', V.to(self.device))

        # Singular values: learnable or fixed
        if self.config.learnable_singular_values:
            # Log-space parameterization for positivity and stability
            # CRITICAL: Clamp S before taking log to avoid log(0) and ensure reasonable range
            S_clamped = S.clamp(min=1e-6, max=1e3)  # Prevent extreme values
            log_S_init = torch.log(S_clamped).clamp(min=-10.0, max=7.0)  # log(S) in reasonable range
            self.log_S = nn.Parameter(log_S_init.to(self.device))
            self.register_buffer('S_init', S.to(self.device))  # Keep for regularization
            print(f"  Using learnable singular values (log-parameterized)")
            print(f"    Initial log_S range: [{log_S_init.min():.2f}, {log_S_init.max():.2f}]")
        else:
            self.register_buffer('S', S.to(self.device))

        # Bias
        if bias is not None:
            self.register_buffer('bias', bias.to(self.device))
        else:
            self.bias = None

        # Importance predictor
        self.importance_predictor = AdaptiveRankPredictor(
            hidden_size=input_size,
            strategy=self.config.importance_strategy,
            use_running_stats=True,
            device=self.device
        )

        # Precompute index tensor for truncation
        self.register_buffer(
            'sequence_tensor',
            torch.arange(self.r_max, dtype=torch.float32, device=self.device)
        )

        # Bucketed inference: precompute bucket boundaries
        if self.config.use_bucketed_inference:
            bucket_ranks = torch.linspace(
                self.r_min, self.r_max, self.config.num_inference_buckets + 1
            ).long()
            self.register_buffer('bucket_boundaries', bucket_ranks)
            print(f"  Bucketed inference enabled: {self.config.num_inference_buckets} buckets")
            print(f"    Bucket boundaries: {bucket_ranks.tolist()}")
        else:
            self.bucket_boundaries = None

        # Statistics tracking
        self.register_buffer(
            'avg_rank_tracker',
            torch.tensor(float(self.r_min + self.r_max) / 2, device=self.device)
        )
        self.forward_count = 0

        # Multi-scale outputs cache
        self.multiscale_outputs: Dict[str, torch.Tensor] = {}

        print(f"  Initialization complete!")
        print(f"  Compression: {self._get_compression_ratio():.2%}")

    @property
    def S(self) -> torch.Tensor:
        """Get singular values (learnable or fixed) with numerical stability."""
        if self.config.learnable_singular_values:
            # CRITICAL: Clamp log_S to prevent exp explosion
            # log_S in [-20, 10] => S in [2e-9, 22026]
            log_S_clamped = torch.clamp(self.log_S, min=-20.0, max=10.0)
            return torch.exp(log_S_clamped)
        else:
            return self._buffers['S']

    def _compute_svd(
        self,
        weight: torch.Tensor,
        calibration_data: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute SVD decomposition with optional activation-aware initialization.

        Args:
            weight: Weight matrix [output_size, input_size]
            calibration_data: Optional calibration activations [num_samples, input_size]

        Returns:
            U: Left singular vectors [output_size, r_max]
            S: Singular values [r_max]
            V: Right singular vectors [r_max, input_size]
        """
        print(f"[{self.name}] Computing SVD decomposition...")
        print(f"  Method: {'Activation-aware' if self.config.activation_aware_init and calibration_data is not None else 'Standard'}")

        with torch.no_grad():
            # Move to CPU for memory efficiency
            weight_cpu = weight.cpu().float()

            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            try:
                if self.config.activation_aware_init and calibration_data is not None:
                    U, S, V = self._activation_aware_svd(weight_cpu, calibration_data.cpu().float())
                else:
                    U, S, V = self._standard_svd(weight_cpu)

                # Verify shapes
                assert U.shape == (self.output_size, self.r_max)
                assert S.shape == (self.r_max,)
                assert V.shape == (self.r_max, self.input_size)

                print(f"  SVD complete: U{list(U.shape)}, S{list(S.shape)}, V{list(V.shape)}")
                print(f"  Singular value range: [{S.min():.6f}, {S.max():.6f}]")

            except Exception as e:
                print(f"[{self.name}] ❌ SVD failed: {e}")
                print(f"  Using random initialization as fallback")

                U = torch.randn(self.output_size, self.r_max, dtype=torch.float32) * 0.01
                S = torch.ones(self.r_max, dtype=torch.float32)
                V = torch.randn(self.r_max, self.input_size, dtype=torch.float32) * 0.01

            # Clean up
            del weight_cpu
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return U, S, V

    def _standard_svd(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Standard truncated SVD on CPU."""
        matrix_size = weight.numel()

        if matrix_size > 4 * 1024 * 1024:  # > 4M elements
            print(f"  Large matrix ({matrix_size/1e6:.1f}M elements), using randomized SVD")
            U, S, V = self._randomized_svd(weight, self.r_max)
        else:
            print(f"  Using standard SVD on CPU")
            U, S, Vh = torch.linalg.svd(weight, full_matrices=False)

            # Truncate to r_max
            U = U[:, :self.r_max]
            S = S[:self.r_max]
            V = Vh[:self.r_max, :]

        return U, S, V

    def _activation_aware_svd(
        self,
        weight: torch.Tensor,
        calibration_data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Simplified activation-aware SVD inspired by Dobi-SVD.

        Key idea (from Dobi-SVD philosophy):
        - Dobi does SVD on activations: U, S, V = svd(X) in forward pass
        - For Matryoshka (precomputed SVD on weights), we:
          1. Do standard SVD on weight: U, S, V = svd(W)
          2. Reorder components by activation importance
          3. Keep most important components first

        This is much simpler and more stable than matrix transformation.
        """
        print(f"  Using {calibration_data.shape[0]} calibration samples for activation-aware SVD")

        # Step 1: Standard SVD on weight
        print(f"  Computing standard SVD on weight...")
        U, S, Vh = torch.linalg.svd(weight, full_matrices=False)

        # Step 2: Compute activation importance for each component
        # For each singular component, measure how much it's activated by real data
        X = calibration_data  # [num_samples, input_size]

        print(f"  Computing component activations...")
        # V^T @ X^T: each row of Vh represents a component direction
        # X @ Vh.T gives how each component responds to inputs
        component_activations = X @ Vh.T  # [num_samples, rank]

        # Importance = average absolute activation across all samples
        component_importance = component_activations.abs().mean(dim=0)  # [rank]

        print(f"  Component importance range: [{component_importance.min():.4f}, {component_importance.max():.4f}]")

        # Step 3: Reorder by importance (most important first)
        importance_order = component_importance.argsort(descending=True)

        # Apply reordering
        U_reordered = U[:, importance_order]
        S_reordered = S[importance_order]
        V_reordered = Vh[importance_order, :]

        # Truncate to r_max
        U_final = U_reordered[:, :self.r_max]
        S_final = S_reordered[:self.r_max]
        V_final = V_reordered[:self.r_max, :]

        print(f"  Activation-aware SVD complete!")
        print(f"  Top 5 component importance: {component_importance[importance_order[:5]].tolist()}")

        return U_final, S_final, V_final

    @staticmethod
    def _randomized_svd(
        matrix: torch.Tensor,
        rank: int,
        n_oversamples: int = 10,
        n_iter: int = 2
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Randomized SVD using power iteration.

        Based on: "Finding structure with randomness" (Halko et al., 2011)
        """
        m, n = matrix.shape
        r = min(rank + n_oversamples, min(m, n))

        # Random projection
        Omega = torch.randn(n, r, dtype=matrix.dtype, device=matrix.device)

        # Power iteration for better accuracy
        Y = matrix @ Omega
        for _ in range(n_iter):
            Y = matrix @ (matrix.T @ Y)

        # QR decomposition
        Q, _ = torch.linalg.qr(Y)

        # Project matrix
        B = Q.T @ matrix

        # SVD of small matrix
        U_tilde, S, Vh = torch.linalg.svd(B, full_matrices=False)

        # Recover U
        U = Q @ U_tilde

        # Truncate to desired rank
        U = U[:, :rank]
        S = S[:rank]
        V = Vh[:rank, :]

        return U, S, V

    def _auto_rank_range(self, S: torch.Tensor) -> Tuple[int, int]:
        """
        Automatically determine r_min and r_max based on spectral energy.

        Args:
            S: Singular values [r_max]

        Returns:
            r_min, r_max: Adjusted rank range
        """
        total_energy = (S ** 2).sum()
        cumulative_energy = (S ** 2).cumsum(dim=0) / total_energy

        # Find r_min: rank capturing energy_threshold_low
        r_min_candidates = (cumulative_energy >= self.config.energy_threshold_low).nonzero(as_tuple=True)[0]
        r_min = r_min_candidates[0].item() + 1 if len(r_min_candidates) > 0 else 1

        # Find r_max: rank capturing energy_threshold_high
        r_max_candidates = (cumulative_energy >= self.config.energy_threshold_high).nonzero(as_tuple=True)[0]
        r_max = r_max_candidates[0].item() + 1 if len(r_max_candidates) > 0 else len(S)

        # Ensure valid range
        r_min = max(1, min(r_min, len(S) - 1))
        r_max = max(r_min + 1, min(r_max, len(S)))

        # Log energy captured
        energy_at_rmin = cumulative_energy[r_min - 1].item() if r_min > 0 else 0
        energy_at_rmax = cumulative_energy[r_max - 1].item() if r_max > 0 else 0
        print(f"  Energy at r_min={r_min}: {energy_at_rmin:.2%}")
        print(f"  Energy at r_max={r_max}: {energy_at_rmax:.2%}")

        return r_min, r_max

    def compute_soft_truncation(self, adaptive_rank: torch.Tensor) -> torch.Tensor:
        """
        Compute soft truncation weights using improved sigmoid function.

        Args:
            adaptive_rank: Per-token ranks [batch, seq_len]

        Returns:
            truncation: Truncation weights [batch, seq_len, r_max]
        """
        r_expanded = adaptive_rank.unsqueeze(-1)  # [batch, seq, 1]
        seq_expanded = self.sequence_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, r_max]

        # Improved soft truncation with margin
        # margin=2.0 ensures components near r-1 get high weight (>0.95)
        margin = self.config.soft_truncation_margin
        truncation = torch.sigmoid((r_expanded - seq_expanded + margin) / self.config.temperature)

        return truncation

    def forward(
        self,
        x: torch.Tensor,
        attention_scores: Optional[torch.Tensor] = None,
        return_rank_info: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Forward pass with adaptive rank selection.

        Args:
            x: Input tensor [batch, seq_len, input_size]
            attention_scores: Optional attention scores
            return_rank_info: If True, return (output, rank_info) tuple

        Returns:
            output: Transformed tensor [batch, seq_len, output_size]
            rank_info: Optional dict with rank statistics
        """
        batch_size, seq_len, input_dim = x.shape
        device = x.device
        input_dtype = x.dtype

        # Get singular values (potentially learnable)
        S = self.S

        # Step 1: Compute token importance
        importance = self.importance_predictor(x, attention_scores)  # [batch, seq_len]

        # Step 2: Map importance to adaptive rank
        adaptive_rank = self.r_min + (self.r_max - self.r_min) * importance
        adaptive_rank = torch.clamp(adaptive_rank, self.r_min, self.r_max)

        # Step 3: Choose forward method
        # Use bucketed inference during evaluation for speed
        if not self.training and self.config.use_bucketed_inference and self.bucket_boundaries is not None:
            output = self._forward_bucketed(x, adaptive_rank, S, input_dtype)
        else:
            output = self._forward_soft(x, adaptive_rank, S, input_dtype)

        # Add bias if present
        if self.bias is not None:
            output = output + self.bias

        # Update statistics (for monitoring)
        if self.training:
            with torch.no_grad():
                avg_rank_current = adaptive_rank.mean().item()
                self.forward_count += 1
                # Exponential moving average
                alpha = 0.9
                self.avg_rank_tracker = alpha * self.avg_rank_tracker + (1 - alpha) * avg_rank_current

        # Multi-scale outputs (for training loss) - not used in this version
        # Can be added if needed

        if return_rank_info:
            rank_info = {
                'importance': importance,
                'adaptive_rank': adaptive_rank,
                'avg_rank': adaptive_rank.mean().item(),
                'min_rank': adaptive_rank.min().item(),
                'max_rank': adaptive_rank.max().item(),
                'rank_std': adaptive_rank.std().item(),
                'singular_values': S.detach() if not self.config.learnable_singular_values else S
            }
            return output, rank_info

        return output

    def _forward_soft(
        self,
        x: torch.Tensor,
        adaptive_rank: torch.Tensor,
        S: torch.Tensor,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Standard soft-truncation forward pass (used during training).

        Args:
            x: Input tensor [batch, seq_len, input_size]
            adaptive_rank: Per-token ranks [batch, seq_len]
            S: Singular values [r_max]
            dtype: Computation dtype

        Returns:
            output: [batch, seq_len, output_size]
        """
        # CRITICAL: Check for NaN/Inf in inputs
        if not torch.isfinite(x).all():
            raise ValueError(f"NaN/Inf detected in input x at {self.name}")
        if not torch.isfinite(S).all():
            raise ValueError(f"NaN/Inf detected in singular values S at {self.name}")

        # Compute soft truncation weights
        truncation = self.compute_soft_truncation(adaptive_rank)  # [batch, seq, r_max]

        # Adaptive singular values
        S_adaptive = S.unsqueeze(0).unsqueeze(0) * truncation  # [batch, seq, r_max]

        # Transform
        V = self.V.to(dtype)
        U = self.U.to(dtype)
        S_adaptive = S_adaptive.to(dtype)

        # x @ V^T -> [batch, seq, r_max]
        xV = torch.matmul(x, V.T)

        # Element-wise multiply with S
        xVS = xV * S_adaptive

        # @ U^T -> [batch, seq, output_size]
        output = torch.matmul(xVS, U.T)

        # CRITICAL: Check for NaN/Inf in output
        if not torch.isfinite(output).all():
            raise ValueError(f"NaN/Inf detected in output at {self.name}")

        return output

    def _forward_bucketed(
        self,
        x: torch.Tensor,
        adaptive_rank: torch.Tensor,
        S: torch.Tensor,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Bucketed forward pass for efficient inference with dynamic computation.

        Key idea: Group tokens by their rank, process each group with only
        the required number of singular components, avoiding unnecessary computation.

        Args:
            x: Input tensor [batch, seq_len, input_size]
            adaptive_rank: Per-token ranks [batch, seq_len]
            S: Singular values [r_max]
            dtype: Computation dtype

        Returns:
            output: [batch, seq_len, output_size]
        """
        # CRITICAL: Check for NaN/Inf in inputs
        if not torch.isfinite(x).all():
            raise ValueError(f"NaN/Inf detected in input x at {self.name}")
        if not torch.isfinite(S).all():
            raise ValueError(f"NaN/Inf detected in singular values S at {self.name}")

        batch_size, seq_len, _ = x.shape
        output = torch.zeros(batch_size, seq_len, self.output_size, device=x.device, dtype=dtype)

        V = self.V.to(dtype)
        U = self.U.to(dtype)
        S = S.to(dtype)

        # Process each bucket
        for i in range(len(self.bucket_boundaries) - 1):
            r_low = self.bucket_boundaries[i].item()
            r_high = self.bucket_boundaries[i + 1].item()
            r_use = r_high  # Use the upper bound of the bucket

            # Find tokens in this bucket
            if i == 0:
                mask = adaptive_rank < r_high
            elif i == len(self.bucket_boundaries) - 2:
                mask = adaptive_rank >= r_low
            else:
                mask = (adaptive_rank >= r_low) & (adaptive_rank < r_high)

            if not mask.any():
                continue

            # Extract tokens (flatten batch and seq)
            flat_mask = mask.view(-1)
            x_flat = x.view(-1, x.shape[-1])
            x_bucket = x_flat[flat_mask]  # [num_tokens_in_bucket, input_size]

            if x_bucket.shape[0] == 0:
                continue

            # Compute with truncated SVD components
            V_trunc = V[:r_use, :]  # [r_use, input_size]
            U_trunc = U[:, :r_use]  # [output_size, r_use]
            S_trunc = S[:r_use]     # [r_use]

            # Forward: x @ V^T @ diag(S) @ U^T
            xV = torch.matmul(x_bucket, V_trunc.T)  # [num_tokens, r_use]
            xVS = xV * S_trunc.unsqueeze(0)        # [num_tokens, r_use]
            out_bucket = torch.matmul(xVS, U_trunc.T)  # [num_tokens, output_size]

            # Put back
            output_flat = output.view(-1, self.output_size)
            output_flat[flat_mask] = out_bucket

        # CRITICAL: Check for NaN/Inf in output
        if not torch.isfinite(output).all():
            raise ValueError(f"NaN/Inf detected in bucketed output at {self.name}")

        return output

    def get_rank_regularization_loss(self) -> torch.Tensor:
        """
        Compute rank regularization loss to encourage compression.

        Returns:
            loss: Scalar tensor representing normalized average rank
        """
        if self.avg_rank_tracker.item() == 0:
            return torch.tensor(0.0, device=self.device)

        # Normalize to [0, 1] range
        normalized_avg_rank = (self.avg_rank_tracker - self.r_min) / (self.r_max - self.r_min)

        return self.config.rank_regularization_weight * normalized_avg_rank

    def get_spectral_regularization_loss(self) -> torch.Tensor:
        """
        Compute spectral regularization to keep singular values close to initialization.

        Only active if learnable_singular_values is True.

        Returns:
            loss: Scalar tensor
        """
        if not self.config.learnable_singular_values:
            return torch.tensor(0.0, device=self.device)

        if self.config.spectral_regularization_weight <= 0:
            return torch.tensor(0.0, device=self.device)

        S_current = self.S
        S_init = self.S_init

        # L2 distance between current and initial singular values
        spectral_diff = ((S_current - S_init) ** 2).mean()

        return self.config.spectral_regularization_weight * spectral_diff

    def get_regularization_losses(self) -> Dict[str, torch.Tensor]:
        """
        Get all regularization losses.

        Returns:
            Dict with 'rank_reg' and 'spectral_reg' keys
        """
        return {
            'rank_reg': self.get_rank_regularization_loss(),
            'spectral_reg': self.get_spectral_regularization_loss()
        }

    def _get_compression_ratio(self) -> float:
        """Compute compression ratio based on average rank."""
        avg_rank = self.avg_rank_tracker.item()
        if avg_rank == 0:
            avg_rank = (self.r_min + self.r_max) / 2

        # Parameters: U (output×r) + S (r) + V (r×input)
        actual_params = self.output_size * avg_rank + avg_rank + avg_rank * self.input_size
        original_params = self.output_size * self.input_size

        return actual_params / original_params

    def __repr__(self):
        return (f"MatryoshkaSVDLayer(name={self.name}, "
                f"in={self.input_size}, out={self.output_size}, "
                f"r=[{self.r_min}, {self.r_max}], "
                f"strategy={self.importance_predictor.strategy}, "
                f"learnable_S={self.config.learnable_singular_values}, "
                f"avg_rank={self.avg_rank_tracker.item():.1f})")


# ============================================================================
# Model Conversion Utility
# ============================================================================

def replace_linear_with_matryoshka_svd(
    model: nn.Module,
    target_layers: Optional[list] = None,
    config: Optional[MatryoshkaSVDConfig] = None,
    calibration_data: Optional[Dict[str, torch.Tensor]] = None,
    verbose: bool = True,
    **kwargs
) -> nn.Module:
    """
    Replace Linear layers in a model with MatryoshkaSVDLayer.

    Args:
        model: Model to modify
        target_layers: List of layer names to replace (e.g., ['q_proj', 'v_proj'])
                      If None, replaces all Linear layers
        config: MatryoshkaSVDConfig instance
        calibration_data: Dict mapping layer names to calibration activations
        verbose: Whether to print progress
        **kwargs: Additional arguments (for backward compatibility)

    Returns:
        Modified model with MatryoshkaSVD layers
    """
    config = config or MatryoshkaSVDConfig()

    # Override config with kwargs (backward compatibility)
    if 'r_max' in kwargs:
        config.r_max = kwargs['r_max']
    if 'r_min' in kwargs:
        config.r_min = kwargs['r_min']
    if 'importance_strategy' in kwargs:
        config.importance_strategy = kwargs['importance_strategy']
    if 'temperature' in kwargs:
        config.temperature = kwargs['temperature']

    calibration_data = calibration_data or {}

    replaced_count = 0
    total_params_before = 0
    total_params_after = 0

    def replace_in_module(parent_module, parent_name=""):
        nonlocal replaced_count, total_params_before, total_params_after

        for name, module in parent_module.named_children():
            full_name = f"{parent_name}.{name}" if parent_name else name

            if isinstance(module, nn.Linear):
                # Check if this layer should be replaced
                should_replace = (target_layers is None) or any(
                    target in full_name for target in target_layers
                )

                if should_replace:
                    if verbose:
                        print(f"\n{'='*80}")
                        print(f"Replacing {full_name}")
                        print(f"{'='*80}")

                    # Get calibration data for this layer if available
                    layer_calib_data = calibration_data.get(full_name, None)

                    # Create MatryoshkaSVD layer
                    try:
                        svd_layer = MatryoshkaSVDLayer(
                            input_size=module.in_features,
                            output_size=module.out_features,
                            weight=module.weight.data,
                            bias=module.bias.data if module.bias is not None else None,
                            config=config,
                            calibration_data=layer_calib_data,
                            name=full_name,
                            device=module.weight.device
                        )

                        # Replace
                        setattr(parent_module, name, svd_layer)

                        # Update statistics
                        params_before = module.in_features * module.out_features
                        if module.bias is not None:
                            params_before += module.out_features

                        avg_rank = svd_layer.avg_rank_tracker.item()
                        params_after = (module.in_features + module.out_features) * avg_rank + avg_rank
                        if module.bias is not None:
                            params_after += module.out_features

                        total_params_before += params_before
                        total_params_after += params_after
                        replaced_count += 1

                        if verbose:
                            print(f"✓ Replaced successfully!")
                            print(f"  Params: {params_before:,} → {params_after:,} ({params_after/params_before:.2%})")

                    except Exception as e:
                        print(f"❌ Failed to replace {full_name}: {e}")
                        import traceback
                        traceback.print_exc()

            else:
                # Recurse
                replace_in_module(module, full_name)

    replace_in_module(model)

    if verbose:
        print(f"\n{'='*80}")
        print(f"Replacement Summary")
        print(f"{'='*80}")
        print(f"Replaced {replaced_count} layers")
        print(f"Total params: {total_params_before:,} → {total_params_after:,}")
        print(f"Compression ratio: {total_params_after/total_params_before:.2%}")
        print(f"{'='*80}\n")

    return model
