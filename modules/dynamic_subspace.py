"""
Dynamic Subspace Routing for SVD-Compressed LLMs

This module implements token-wise dynamic subspace selection for SVD compression.
Instead of using a fixed rank for all tokens, different tokens are routed to
different subspaces (high/mid/low rank) based on their importance.

Key Components:
- TokenRouter: Computes token importance and routing decisions
- MultiSubspaceSVDLayer: Extends SVDTransformLayer with dynamic routing
- load_balance_loss: Ensures balanced subspace utilization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .stable_svd import stable_lowrank_SVD, computeSVD_dtype, model_load_dtype, val_epsilon

# Import advanced routing strategies
try:
    from .advanced_routing import UnifiedRouter
    ADVANCED_ROUTING_AVAILABLE = True
except ImportError:
    ADVANCED_ROUTING_AVAILABLE = False
    print("Warning: Advanced routing not available. Install advanced_routing.py to use.")


class TokenRouter(nn.Module):
    """
    Token importance computation and routing module.

    Supports multiple routing strategies:
    - 'norm': Fast activation norm-based routing (default)
    - 'learned': Learnable importance predictor
    - 'attention': Attention score-based (requires attention hooks)

    Advanced routing methods (requires advanced_routing=True):
    - 'topk': Top-K routing (Switch Transformer)
    - 'expert_choice': Expert Choice routing (Google 2022)
    - 'sinkhorn': Sinkhorn routing (Optimal Transport)
    - 'gating': Gating Network routing (learnable MLP)
    - 'adaptive': Adaptive threshold routing (dynamic quantiles)
    """

    def __init__(self,
                 hidden_size,
                 n_subspaces=3,
                 routing_strategy='norm',
                 learnable_thresholds=False,
                 advanced_routing=None,
                 advanced_routing_kwargs=None,
                 device=None):
        super(TokenRouter, self).__init__()

        self.hidden_size = hidden_size
        self.n_subspaces = n_subspaces
        self.routing_strategy = routing_strategy
        self.device = device
        self.advanced_routing = advanced_routing

        # Initialize advanced router if specified
        if advanced_routing is not None:
            if not ADVANCED_ROUTING_AVAILABLE:
                raise ImportError("Advanced routing requested but advanced_routing.py not found")

            # Default kwargs
            if advanced_routing_kwargs is None:
                advanced_routing_kwargs = {}

            # Create unified router
            self.advanced_router = UnifiedRouter(
                strategy=advanced_routing,
                n_subspaces=n_subspaces,
                hidden_size=hidden_size,
                device=device,
                **advanced_routing_kwargs
            )
            print(f"Using advanced routing: {advanced_routing}")
        else:
            self.advanced_router = None

        # Initialize thresholds for routing (used for threshold-based routing only)
        # Default: [0.33, 0.67] for 3 subspaces -> low:[0,0.33), mid:[0.33,0.67), high:[0.67,1.0]
        initial_thresholds = torch.linspace(0, 1, n_subspaces + 1)[1:-1]

        if learnable_thresholds:
            self.thresholds = nn.Parameter(initial_thresholds.to(device))
        else:
            self.register_buffer('thresholds', initial_thresholds.to(device))

        # Learnable importance predictor (optional)
        if routing_strategy == 'learned':
            self.importance_net = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.ReLU(),
                nn.Linear(hidden_size // 4, 1)
            )
            # Initialize weights with small values for stability
            for module in self.importance_net.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=0.1)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)

            if device is not None:
                self.importance_net = self.importance_net.to(device)

        # Statistics tracking
        self.register_buffer('routing_counts', torch.zeros(n_subspaces))
        self.register_buffer('total_tokens', torch.tensor(0.0))

    def compute_importance(self, x, attention_scores=None, value_vectors=None):
        """
        Compute token importance scores using various strategies.

        Based on EMNLP 2024 research: "Attention Score is not All You Need"
        https://aclanthology.org/2024.emnlp-main.1178.pdf

        Supported strategies:
        - 'norm': L2 norm (fast, baseline)
        - 'l1_norm': L1 norm
        - 'value_aware': VATP method (SOTA, attention × value_norm)
        - 'learned': Learnable predictor
        - 'attention': Attention-only (legacy, not recommended)
        - 'hybrid': Multi-signal combination

        Args:
            x: Tensor of shape [batch, seq_len, hidden_size] or [seq_len, hidden_size]
            attention_scores: Optional [batch, num_heads, seq_len, seq_len]
            value_vectors: Optional [batch, seq_len, hidden_size]

        Returns:
            importance: Tensor of shape [batch, seq_len] or [seq_len]
        """
        if self.routing_strategy == 'norm':
            # L2 norm of activation (fast, no extra parameters)
            # Baseline method, widely used
            importance = x.norm(dim=-1, p=2) / (self.hidden_size ** 0.5)

        elif self.routing_strategy == 'l1_norm':
            # L1 norm (used in VATP analysis)
            importance = x.norm(dim=-1, p=1) / self.hidden_size

        elif self.routing_strategy == 'value_aware':
            # VATP (Value-Aware Token Pruning) - EMNLP 2024 SOTA
            # Key insight: importance = attention_score × ||value_vector||
            # Outperforms attention-only in 12-14/16 tasks
            if attention_scores is not None:
                # Method 1: Use real attention scores (best, if available)
                # Average attention across heads and queries
                avg_attention = attention_scores.mean(dim=1).mean(dim=-2)

                # Compute value norms (use x as value if not provided)
                values = value_vectors if value_vectors is not None else x
                value_norms = values.norm(dim=-1, p=2) / (self.hidden_size ** 0.5)

                # VATP formula
                importance = avg_attention * value_norms
            else:
                # Method 2: Approximate VATP without actual attention
                # Use self-similarity as attention proxy + value norms
                # This preserves VATP's key insight while being practical

                # NUMERICAL STABILITY: Check input before normalization
                x_norm_check = x.norm(dim=-1, p=2)
                if (x_norm_check < 1e-8).any():
                    # Some tokens have near-zero norm, add small epsilon to avoid NaN in normalize
                    x_safe = x + 1e-8
                else:
                    x_safe = x

                # Compute self-similarity (cosine similarity between tokens)
                # Normalize activations
                x_norm = F.normalize(x_safe, p=2, dim=-1, eps=1e-8)  # [batch, seq, hidden]

                # Compute pairwise similarity (approximates attention pattern)
                # similarity[i,j] = how similar token i is to token j
                similarity = torch.matmul(x_norm, x_norm.transpose(-2, -1))  # [batch, seq, seq]

                # Average similarity to all other tokens (like avg attention received)
                avg_similarity = similarity.mean(dim=-1)  # [batch, seq]

                # Compute value norms (activation magnitude)
                value_norms = x.norm(dim=-1, p=2) / (self.hidden_size ** 0.5)

                # Approximate VATP: similarity × value_norm
                # Intuition:
                # - High similarity → token is contextually relevant
                # - High value norm → token has strong signal
                # - Product → tokens that are both relevant and strong
                importance = avg_similarity * value_norms

                # Re-normalize to [0, 1] for stability
                importance_min = importance.min()
                importance_max = importance.max()
                importance_range = importance_max - importance_min
                # Prevent division by zero
                if importance_range < 1e-10:
                    importance = torch.ones_like(importance) * 0.5
                else:
                    importance = (importance - importance_min) / (importance_range + 1e-10)

        elif self.routing_strategy == 'learned':
            # Learnable importance predictor (TokenButler-style)
            # Achieves 70-75% accuracy with only 1-1.2% extra params
            importance = self.importance_net(x).squeeze(-1)
            importance = torch.sigmoid(importance)  # Normalize to [0, 1]

        elif self.routing_strategy == 'attention':
            # Attention-only method (NOT RECOMMENDED based on EMNLP'24)
            # Attention sinks have high scores but low contribution
            if attention_scores is not None:
                importance = attention_scores.mean(dim=1).mean(dim=-2)
            else:
                raise ValueError("attention_scores required for 'attention' strategy")

        elif self.routing_strategy == 'hybrid':
            # Multi-signal combination
            # Combines norm, attention (if available), and variance
            norm_score = x.norm(dim=-1, p=2) / (self.hidden_size ** 0.5)

            if attention_scores is not None:
                attn_score = attention_scores.mean(dim=1).mean(dim=-2)
            else:
                attn_score = torch.ones_like(norm_score)

            var_score = x.var(dim=-1)

            # Normalize each component
            norm_score = (norm_score - norm_score.min()) / (norm_score.max() - norm_score.min() + 1e-10)
            attn_score = (attn_score - attn_score.min()) / (attn_score.max() - attn_score.min() + 1e-10)
            var_score = (var_score - var_score.min()) / (var_score.max() - var_score.min() + 1e-10)

            # Weighted combination (equal weights)
            importance = 0.4 * norm_score + 0.3 * attn_score + 0.3 * var_score

        else:
            raise ValueError(f"Unknown routing strategy: {self.routing_strategy}")

        # NUMERICAL STABILITY: Check for NaN/Inf in importance scores
        if torch.isnan(importance).any() or torch.isinf(importance).any():
            print(f"[ERROR] NaN/Inf detected in importance computation (strategy={self.routing_strategy})")
            print(f"  Input x: NaN={torch.isnan(x).any()}, Inf={torch.isinf(x).any()}")
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"  x range: min={x[~torch.isnan(x) & ~torch.isinf(x)].min():.2e}, max={x[~torch.isnan(x) & ~torch.isinf(x)].max():.2e}")
            print(f"  importance NaN count: {torch.isnan(importance).sum()}, Inf count: {torch.isinf(importance).sum()}")
            # Replace NaN/Inf with safe fallback (uniform importance)
            importance = torch.nan_to_num(importance, nan=0.5, posinf=1.0, neginf=0.0)
            print(f"  Replaced with safe values, new range: [{importance.min():.2e}, {importance.max():.2e}]")

        return importance

    def route_tokens(self, importance, temperature=1.0, hard=True):
        """
        Route tokens to subspaces based on importance.

        Args:
            importance: Importance scores [batch, seq_len] or [seq_len]
            temperature: Temperature for soft routing (lower = harder decisions)
            hard: If True, use hard routing (discrete). If False, use soft routing (weighted)

        Returns:
            routing: Routing assignments [batch, seq_len] (hard) or [batch, seq_len, n_subspaces] (soft)
        """
        # NUMERICAL STABILITY: Check importance at entry
        has_nan = torch.isnan(importance).any()
        has_inf = torch.isinf(importance).any()
        if has_nan or has_inf:
            print(f"[ERROR] route_tokens called with NaN/Inf importance!")
            print(f"  NaN count: {torch.isnan(importance).sum()}, Inf count: {torch.isinf(importance).sum()}")
            # Clean it here as last resort
            importance = torch.nan_to_num(importance, nan=0.5, posinf=1.0, neginf=0.0)

        # NUMERICAL STABILITY: Handle uniform importance case BEFORE routing
        # This applies to ALL routing strategies (advanced and default)
        importance_min = importance.min()
        importance_max = importance.max()
        importance_range = importance_max - importance_min

        # Use higher threshold to catch floating point precision issues
        if importance_range < 1e-6:
            # All importance values are the same (uniform distribution)
            # Return uniform routing for soft, or assign all to subspace 0 for hard
            print(f"[INFO] Uniform importance detected (range={importance_range:.2e}). Using uniform routing.")
            if not hard:
                batch_size = importance.size(0) if importance.dim() > 1 else 1
                seq_len = importance.size(-1) if importance.dim() > 1 else importance.size(0)
                if importance.dim() == 1:
                    routing_probs = torch.ones(seq_len, self.n_subspaces, device=importance.device) / self.n_subspaces
                else:
                    routing_probs = torch.ones(batch_size, seq_len, self.n_subspaces, device=importance.device) / self.n_subspaces
                return routing_probs
            else:
                # For hard routing, assign all to subspace 0
                return torch.zeros_like(importance, dtype=torch.long)

        # Use advanced router if specified
        if self.advanced_router is not None:
            if hard:
                routing = self.advanced_router.route_hard(importance)
            else:
                routing = self.advanced_router.route_soft(importance, temperature=temperature)

            # CRITICAL: Check if advanced router produced NaN
            if not hard:
                if torch.isnan(routing).any() or torch.isinf(routing).any():
                    print(f"[ERROR] Advanced router produced NaN/Inf routing weights!")
                    print(f"  Router type: {type(self.advanced_router)}")
                    print(f"  Importance was valid: min={importance.min():.2e}, max={importance.max():.2e}")
                    print(f"  Routing NaN count: {torch.isnan(routing).sum()}")
                    print(f"  Falling back to uniform routing")
                    # Fallback to uniform routing
                    batch_size = importance.size(0) if importance.dim() > 1 else 1
                    seq_len = importance.size(-1) if importance.dim() > 1 else importance.size(0)
                    if importance.dim() == 1:
                        routing = torch.ones(seq_len, self.n_subspaces, device=importance.device) / self.n_subspaces
                    else:
                        routing = torch.ones(batch_size, seq_len, self.n_subspaces, device=importance.device) / self.n_subspaces

            # Update statistics
            if self.training and hard:
                for i in range(self.n_subspaces):
                    self.routing_counts[i] += (routing == i).sum().float()
                self.total_tokens += routing.numel()

            return routing

        # Default threshold-based routing
        # Normalize importance to [0, 1]
        # Note: uniform case already handled above
        normalized_importance = (importance - importance_min) / (importance_range + val_epsilon)

        if hard:
            # Hard routing: assign each token to one subspace
            routing = torch.zeros_like(importance, dtype=torch.long)
            for i, threshold in enumerate(self.thresholds):
                routing = torch.where(normalized_importance >= threshold,
                                     torch.tensor(i + 1, device=routing.device),
                                     routing)

            # Update statistics
            if self.training:
                for i in range(self.n_subspaces):
                    self.routing_counts[i] += (routing == i).sum().float()
                self.total_tokens += routing.numel()

            return routing
        else:
            # Soft routing: compute weighted combination
            # Create logits based on distance to thresholds
            routing_logits = torch.zeros(*importance.shape, self.n_subspaces, device=importance.device)

            for i in range(self.n_subspaces):
                if i == 0:
                    # Low importance: distance from 0
                    routing_logits[..., i] = -torch.abs(normalized_importance - 0)
                elif i == self.n_subspaces - 1:
                    # High importance: distance from 1
                    routing_logits[..., i] = -torch.abs(normalized_importance - 1)
                else:
                    # Mid importance: distance from mid threshold
                    mid_point = self.thresholds[i - 1]
                    routing_logits[..., i] = -torch.abs(normalized_importance - mid_point)

            # NUMERICAL STABILITY: Check routing_logits before softmax
            if torch.isnan(routing_logits).any() or torch.isinf(routing_logits).any():
                print(f"[ERROR] NaN/Inf in routing_logits before softmax")
                print(f"  routing_logits NaN: {torch.isnan(routing_logits).sum()}, Inf: {torch.isinf(routing_logits).sum()}")
                print(f"  normalized_importance: NaN={torch.isnan(normalized_importance).any()}, range=[{normalized_importance.min():.2e}, {normalized_importance.max():.2e}]")
                routing_logits = torch.nan_to_num(routing_logits, nan=-10.0, posinf=10.0, neginf=-10.0)

            # Apply temperature and softmax
            scaled_logits = routing_logits / max(temperature, 1e-6)  # Prevent division by very small temperature

            # NUMERICAL STABILITY: Clamp scaled logits to prevent overflow in exp()
            scaled_logits = torch.clamp(scaled_logits, min=-20.0, max=20.0)

            routing_probs = F.softmax(scaled_logits, dim=-1)

            # NUMERICAL STABILITY: Final check for NaN in routing_probs
            if torch.isnan(routing_probs).any() or torch.isinf(routing_probs).any():
                print(f"[ERROR] NaN/Inf in routing_probs after softmax!")
                print(f"  This should not happen. Falling back to uniform distribution.")
                routing_probs = torch.ones_like(routing_probs) / self.n_subspaces

            return routing_probs

    def get_routing_distribution(self):
        """Get the distribution of tokens across subspaces."""
        if self.total_tokens == 0:
            # Return uniform distribution when no tokens have been routed yet
            return torch.ones(self.n_subspaces, device=self.routing_counts.device) / self.n_subspaces

        distribution = self.routing_counts / self.total_tokens
        # Ensure numerical stability: clamp to prevent exact zeros
        return torch.clamp(distribution, min=1e-8)

    def reset_statistics(self):
        """Reset routing statistics."""
        self.routing_counts.zero_()
        self.total_tokens.zero_()


class MultiSubspaceSVDLayer(nn.Module):
    """
    Multi-Subspace SVD Transform Layer with Dynamic Routing.

    This layer maintains multiple SVD subspaces with different ranks and
    dynamically routes tokens to appropriate subspaces based on importance.

    Args:
        gammas: List of gamma values for each subspace [gamma_low, gamma_mid, gamma_high]
        n_subspaces: Number of subspaces (default: 3)
        routing_strategy: Strategy for token routing ('norm', 'learned', 'attention')
        training_mode: Training mode ('joint', 'stage1_subspace', 'stage2_routing')
        Other args: Same as SVDTransformLayer
    """

    def __init__(self,
                 gammas=None,  # List of gammas for each subspace
                 n_subspaces=3,
                 SEQ_LEN=None,
                 beta=None,
                 input_size=None,
                 output_size=None,
                 weight_size=None,
                 weight=None,
                 bias=None,
                 name=None,
                 device=None,
                 routing_strategy='norm',
                 learnable_thresholds=False,
                 use_soft_routing=False,
                 routing_temperature=1.0,
                 load_balance_weight=0.01,
                 advanced_routing=None,
                 advanced_routing_kwargs=None):
        super(MultiSubspaceSVDLayer, self).__init__()

        assert gammas is not None and len(gammas) == n_subspaces, \
            f"Must provide {n_subspaces} gamma values"

        self.n_subspaces = n_subspaces
        self.beta = beta
        self.use_soft_routing = use_soft_routing
        self.routing_temperature = routing_temperature
        self.load_balance_weight = load_balance_weight

        # Create parameter list for gammas (all trainable)
        # CRITICAL: Must explicitly set requires_grad=True!
        self.gammas = nn.ParameterList([
            nn.Parameter(torch.tensor(g, dtype=computeSVD_dtype, device=device),
                        requires_grad=True)
            for g in gammas
        ])

        if name:
            self.name = name

        # Original linear layer
        if bias is None:
            self.ori = nn.Linear(input_size, output_size, bias=False).to(device)
        else:
            self.ori = nn.Linear(input_size, output_size, bias=True).to(device)
            self.ori.bias = nn.Parameter(bias).to(device)
        self.ori.weight = nn.Parameter(weight).to(device)

        # Size info for compression calculation
        self.ori_weight_size = weight_size
        in_plus_out_size = input_size + output_size
        nblocks = input_size / SEQ_LEN
        self.nblocks_total_size = in_plus_out_size * nblocks

        # Token router
        self.router = TokenRouter(
            hidden_size=output_size,
            n_subspaces=n_subspaces,
            routing_strategy=routing_strategy,
            learnable_thresholds=learnable_thresholds,
            advanced_routing=advanced_routing,
            advanced_routing_kwargs=advanced_routing_kwargs,
            device=device
        )

        # Cache for routing visualization (only in eval mode)
        self.register_buffer('cached_routing', None)
        self.register_buffer('cached_importance', None)

    def forward(self, x):
        """
        Forward pass with dynamic subspace routing.

        Training mode: Uses soft routing (weighted combination) for differentiability
        Inference mode: Uses hard routing (discrete assignment) for efficiency
        """
        # Apply original linear transformation
        x = self.ori(x).to(computeSVD_dtype)

        # Handle dimension
        if x.dim() == 3:
            x = x.squeeze(0)
            squeeze_need = 1
        else:
            squeeze_need = 0

        assert x.dim() == 2, f"Expected 2D tensor, got {x.dim()}D"
        m, n = x.shape
        full_rank = min(m, n)

        # Compute token importance
        importance = self.router.compute_importance(x)

        # Cache for analysis (eval mode only)
        if not self.training:
            self.cached_importance = importance.detach().clone()

        if self.training or self.use_soft_routing:
            # TRAINING MODE: Soft routing (weighted combination)
            routing_weights = self.router.route_tokens(
                importance,
                temperature=self.routing_temperature,
                hard=False
            )  # Shape: [m, n_subspaces]

            # MEMORY OPTIMIZATION: Initialize output tensor
            x_transformed = torch.zeros_like(x, dtype=model_load_dtype)

            for subspace_id, gamma in enumerate(self.gammas):
                # Determine SVD rank for this subspace
                gamma_range = gamma.detach()
                gamma_range = int(gamma_range.int() + 5)
                gamma_range = min(full_rank, max(1, gamma_range))

                # Check for numerical issues before SVD
                if torch.isnan(x).any() or torch.isinf(x).any():
                    # Fallback: replace problematic values
                    x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
                    x = torch.where(torch.isinf(x), torch.zeros_like(x), x)

                # Get routing weight for this subspace
                weight = routing_weights[:, subspace_id].unsqueeze(-1)  # [m, 1]

                # Skip if weight is very small (memory optimization)
                if weight.abs().max() < 1e-6:
                    continue

                # Compute SVD with error handling
                try:
                    U, S, V = stable_lowrank_SVD.apply(x, gamma_range)
                except RuntimeError as e:
                    # If SVD fails, use fallback: just pass through
                    print(f"Warning: SVD failed for subspace {subspace_id}, using identity: {e}")
                    fallback = (weight * x).to(model_load_dtype)
                    x_transformed = x_transformed + fallback
                    del fallback
                    continue

                # MEMORY OPTIMIZATION: Efficient operations but no in-place on gradient tensors
                sequence = torch.arange(1, len(S) + 1, device=x.device, dtype=x.dtype)
                real_gamma = min(len(S), max(1, gamma))

                # Apply truncation (no in-place operations!)
                Trunc = 0.5 * torch.tanh(self.beta * (real_gamma - sequence)) + 0.5
                S_transformed = S * Trunc

                # Reconstruct using memory-efficient matrix multiplication
                # x_sub = U @ diag(S_transformed) @ V^T
                # Optimize: (U @ diag(S)) @ V^T saves memory
                US = U * S_transformed.unsqueeze(0)  # [m, rank]
                x_sub = torch.matmul(US, V.T)  # [m, n]

                # Free intermediate tensors immediately
                del U, S, V, sequence, Trunc, S_transformed, US

                # Weight and accumulate (no in-place on x_transformed - needs gradients!)
                weighted_sub = (weight * x_sub).to(model_load_dtype)
                x_transformed = x_transformed + weighted_sub
                del x_sub, weight, weighted_sub

            real_x = x_transformed

        else:
            # INFERENCE MODE: Hard routing (discrete assignment)
            routing = self.router.route_tokens(importance, hard=True)  # Shape: [m]

            # Cache routing for analysis
            self.cached_routing = routing.detach().clone()

            # Process each subspace separately
            x_transformed = torch.zeros_like(x)

            for subspace_id, gamma in enumerate(self.gammas):
                # Get tokens assigned to this subspace
                mask = (routing == subspace_id)

                if not mask.any():
                    continue  # Skip if no tokens assigned

                # Extract tokens for this subspace
                x_sub = x[mask]

                # Determine SVD rank
                gamma_range = gamma.detach()
                gamma_range = int(gamma_range.int() + 5)
                gamma_range = min(min(x_sub.shape), max(1, gamma_range))

                # Compute SVD
                U, S, V = stable_lowrank_SVD.apply(x_sub, gamma_range)
                sequence = torch.arange(1, len(S) + 1).to(x.device)
                real_gamma = min(len(S), max(1, gamma))

                # Apply truncation
                Trunc = 0.5 * torch.tanh(self.beta * (real_gamma - sequence)) + 0.5
                S_transformed = S * Trunc
                S_diag = torch.diag_embed(S_transformed)

                # Reconstruct and assign back
                x_sub_transformed = torch.matmul(torch.matmul(U, S_diag), V.T)
                x_transformed[mask] = x_sub_transformed

            real_x = x_transformed.to(model_load_dtype)

        if squeeze_need == 1:
            real_x = real_x.unsqueeze(0)

        return real_x

    def get_routing_stats(self):
        """Get routing statistics for this layer."""
        return {
            'routing_distribution': self.router.get_routing_distribution().cpu().numpy(),
            'gammas': [g.item() for g in self.gammas],
            'cached_routing': self.cached_routing.cpu().numpy() if self.cached_routing is not None else None,
            'cached_importance': self.cached_importance.cpu().numpy() if self.cached_importance is not None else None
        }

    def reset_routing_stats(self):
        """Reset routing statistics."""
        self.router.reset_statistics()
        self.cached_routing = None
        self.cached_importance = None


def compute_load_balance_loss(model, target_distribution=None):
    """
    Compute load balance loss to encourage balanced routing across subspaces.

    Similar to the auxiliary loss in Mixture of Experts (MoE) models.

    Args:
        model: Model with MultiSubspaceSVDLayer or SharedParamMultiSubspaceSVDLayer modules
        target_distribution: Target distribution across subspaces (default: uniform)

    Returns:
        loss: Load balance loss (scalar)
    """
    total_loss = 0.0
    n_layers = 0

    for name, module in model.named_modules():
        if isinstance(module, (MultiSubspaceSVDLayer, SharedParamMultiSubspaceSVDLayer)):
            distribution = module.router.get_routing_distribution()

            if target_distribution is None:
                # Default: uniform distribution
                target = torch.ones_like(distribution) / len(distribution)
            else:
                target = target_distribution

            # KL divergence loss with numerical stability
            # Add epsilon to prevent log(0) which causes NaN
            eps = 1e-8
            distribution_safe = torch.clamp(distribution, min=eps)
            target_safe = torch.clamp(target, min=eps)

            # Use F.kl_div with log_target=False (target is not in log space)
            loss = F.kl_div(
                distribution_safe.log(),
                target_safe,
                reduction='batchmean'
            )

            # Check for NaN and skip if invalid
            if not torch.isnan(loss) and not torch.isinf(loss):
                total_loss += loss
                n_layers += 1

    if n_layers == 0:
        return torch.tensor(0.0)

    return total_loss / n_layers


def get_model_routing_statistics(model):
    """
    Collect routing statistics from all MultiSubspaceSVDLayer and SharedParamMultiSubspaceSVDLayer modules.

    Returns:
        stats: Dictionary with routing statistics per layer
    """
    stats = {}

    for name, module in model.named_modules():
        if isinstance(module, (MultiSubspaceSVDLayer, SharedParamMultiSubspaceSVDLayer)):
            stats[name] = module.get_routing_stats()

    return stats


def reset_model_routing_statistics(model):
    """Reset routing statistics for all MultiSubspaceSVDLayer modules."""
    for name, module in model.named_modules():
        if isinstance(module, MultiSubspaceSVDLayer) or isinstance(module, SharedParamMultiSubspaceSVDLayer):
            module.reset_routing_stats()


class SharedParamMultiSubspaceSVDLayer(nn.Module):
    """
    Parameter-Shared Multi-Subspace SVD Layer for TRAINING.

    !!主要优势!!:
    - 所有子空间共享同一个 U, V 矩阵（只计算一次SVD）
    - 每个子空间只有不同的 gamma 截断参数
    - 显存占用: ~66% 减少 (相比标准MultiSubspaceSVDLayer)
    - 训练速度: ~30% 加快 (只计算一次SVD)

    数学原理:
        标准方式: W_k = U_k Σ_k V_k^T  (每个子空间独立SVD)
        参数共享: W_k = U Σ_k V^T      (共享U,V，只有Σ截断不同)

        其中 Σ_k = Σ ⊙ Trunc(γ_k)
        Trunc(γ_k) = 0.5 * tanh(β(γ_k - sequence)) + 0.5

    参数量对比:
        标准方式: N × (m·r + r + n·r) = N·r·(m+n+1)
        参数共享: (m·r + n·r) + N·r = r·(m+n+N)

        节省比例: 1 - (m+n+N)/(N·(m+n+1))
        例如 N=3, m=2048, n=4096:
            标准: 3 × 128 × 6145 = 2,359,296
            共享: 128 × (6144 + 3) = 786,816
            节省: 66.7%

    Args:
        Same as MultiSubspaceSVDLayer, plus:
        svd_rank: Rank for the shared SVD (default: max(gammas) + 10)
    """

    def __init__(self,
                 gammas=None,
                 n_subspaces=3,
                 SEQ_LEN=None,
                 beta=None,
                 input_size=None,
                 output_size=None,
                 weight_size=None,
                 weight=None,
                 bias=None,
                 name=None,
                 device=None,
                 routing_strategy='norm',
                 learnable_thresholds=False,
                 use_soft_routing=False,
                 routing_temperature=1.0,
                 load_balance_weight=0.01,
                 advanced_routing=None,
                 advanced_routing_kwargs=None,
                 svd_rank=None,
                 use_gradient_checkpointing=False):
        super(SharedParamMultiSubspaceSVDLayer, self).__init__()

        assert gammas is not None and len(gammas) == n_subspaces, \
            f"Must provide {n_subspaces} gamma values"

        self.n_subspaces = n_subspaces
        self.beta = beta
        self.use_soft_routing = use_soft_routing
        self.routing_temperature = routing_temperature
        self.load_balance_weight = load_balance_weight
        self.input_size = input_size
        self.output_size = output_size
        self.use_gradient_checkpointing = use_gradient_checkpointing

        if name:
            self.name = name

        # Gamma parameters (trainable)
        # CRITICAL: Must explicitly set requires_grad=True in Parameter constructor!
        # torch.tensor() creates tensors with requires_grad=False by default

        # VALIDATION: Check gamma values before creating parameters
        for i, g in enumerate(gammas):
            g_val = g.item() if isinstance(g, torch.Tensor) else g
            if not (0 < g_val < 10000):  # Sanity check
                raise ValueError(f"Invalid gamma[{i}] = {g_val}. Gamma must be positive and finite.")
            if torch.isnan(torch.tensor(g_val)):
                raise ValueError(f"Gamma[{i}] is NaN! Cannot initialize layer with NaN gamma values.")

        self.gammas = nn.ParameterList([
            nn.Parameter(torch.tensor(g, dtype=computeSVD_dtype, device=device),
                        requires_grad=True)
            for g in gammas
        ])

        print(f"[SharedParam] Initialized gammas: {[f'{g.item():.2f}' for g in self.gammas]}")

        # ========================================
        # 关键: 只计算一次SVD，所有子空间共享!
        # ========================================

        # Determine SVD rank (use max gamma + buffer)
        if svd_rank is None:
            max_gamma = max([g if isinstance(g, (int, float)) else g.item() for g in gammas])
            svd_rank = int(max_gamma) + 10

        # Ensure svd_rank doesn't exceed matrix dimensions
        svd_rank = min(svd_rank, min(output_size, input_size))
        self.svd_rank = svd_rank

        print(f"[SharedParam] Layer {name}: Computing shared SVD with rank={svd_rank}")

        # Compute SVD once on the original weight
        with torch.no_grad():
            # Move to appropriate dtype for SVD
            weight_svd = weight.to(computeSVD_dtype)

            # DIAGNOSTIC: Print weight statistics BEFORE SVD
            print(f"\n[SVD DIAGNOSTIC] Layer {name}:")
            print(f"  - Weight shape: {weight_svd.shape}")
            print(f"  - Weight range: [{weight_svd.min().item():.6f}, {weight_svd.max().item():.6f}]")
            print(f"  - Weight mean/std: {weight_svd.mean().item():.6f} / {weight_svd.std().item():.6f}")
            print(f"  - Weight norm: {weight_svd.norm().item():.2f}")
            print(f"  - SVD rank to compute: {svd_rank}")

            # Compute low-rank SVD
            try:
                U, S, Vh = torch.linalg.svd(weight_svd, full_matrices=False)

                # DIAGNOSTIC: Print singular value statistics BEFORE truncation
                print(f"  - Full S range: [{S.min().item():.6f}, {S.max().item():.6f}]")
                print(f"  - Full S mean/std: {S.mean().item():.6f} / {S.std().item():.6f}")
                print(f"  - Top 5 singular values: {S[:5].tolist()}")
                print(f"  - S values at rank {svd_rank}: {S[svd_rank-5:svd_rank].tolist() if svd_rank >= 5 else S[:svd_rank].tolist()}")

                # Truncate to desired rank
                U = U[:, :svd_rank]   # [output_size, svd_rank]
                S = S[:svd_rank]       # [svd_rank]
                V = Vh[:svd_rank, :]   # [svd_rank, input_size]

                # DIAGNOSTIC: Compute reconstruction error
                weight_reconstructed = U @ torch.diag(S) @ V
                reconstruction_error = (weight_svd - weight_reconstructed).norm() / weight_svd.norm()
                print(f"  - Truncated S range: [{S.min().item():.6f}, {S.max().item():.6f}]")
                print(f"  - Reconstruction error: {reconstruction_error.item():.6f}")
                print(f"[SharedParam] SVD shapes: U={U.shape}, S={S.shape}, V={V.shape}")

            except Exception as e:
                print(f"[SharedParam] Warning: SVD failed ({e}), using random initialization")
                U = torch.randn(output_size, svd_rank, dtype=computeSVD_dtype, device=device) * 0.01
                S = torch.ones(svd_rank, dtype=computeSVD_dtype, device=device)
                V = torch.randn(svd_rank, input_size, dtype=computeSVD_dtype, device=device) * 0.01

        # Register as buffers (non-trainable, shared across subspaces)
        self.register_buffer('U_shared', U.to(device))
        self.register_buffer('S_shared', S.to(device))
        self.register_buffer('V_shared', V.to(device))

        # Bias (if any)
        if bias is not None:
            self.register_buffer('bias', bias.to(device))
        else:
            self.bias = None

        # Token router
        self.router = TokenRouter(
            hidden_size=output_size,
            n_subspaces=n_subspaces,
            routing_strategy=routing_strategy,
            learnable_thresholds=learnable_thresholds,
            advanced_routing=advanced_routing,
            advanced_routing_kwargs=advanced_routing_kwargs,
            device=device
        )

        # Size info for compression calculation
        self.ori_weight_size = weight_size
        in_plus_out_size = input_size + output_size
        nblocks = input_size / SEQ_LEN
        self.nblocks_total_size = in_plus_out_size * nblocks

        # Cache for routing visualization
        self.register_buffer('cached_routing', None)
        self.register_buffer('cached_importance', None)

        # DIAGNOSTIC: Forward pass counter for periodic logging
        self.forward_counter = 0

        # OPTIMIZATION: Precompute sequence tensor to avoid recomputation in forward pass
        # This saves significant computation especially for large svd_rank
        sequence_tensor = torch.arange(1, svd_rank + 1, dtype=torch.float32, device=device)
        self.register_buffer('sequence_tensor', sequence_tensor)

        # Print parameter savings
        standard_params = n_subspaces * svd_rank * (output_size + input_size + 1)
        shared_params = svd_rank * (output_size + input_size) + n_subspaces
        reduction = (1 - shared_params / standard_params) * 100
        print(f"[SharedParam] Parameter reduction: {reduction:.1f}% "
              f"({standard_params:,} → {shared_params:,})")

    def _forward_impl(self, x, routing_weights):
        """
        Checkpointable computation: Apply multi-subspace SVD with routing.

        This function is separated to enable gradient checkpointing.
        """
        batch_size, seq_len, hidden_size = x.shape
        device = x.device
        input_dtype = x.dtype

        x_flat = x.view(-1, hidden_size)
        routing_weights_flat = routing_weights.view(-1, self.n_subspaces)

        # Convert shared parameters
        V_shared = self.V_shared.to(input_dtype)
        S_shared = self.S_shared.to(input_dtype)
        U_shared = self.U_shared.to(input_dtype)

        # NUMERICAL STABILITY: Clamp singular values to prevent extreme values
        # Must match the clamping in forward() to prevent activation explosion
        # BALANCED: max=100 prevents overflow while maintaining reasonable capacity
        # Higher values (1000, 10000) cause NaN cascade through LayerNorm
        S_shared = torch.clamp(S_shared, min=1e-6, max=100.0)

        output = None

        # IMPORTANT: When gradient checkpointing is enabled, we CANNOT use out= parameter
        # because it doesn't support automatic differentiation (autograd)
        # So we use buffer reuse only when checkpointing is disabled
        use_buffer_reuse = not self.use_gradient_checkpointing

        if use_buffer_reuse:
            # Pre-allocate buffer for memory optimization
            xV_buffer = torch.empty(batch_size * seq_len, self.svd_rank,
                                   dtype=input_dtype, device=device)

        for subspace_id, gamma in enumerate(self.gammas):
            # NUMERICAL STABILITY: Check if gamma parameter is NaN
            if torch.isnan(gamma).any() or torch.isinf(gamma).any():
                print(f"[CRITICAL] Gamma parameter for subspace {subspace_id} is NaN/Inf!")
                print(f"  This indicates gradient explosion or numerical instability in gamma updates.")
                print(f"  Resetting gamma to default value.")
                # Reset to a safe default value based on subspace position
                default_gamma = (subspace_id + 1) * (self.svd_rank / (self.n_subspaces + 1))
                with torch.no_grad():
                    gamma.fill_(default_gamma)

            weight = routing_weights_flat[:, subspace_id].unsqueeze(-1)

            # Compute truncation with numerical stability
            sequence = self.sequence_tensor.to(input_dtype)
            gamma_val = gamma.clamp(min=1.0, max=float(self.svd_rank))

            # NUMERICAL STABILITY: Clamp beta argument to prevent overflow
            # tanh saturates at ~±20, so clamp input to tanh to [-20, 20]
            beta_arg = self.beta * (gamma_val - sequence)
            beta_arg = torch.clamp(beta_arg, min=-20.0, max=20.0)
            trunc = 0.5 * torch.tanh(beta_arg) + 0.5

            S_truncated = S_shared * trunc

            # NUMERICAL STABILITY: Check for NaN in truncation
            if torch.isnan(S_truncated).any() or torch.isinf(S_truncated).any():
                print(f"[ERROR] NaN/Inf in S_truncated for subspace {subspace_id}")
                print(f"  gamma_val: {gamma_val.item()}, beta: {self.beta}")
                print(f"  S_shared range: [{S_shared.min().item():.2e}, {S_shared.max().item():.2e}]")
                print(f"  trunc range: [{trunc.min().item():.2e}, {trunc.max().item():.2e}]")
                S_truncated = torch.nan_to_num(S_truncated, nan=1e-6, posinf=100.0, neginf=-100.0)

            # Reconstruct: choose between buffer reuse (faster) or regular ops (autograd-safe)
            if use_buffer_reuse:
                # Buffer reuse version: faster but incompatible with gradient checkpointing
                torch.mm(x_flat, V_shared.T, out=xV_buffer)
                # BALANCED: ±200 allows capacity while preventing overflow cascade
                xV_buffer.clamp_(min=-200.0, max=200.0)
                xV_buffer.mul_(S_truncated.unsqueeze(0))
                # BALANCED: ±200 after S multiplication (S_max=100, so max ~20,000 → clamped to 200)
                xV_buffer.clamp_(min=-200.0, max=200.0)
                x_sub = xV_buffer @ U_shared.T
                # BALANCED: ±500 final output - safe for LayerNorm, prevents NaN cascade
                x_sub.clamp_(min=-500.0, max=500.0)
                x_sub.mul_(weight)
            else:
                # Regular version: compatible with gradient checkpointing
                xV = x_flat @ V_shared.T
                # BALANCED: ±200 allows capacity while preventing overflow cascade
                xV = torch.clamp(xV, min=-200.0, max=200.0)
                xV = xV * S_truncated.unsqueeze(0)
                # BALANCED: ±200 after S multiplication (S_max=100, so max ~20,000 → clamped to 200)
                xV = torch.clamp(xV, min=-200.0, max=200.0)
                x_sub = xV @ U_shared.T
                # BALANCED: ±500 final output - safe for LayerNorm, prevents NaN cascade
                x_sub = torch.clamp(x_sub, min=-500.0, max=500.0)
                x_sub = x_sub * weight

            if output is None:
                output = x_sub
            else:
                output.add_(x_sub)

        # NUMERICAL STABILITY: Final output check
        output_reshaped = output.view(batch_size, seq_len, self.output_size)

        if torch.isnan(output_reshaped).any() or torch.isinf(output_reshaped).any():
            print(f"[ERROR] NaN/Inf in final output of SharedParamMultiSubspaceSVDLayer")
            print(f"  Output shape: {output_reshaped.shape}")
            nan_count = torch.isnan(output_reshaped).sum().item()
            inf_count = torch.isinf(output_reshaped).sum().item()
            total_elements = output_reshaped.numel()
            print(f"  NaN count: {nan_count} / {total_elements} ({100*nan_count/total_elements:.1f}%)")
            print(f"  Inf count: {inf_count} / {total_elements} ({100*inf_count/total_elements:.1f}%)")

            # Only try to print range if there are valid values
            valid_mask = ~torch.isnan(output_reshaped) & ~torch.isinf(output_reshaped)
            if valid_mask.any():
                valid_values = output_reshaped[valid_mask]
                print(f"  Valid values range: [{valid_values.min().item():.2e}, {valid_values.max().item():.2e}]")
            else:
                print(f"  WARNING: ALL outputs are NaN/Inf! This is a critical failure.")
                print(f"  Check gamma values, routing weights, and input data.")

            # Replace NaN/Inf with safe values
            # BALANCED: ±500 is safe for downstream LayerNorm and prevents NaN cascade
            output_reshaped = torch.nan_to_num(output_reshaped, nan=0.0, posinf=500.0, neginf=-500.0)

        # NUMERICAL STABILITY: Clamp output to prevent extreme activations
        # This prevents downstream layers (especially LayerNorm) from getting unstable inputs
        # BALANCED: ±500 prevents NaN cascade while maintaining model capacity
        output_reshaped = torch.clamp(output_reshaped, min=-500.0, max=500.0)

        return output_reshaped

    def forward(self, x):
        """
        Forward pass with shared U,V and different gamma truncations.

        核心思想:
            对于每个子空间 k:
                1. 使用共享的 U, S, V
                2. 应用该子空间的 gamma_k 截断
                3. 重建: x_k = x @ V^T @ diag(S ⊙ Trunc_k) @ U^T
        """
        # CRITICAL: Check gamma parameters at the start of forward pass
        for i, gamma in enumerate(self.gammas):
            if torch.isnan(gamma).any() or torch.isinf(gamma).any():
                print(f"[CRITICAL ERROR] Gamma[{i}] is NaN/Inf at START of forward pass!")
                print(f"  Gamma value: {gamma.item()}")
                print(f"  This indicates gamma was never properly initialized or was corrupted.")
                # Reset to safe default
                default_gamma = float((i + 1) * (self.svd_rank / (self.n_subspaces + 1)))
                print(f"  Resetting to default: {default_gamma}")
                with torch.no_grad():
                    gamma.fill_(default_gamma)

        batch_size, seq_len, hidden_size = x.shape
        device = x.device
        input_dtype = x.dtype  # Get input dtype (could be FP16 or FP32)

        # NUMERICAL STABILITY: Check input for NaN/Inf
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"[ERROR] NaN/Inf in input to SharedParamMultiSubspaceSVDLayer")
            print(f"  This indicates upstream layer is producing invalid outputs!")
            print(f"  Clamping to ±500 to prevent cascade failure")
            x = torch.nan_to_num(x, nan=0.0, posinf=500.0, neginf=-500.0)
            x = torch.clamp(x, min=-500.0, max=500.0)

        # Flatten for processing
        x_flat = x.view(-1, hidden_size)  # [batch*seq, hidden]

        # Convert shared parameters to input dtype for computation
        V_shared = self.V_shared.to(input_dtype)
        S_shared = self.S_shared.to(input_dtype)
        U_shared = self.U_shared.to(input_dtype)

        # NUMERICAL STABILITY: Check SVD parameters
        if torch.isnan(V_shared).any() or torch.isnan(S_shared).any() or torch.isnan(U_shared).any():
            print(f"[ERROR] NaN in SVD parameters! V: {torch.isnan(V_shared).any()}, S: {torch.isnan(S_shared).any()}, U: {torch.isnan(U_shared).any()}")
            # This shouldn't happen - SVD params should be frozen
            return torch.zeros(batch_size, seq_len, self.output_size, device=device, dtype=input_dtype)

        # NUMERICAL STABILITY: Clamp singular values to prevent extreme values
        # Singular values should be in reasonable range to prevent activation explosion
        # BALANCED: max=100 prevents overflow cascade to downstream LayerNorm
        S_shared = torch.clamp(S_shared, min=1e-6, max=100.0)

        # Linear transformation: x_transformed = x @ W^T
        # Where W ≈ U @ diag(S) @ V (SVD approximation)
        # So x_transformed ≈ x @ V^T @ diag(S) @ U^T

        # Compute importance scores for routing
        # NOTE: importance computation is differentiable to allow:
        # 1. Temperature scheduling to work properly (gradients flow through softmax)
        # 2. Routing to adapt based on which tokens benefit from which subspaces
        # 3. Better gradient signal for gamma parameters
        x_approx = x_flat @ V_shared.T

        # NUMERICAL STABILITY: Check intermediate result and clamp
        if torch.isnan(x_approx).any() or torch.isinf(x_approx).any():
            print(f"[ERROR] NaN/Inf after x @ V.T in routing computation")
            print(f"  x_flat range: [{x_flat.min():.2e}, {x_flat.max():.2e}]")
            print(f"  V_shared range: [{V_shared.min():.2e}, {V_shared.max():.2e}]")
            x_approx = torch.nan_to_num(x_approx, nan=0.0, posinf=200.0, neginf=-200.0)
        # BALANCED: ±200 for routing computation (doesn't need full model range)
        x_approx = torch.clamp(x_approx, min=-200.0, max=200.0)

        x_approx = x_approx * S_shared.unsqueeze(0)

        # NUMERICAL STABILITY: Check after scaling and clamp
        if torch.isnan(x_approx).any() or torch.isinf(x_approx).any():
            print(f"[ERROR] NaN/Inf after x_approx * S in routing computation")
            print(f"  S_shared range: [{S_shared.min():.2e}, {S_shared.max():.2e}]")
            x_approx = torch.nan_to_num(x_approx, nan=0.0, posinf=200.0, neginf=-200.0)
        # BALANCED: ±200 after S multiplication (S_max=100)
        x_approx = torch.clamp(x_approx, min=-200.0, max=200.0)

        x_approx = x_approx @ U_shared.T

        # NUMERICAL STABILITY: Check final routing input and clamp
        if torch.isnan(x_approx).any() or torch.isinf(x_approx).any():
            print(f"[ERROR] NaN/Inf after x_approx @ U.T in routing computation")
            print(f"  U_shared range: [{U_shared.min():.2e}, {U_shared.max():.2e}]")
            x_approx = torch.nan_to_num(x_approx, nan=0.0, posinf=200.0, neginf=-200.0)
        # BALANCED: ±200 for final routing input (internal use only)
        x_approx = torch.clamp(x_approx, min=-200.0, max=200.0)

        x_for_routing = x_approx.view(batch_size, seq_len, self.output_size)
        importance = self.router.compute_importance(x_for_routing)

        # Note: x_approx will be freed by Python GC automatically
        # No need for explicit del when not using no_grad()

        # Cache for analysis
        if not self.training:
            self.cached_importance = importance.detach().clone()

        # ========================================
        # 路由决策 (Soft or Hard)
        # ========================================

        if self.training or self.use_soft_routing:
            # TRAINING: Soft routing (weighted combination)
            routing_weights = self.router.route_tokens(
                importance,
                temperature=self.routing_temperature,
                hard=False
            )  # [batch, seq, n_subspaces]

            # MEMORY OPTIMIZATION: Use gradient checkpointing if enabled
            # This trades compute for memory by recomputing activations during backward
            if self.training and self.use_gradient_checkpointing:
                output = checkpoint(self._forward_impl, x, routing_weights, use_reentrant=False)
            else:
                output = self._forward_impl(x, routing_weights)

            # Convert to model dtype for subsequent layers
            real_x = output.to(model_load_dtype)

        else:
            # INFERENCE: Hard routing (discrete assignment)
            routing = self.router.route_tokens(importance, hard=True)  # [batch, seq]
            routing_flat = routing.view(-1)  # [batch*seq]

            # Cache for analysis
            self.cached_routing = routing.detach().clone()

            # Initialize output with correct dtype
            output = torch.zeros(batch_size * seq_len, self.output_size,
                               device=device, dtype=input_dtype)

            # Process each subspace
            for subspace_id, gamma in enumerate(self.gammas):
                # Get tokens assigned to this subspace
                mask = (routing_flat == subspace_id)

                if not mask.any():
                    continue

                # Extract tokens for this subspace
                x_sub = x_flat[mask]  # [n_tokens, hidden]

                # Compute truncation - OPTIMIZATION: use precomputed sequence
                sequence = self.sequence_tensor.to(input_dtype)
                gamma_val = gamma.clamp(min=1.0, max=float(self.svd_rank))
                trunc = 0.5 * torch.tanh(self.beta * (gamma_val - sequence)) + 0.5
                S_truncated = S_shared * trunc

                # Reconstruct
                xV = x_sub @ V_shared.T
                xVS = xV * S_truncated.unsqueeze(0)
                x_sub_transformed = xVS @ U_shared.T

                # Assign back
                output[mask] = x_sub_transformed

            real_x = output.view(batch_size, seq_len, self.output_size).to(model_load_dtype)

        # Add bias
        if self.bias is not None:
            real_x = real_x + self.bias

        # DIAGNOSTIC: Print statistics periodically during training
        if self.training:
            self.forward_counter += 1
            # Print on first pass, then every 10 passes
            if self.forward_counter == 1 or self.forward_counter % 10 == 0:
                layer_name = self.name if hasattr(self, 'name') else 'unknown'
                print(f"\n[ACTIVATION DIAGNOSTIC] Layer: {layer_name}, Forward pass {self.forward_counter}:")
                print(f"  - Output range: [{real_x.min().item():.2e}, {real_x.max().item():.2e}]")
                print(f"  - Output mean/std: {real_x.mean().item():.2e} / {real_x.std().item():.2e}")
                print(f"  - Gamma values: {[f'{g.item():.2f}' for g in self.gammas]}")
                print(f"  - S_shared range: [{self.S_shared.min().item():.2e}, {self.S_shared.max().item():.2e}]")
                if hasattr(self.router, 'get_routing_distribution'):
                    routing_dist = self.router.get_routing_distribution()
                    print(f"  - Routing distribution: {[f'{d:.3f}' for d in routing_dist]}")

        return real_x

    def get_routing_stats(self):
        """Get routing statistics for this layer."""
        return {
            'routing_distribution': self.router.get_routing_distribution().cpu().numpy(),
            'gammas': [g.item() for g in self.gammas],
            'cached_routing': self.cached_routing.cpu().numpy() if self.cached_routing is not None else None,
            'cached_importance': self.cached_importance.cpu().numpy() if self.cached_importance is not None else None,
            'shared_svd_rank': self.svd_rank
        }

    def reset_routing_stats(self):
        """Reset routing statistics."""
        self.router.reset_statistics()
        self.cached_routing = None
        self.cached_importance = None
