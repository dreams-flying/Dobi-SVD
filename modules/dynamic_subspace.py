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
from .stable_svd import stable_lowrank_SVD, computeSVD_dtype, model_load_dtype, val_epsilon


class TokenRouter(nn.Module):
    """
    Token importance computation and routing module.

    Supports multiple routing strategies:
    - 'norm': Fast activation norm-based routing (default)
    - 'learned': Learnable importance predictor
    - 'attention': Attention score-based (requires attention hooks)
    """

    def __init__(self,
                 hidden_size,
                 n_subspaces=3,
                 routing_strategy='norm',
                 learnable_thresholds=False,
                 device=None):
        super(TokenRouter, self).__init__()

        self.hidden_size = hidden_size
        self.n_subspaces = n_subspaces
        self.routing_strategy = routing_strategy
        self.device = device

        # Initialize thresholds for routing
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
            ).to(device)

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
                # Average attention across heads and queries
                avg_attention = attention_scores.mean(dim=1).mean(dim=-2)

                # Compute value norms (use x as value if not provided)
                values = value_vectors if value_vectors is not None else x
                value_norms = values.norm(dim=-1, p=2) / (self.hidden_size ** 0.5)

                # VATP formula
                importance = avg_attention * value_norms
            else:
                # Fallback to L2 norm if attention not available
                importance = x.norm(dim=-1, p=2) / (self.hidden_size ** 0.5)

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
        # Normalize importance to [0, 1]
        importance_min = importance.min()
        importance_max = importance.max()
        normalized_importance = (importance - importance_min) / (importance_max - importance_min + val_epsilon)

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

            # Apply temperature and softmax
            routing_probs = F.softmax(routing_logits / temperature, dim=-1)

            return routing_probs

    def get_routing_distribution(self):
        """Get the distribution of tokens across subspaces."""
        if self.total_tokens == 0:
            return torch.zeros(self.n_subspaces)
        return self.routing_counts / self.total_tokens

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
                 load_balance_weight=0.01):
        super(MultiSubspaceSVDLayer, self).__init__()

        assert gammas is not None and len(gammas) == n_subspaces, \
            f"Must provide {n_subspaces} gamma values"

        self.n_subspaces = n_subspaces
        self.beta = beta
        self.use_soft_routing = use_soft_routing
        self.routing_temperature = routing_temperature
        self.load_balance_weight = load_balance_weight

        # Create parameter list for gammas (all trainable)
        self.gammas = nn.ParameterList([
            nn.Parameter(torch.tensor(g, dtype=computeSVD_dtype)).to(device)
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

            # Compute SVD for each subspace
            x_transformed = torch.zeros_like(x).to(model_load_dtype)

            for subspace_id, gamma in enumerate(self.gammas):
                # Determine SVD rank for this subspace
                gamma_range = gamma.detach()
                gamma_range = int(gamma_range.int() + 5)
                gamma_range = min(full_rank, max(1, gamma_range))

                # Compute SVD
                U, S, V = stable_lowrank_SVD.apply(x, gamma_range)
                sequence = torch.arange(1, len(S) + 1).to(x.device)
                real_gamma = min(len(S), max(1, gamma))

                # Apply truncation
                Trunc = 0.5 * torch.tanh(self.beta * (real_gamma - sequence)) + 0.5
                S_transformed = S * Trunc
                S_diag = torch.diag_embed(S_transformed)

                # Reconstruct
                x_sub = torch.matmul(torch.matmul(U, S_diag), V.T)

                # Weight by routing probability
                weight = routing_weights[:, subspace_id].unsqueeze(-1)  # [m, 1]
                x_transformed += (weight * x_sub).to(model_load_dtype)

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
        model: Model with MultiSubspaceSVDLayer modules
        target_distribution: Target distribution across subspaces (default: uniform)

    Returns:
        loss: Load balance loss (scalar)
    """
    total_loss = 0.0
    n_layers = 0

    for name, module in model.named_modules():
        if isinstance(module, MultiSubspaceSVDLayer):
            distribution = module.router.get_routing_distribution()

            if target_distribution is None:
                # Default: uniform distribution
                target = torch.ones_like(distribution) / len(distribution)
            else:
                target = target_distribution

            # KL divergence loss
            loss = F.kl_div(
                distribution.log(),
                target,
                reduction='batchmean'
            )
            total_loss += loss
            n_layers += 1

    if n_layers == 0:
        return torch.tensor(0.0)

    return total_loss / n_layers


def get_model_routing_statistics(model):
    """
    Collect routing statistics from all MultiSubspaceSVDLayer modules.

    Returns:
        stats: Dictionary with routing statistics per layer
    """
    stats = {}

    for name, module in model.named_modules():
        if isinstance(module, MultiSubspaceSVDLayer):
            stats[name] = module.get_routing_stats()

    return stats


def reset_model_routing_statistics(model):
    """Reset routing statistics for all MultiSubspaceSVDLayer modules."""
    for name, module in model.named_modules():
        if isinstance(module, MultiSubspaceSVDLayer):
            module.reset_routing_stats()
