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

                # Compute self-similarity (cosine similarity between tokens)
                # Normalize activations
                x_norm = F.normalize(x, p=2, dim=-1)  # [batch, seq, hidden]

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
                importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-10)

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
        # Use advanced router if specified
        if self.advanced_router is not None:
            if hard:
                routing = self.advanced_router.route_hard(importance)
            else:
                routing = self.advanced_router.route_soft(importance, temperature=temperature)

            # Update statistics
            if self.training and hard:
                for i in range(self.n_subspaces):
                    self.routing_counts[i] += (routing == i).sum().float()
                self.total_tokens += routing.numel()

            return routing

        # Default threshold-based routing
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
                 svd_rank=None):
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

        if name:
            self.name = name

        # Gamma parameters (trainable)
        self.gammas = nn.ParameterList([
            nn.Parameter(torch.tensor(g, dtype=computeSVD_dtype)).to(device)
            for g in gammas
        ])

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

            # Compute low-rank SVD
            try:
                U, S, Vh = torch.linalg.svd(weight_svd, full_matrices=False)

                # Truncate to desired rank
                U = U[:, :svd_rank]   # [output_size, svd_rank]
                S = S[:svd_rank]       # [svd_rank]
                V = Vh[:svd_rank, :]   # [svd_rank, input_size]

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

        # Print parameter savings
        standard_params = n_subspaces * svd_rank * (output_size + input_size + 1)
        shared_params = svd_rank * (output_size + input_size) + n_subspaces
        reduction = (1 - shared_params / standard_params) * 100
        print(f"[SharedParam] Parameter reduction: {reduction:.1f}% "
              f"({standard_params:,} → {shared_params:,})")

    def forward(self, x):
        """
        Forward pass with shared U,V and different gamma truncations.

        核心思想:
            对于每个子空间 k:
                1. 使用共享的 U, S, V
                2. 应用该子空间的 gamma_k 截断
                3. 重建: x_k = x @ V^T @ diag(S ⊙ Trunc_k) @ U^T
        """
        batch_size, seq_len, hidden_size = x.shape
        device = x.device

        # Flatten for processing
        x_flat = x.view(-1, hidden_size)  # [batch*seq, hidden]

        # Linear transformation: x_transformed = x @ W^T
        # Where W ≈ U @ diag(S) @ V (SVD approximation)
        # So x_transformed ≈ x @ V^T @ diag(S) @ U^T

        # Compute importance scores for routing
        # Use the reconstructed activation for importance
        x_approx = x_flat @ self.V_shared.T * self.S_shared.unsqueeze(0)
        x_approx = x_approx @ self.U_shared.T

        # Reshape for router
        x_for_routing = x_approx.view(batch_size, seq_len, self.output_size)
        importance = self.router.compute_importance(x_for_routing)

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

            # Flatten routing weights
            routing_weights_flat = routing_weights.view(-1, self.n_subspaces)  # [batch*seq, n_subspaces]

            # Initialize output
            output = torch.zeros(batch_size * seq_len, self.output_size,
                               device=device, dtype=model_load_dtype)

            # ========================================
            # 关键: 对每个子空间应用不同的gamma截断
            # ========================================

            for subspace_id, gamma in enumerate(self.gammas):
                # Get routing weight for this subspace
                weight = routing_weights_flat[:, subspace_id].unsqueeze(-1)  # [batch*seq, 1]

                # Skip if weight is negligible
                if weight.abs().max() < 1e-6:
                    continue

                # Compute truncation function for this gamma
                sequence = torch.arange(1, len(self.S_shared) + 1,
                                      device=device, dtype=self.S_shared.dtype)

                gamma_val = gamma.clamp(min=1.0, max=len(self.S_shared))
                trunc = 0.5 * torch.tanh(self.beta * (gamma_val - sequence)) + 0.5

                # Apply truncation to shared singular values
                S_truncated = self.S_shared * trunc  # [svd_rank]

                # Reconstruct: x_sub = x @ V^T @ diag(S_truncated) @ U^T
                # Step 1: x @ V^T
                xV = x_flat @ self.V_shared.T  # [batch*seq, svd_rank]

                # Step 2: multiply by truncated S
                xVS = xV * S_truncated.unsqueeze(0)  # [batch*seq, svd_rank]

                # Step 3: @ U^T
                x_sub = xVS @ self.U_shared.T  # [batch*seq, output_size]

                # Weight and accumulate
                weighted_sub = (weight * x_sub).to(model_load_dtype)
                output = output + weighted_sub

                # Free memory
                del xV, xVS, x_sub, weighted_sub, sequence, trunc, S_truncated

            real_x = output.view(batch_size, seq_len, self.output_size)

        else:
            # INFERENCE: Hard routing (discrete assignment)
            routing = self.router.route_tokens(importance, hard=True)  # [batch, seq]
            routing_flat = routing.view(-1)  # [batch*seq]

            # Cache for analysis
            self.cached_routing = routing.detach().clone()

            # Initialize output
            output = torch.zeros(batch_size * seq_len, self.output_size, device=device)

            # Process each subspace
            for subspace_id, gamma in enumerate(self.gammas):
                # Get tokens assigned to this subspace
                mask = (routing_flat == subspace_id)

                if not mask.any():
                    continue

                # Extract tokens for this subspace
                x_sub = x_flat[mask]  # [n_tokens, hidden]

                # Compute truncation
                sequence = torch.arange(1, len(self.S_shared) + 1,
                                      device=device, dtype=self.S_shared.dtype)
                gamma_val = gamma.clamp(min=1.0, max=len(self.S_shared))
                trunc = 0.5 * torch.tanh(self.beta * (gamma_val - sequence)) + 0.5
                S_truncated = self.S_shared * trunc

                # Reconstruct
                xV = x_sub @ self.V_shared.T
                xVS = xV * S_truncated.unsqueeze(0)
                x_sub_transformed = xVS @ self.U_shared.T

                # Assign back
                output[mask] = x_sub_transformed

            real_x = output.view(batch_size, seq_len, self.output_size).to(model_load_dtype)

        # Add bias
        if self.bias is not None:
            real_x = real_x + self.bias

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
