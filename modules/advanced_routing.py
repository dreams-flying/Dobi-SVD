"""
Advanced Token Routing Strategies

Implements state-of-the-art routing methods from MoE and dynamic network research:
1. Top-K routing (Switch Transformer, GShard)
2. Expert Choice routing (Latest MoE research)
3. Capacity-constrained routing (Load balancing)
4. Sinkhorn routing (Optimal transport)
5. Gating network routing (Learnable)
6. Adaptive threshold routing (Dynamic)

References:
- Switch Transformer: https://arxiv.org/abs/2101.03961
- Expert Choice: https://arxiv.org/abs/2202.09368
- Sinkhorn MoE: https://arxiv.org/abs/2106.06525
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TopKRouter(nn.Module):
    """
    Top-K routing (Switch Transformer style).

    Each token selects top-k subspaces based on importance scores.
    More flexible than threshold-based, better load balancing.

    Args:
        n_subspaces: Number of subspaces
        top_k: Number of subspaces each token can use (default: 1)
        capacity_factor: Max tokens per subspace (factor × avg_tokens)
    """

    def __init__(self, n_subspaces, top_k=1, capacity_factor=1.25):
        super().__init__()
        self.n_subspaces = n_subspaces
        self.top_k = top_k
        self.capacity_factor = capacity_factor

    def route_hard(self, importance):
        """
        Hard routing: each token goes to top-1 subspace.

        Args:
            importance: [batch, seq_len] or [seq_len]
        Returns:
            routing: [batch, seq_len] or [seq_len] (subspace IDs)
        """
        # Handle 1D input
        squeeze_output = False
        if importance.dim() == 1:
            importance = importance.unsqueeze(0)
            squeeze_output = True

        batch_size, seq_len = importance.shape

        # Normalize importance to [0, n_subspaces)
        # Higher importance → higher subspace ID
        normalized = (importance - importance.min()) / (importance.max() - importance.min() + 1e-10)
        routing = (normalized * self.n_subspaces).long().clamp(0, self.n_subspaces - 1)

        if squeeze_output:
            routing = routing.squeeze(0)

        return routing

    def route_soft_topk(self, importance):
        """
        Soft routing: each token uses top-k subspaces.

        Args:
            importance: [batch, seq_len] or [seq_len]
        Returns:
            routing_weights: [batch, seq_len, n_subspaces] or [seq_len, n_subspaces]
            routing_indices: [batch, seq_len, top_k] or [seq_len, top_k]
        """
        # Handle 1D input
        squeeze_output = False
        if importance.dim() == 1:
            importance = importance.unsqueeze(0)
            squeeze_output = True

        batch_size, seq_len = importance.shape

        # Create scores for each subspace
        # Method: Gaussian mixture centered at each subspace
        scores = torch.zeros(batch_size, seq_len, self.n_subspaces, device=importance.device)

        # Normalize importance to [0, n_subspaces)
        normalized = (importance - importance.min()) / (importance.max() - importance.min() + 1e-10)
        importance_scaled = normalized * (self.n_subspaces - 1)

        # Compute affinity to each subspace
        for i in range(self.n_subspaces):
            # Distance from token importance to subspace center
            distance = torch.abs(importance_scaled - i)
            # Gaussian kernel (closer = higher score)
            scores[..., i] = torch.exp(-distance * 2)

        # Top-K selection
        topk_scores, topk_indices = torch.topk(scores, k=self.top_k, dim=-1)

        # Normalize top-k scores to sum to 1
        routing_weights_topk = F.softmax(topk_scores, dim=-1)

        # Convert to full routing weights
        routing_weights = torch.zeros_like(scores)
        routing_weights.scatter_(-1, topk_indices, routing_weights_topk)

        if squeeze_output:
            routing_weights = routing_weights.squeeze(0)
            topk_indices = topk_indices.squeeze(0)

        return routing_weights, topk_indices

    def route_with_capacity(self, importance):
        """
        Capacity-constrained routing (prevents overload).

        Ensures each subspace receives roughly equal number of tokens.

        Args:
            importance: [batch, seq_len]
        Returns:
            routing: [batch, seq_len]
            overflow_mask: [batch, seq_len] (True if token couldn't be routed)
        """
        batch_size, seq_len = importance.shape

        # Calculate capacity per subspace
        tokens_per_subspace = seq_len // self.n_subspaces
        capacity = int(tokens_per_subspace * self.capacity_factor)

        # Get preferred subspace for each token
        normalized = (importance - importance.min()) / (importance.max() - importance.min() + 1e-10)
        preferred_subspace = (normalized * self.n_subspaces).long().clamp(0, self.n_subspaces - 1)

        # Allocate tokens to subspaces with capacity constraint
        routing = torch.full_like(preferred_subspace, -1)
        overflow_mask = torch.zeros_like(importance, dtype=torch.bool)

        subspace_counts = torch.zeros(self.n_subspaces, device=importance.device)

        # Process each token
        for b in range(batch_size):
            for t in range(seq_len):
                subspace = preferred_subspace[b, t].item()

                if subspace_counts[subspace] < capacity:
                    routing[b, t] = subspace
                    subspace_counts[subspace] += 1
                else:
                    # Find alternative subspace with capacity
                    found = False
                    for alt_subspace in range(self.n_subspaces):
                        if subspace_counts[alt_subspace] < capacity:
                            routing[b, t] = alt_subspace
                            subspace_counts[alt_subspace] += 1
                            found = True
                            break

                    if not found:
                        overflow_mask[b, t] = True
                        routing[b, t] = 0  # Fallback to subspace 0

        return routing, overflow_mask


class ExpertChoiceRouter(nn.Module):
    """
    Expert Choice routing (latest MoE research).

    Instead of tokens choosing experts, experts choose tokens!
    Each subspace selects top-k most relevant tokens.

    Advantages:
    - Guaranteed load balancing
    - No capacity overflow
    - Better parallelization

    Reference: https://arxiv.org/abs/2202.09368
    """

    def __init__(self, n_subspaces, tokens_per_expert=None):
        super().__init__()
        self.n_subspaces = n_subspaces
        self.tokens_per_expert = tokens_per_expert

    def route(self, importance):
        """
        Expert choice routing.

        Args:
            importance: [batch, seq_len] or [seq_len]
        Returns:
            routing: [batch, seq_len] or [seq_len] (which subspace, -1 if not selected)
            weights: [batch, seq_len] or [seq_len] (selection confidence)
        """
        # Handle 1D input
        squeeze_output = False
        if importance.dim() == 1:
            importance = importance.unsqueeze(0)
            squeeze_output = True

        batch_size, seq_len = importance.shape

        if self.tokens_per_expert is None:
            self.tokens_per_expert = seq_len // self.n_subspaces

        # Normalize importance
        normalized = (importance - importance.min()) / (importance.max() - importance.min() + 1e-10)

        # Create affinity scores for each (token, subspace) pair
        # Higher importance tokens are more attractive to higher subspaces
        token_indices = torch.arange(seq_len, device=importance.device).float()

        affinity = torch.zeros(batch_size, seq_len, self.n_subspaces, device=importance.device)

        for i in range(self.n_subspaces):
            # Subspace i prefers tokens with importance around i/n_subspaces
            target_importance = i / max(1, self.n_subspaces - 1)
            distance = torch.abs(normalized - target_importance)
            affinity[..., i] = 1.0 - distance

        # Each expert (subspace) selects top-k tokens
        routing = torch.full((batch_size, seq_len), -1, dtype=torch.long, device=importance.device)
        weights = torch.zeros(batch_size, seq_len, device=importance.device)

        for i in range(self.n_subspaces):
            # Get affinity scores for this expert
            expert_affinity = affinity[..., i]  # [batch, seq_len]

            # Select top-k tokens for this expert
            topk_values, topk_indices = torch.topk(expert_affinity, k=self.tokens_per_expert, dim=-1)

            # Assign these tokens to this expert
            for b in range(batch_size):
                for idx in topk_indices[b]:
                    if routing[b, idx] == -1:  # Not yet assigned
                        routing[b, idx] = i
                        weights[b, idx] = topk_values[b, topk_indices[b] == idx].item()

        # Handle unassigned tokens (assign to nearest available subspace)
        for b in range(batch_size):
            unassigned = (routing[b] == -1).nonzero(as_tuple=True)[0]
            if len(unassigned) > 0:
                routing[b, unassigned] = 0  # Assign to first subspace
                weights[b, unassigned] = 0.1  # Low weight

        if squeeze_output:
            routing = routing.squeeze(0)
            weights = weights.squeeze(0)

        return routing, weights


class SinkhornRouter(nn.Module):
    """
    Sinkhorn routing (optimal transport).

    Finds optimal assignment that:
    - Balances load across subspaces
    - Maximizes importance-subspace affinity

    Uses Sinkhorn-Knopp algorithm for differentiable assignment.

    Reference: https://arxiv.org/abs/2106.06525
    """

    def __init__(self, n_subspaces, sinkhorn_iters=10, temperature=0.1):
        super().__init__()
        self.n_subspaces = n_subspaces
        self.sinkhorn_iters = sinkhorn_iters
        self.temperature = temperature

    def sinkhorn(self, cost_matrix, n_iters=10):
        """
        Sinkhorn-Knopp algorithm for optimal transport.

        Args:
            cost_matrix: [batch, seq_len, n_subspaces] (higher = better match)
            n_iters: Number of iterations
        Returns:
            assignment: [batch, seq_len, n_subspaces] (normalized probabilities)
        """
        # Convert to log space for numerical stability
        log_alpha = cost_matrix / self.temperature

        for _ in range(n_iters):
            # Normalize rows
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
            # Normalize columns
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)

        # Convert back to probabilities
        assignment = torch.exp(log_alpha)

        return assignment

    def route(self, importance, hard=False):
        """
        Sinkhorn routing.

        Args:
            importance: [batch, seq_len] or [seq_len]
            hard: If True, convert to hard assignment
        Returns:
            routing: [batch, seq_len] or [seq_len] (hard) or [batch, seq_len, n_subspaces] or [seq_len, n_subspaces] (soft)
        """
        # Handle 1D input
        squeeze_output = False
        if importance.dim() == 1:
            importance = importance.unsqueeze(0)
            squeeze_output = True

        batch_size, seq_len = importance.shape

        # Normalize importance
        normalized = (importance - importance.min()) / (importance.max() - importance.min() + 1e-10)

        # Create cost matrix (affinity between tokens and subspaces)
        cost_matrix = torch.zeros(batch_size, seq_len, self.n_subspaces, device=importance.device)

        for i in range(self.n_subspaces):
            # Subspace i prefers tokens with importance around i/(n-1)
            target = i / max(1, self.n_subspaces - 1)
            affinity = 1.0 - torch.abs(normalized - target)
            cost_matrix[..., i] = affinity

        # Run Sinkhorn algorithm
        soft_assignment = self.sinkhorn(cost_matrix, self.sinkhorn_iters)

        if hard:
            # Convert to hard assignment
            routing = torch.argmax(soft_assignment, dim=-1)
            if squeeze_output:
                routing = routing.squeeze(0)
            return routing
        else:
            if squeeze_output:
                soft_assignment = soft_assignment.squeeze(0)
            return soft_assignment


class GatingNetworkRouter(nn.Module):
    """
    Learnable gating network (classic MoE style).

    Uses a small neural network to predict routing probabilities.
    Fully differentiable and can be trained end-to-end.
    """

    def __init__(self, hidden_size, n_subspaces, use_noise=True):
        super().__init__()
        self.n_subspaces = n_subspaces
        self.use_noise = use_noise

        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, n_subspaces)
        )

        # Noise for exploration (during training)
        if use_noise:
            self.noise_std = nn.Parameter(torch.ones(n_subspaces))

    def route(self, x, hard=False, add_noise=True):
        """
        Gating network routing.

        Args:
            x: [batch, seq_len, hidden_size]
            hard: If True, use hard routing
            add_noise: Add noise during training
        Returns:
            routing: [batch, seq_len] (hard) or [batch, seq_len, n_subspaces] (soft)
        """
        # Compute gate logits
        gate_logits = self.gate(x)  # [batch, seq_len, n_subspaces]

        # Add noise during training
        if self.training and self.use_noise and add_noise:
            noise = torch.randn_like(gate_logits) * self.noise_std
            gate_logits = gate_logits + noise

        if hard:
            # Hard routing: argmax
            routing = torch.argmax(gate_logits, dim=-1)
            return routing
        else:
            # Soft routing: softmax
            routing_probs = F.softmax(gate_logits, dim=-1)
            return routing_probs


class AdaptiveThresholdRouter(nn.Module):
    """
    Adaptive threshold routing.

    Dynamically adjusts thresholds based on importance distribution
    to ensure balanced load across subspaces.

    Better than fixed thresholds as it adapts to data distribution.
    """

    def __init__(self, n_subspaces, momentum=0.9):
        super().__init__()
        self.n_subspaces = n_subspaces
        self.momentum = momentum

        # Running statistics
        self.register_buffer('running_min', torch.tensor(0.0))
        self.register_buffer('running_max', torch.tensor(1.0))
        self.register_buffer('running_quantiles', torch.linspace(0, 1, n_subspaces + 1))

    def update_quantiles(self, importance):
        """Update running quantiles based on current batch."""
        if self.training:
            # Compute quantiles
            sorted_importance = torch.sort(importance.flatten())[0]
            n = len(sorted_importance)

            quantile_indices = [int(i * n / self.n_subspaces) for i in range(self.n_subspaces + 1)]
            quantiles = sorted_importance[quantile_indices]

            # Update with momentum
            self.running_quantiles = (self.momentum * self.running_quantiles +
                                     (1 - self.momentum) * quantiles)
            self.running_min = self.momentum * self.running_min + (1 - self.momentum) * importance.min()
            self.running_max = self.momentum * self.running_max + (1 - self.momentum) * importance.max()

    def route(self, importance):
        """
        Adaptive threshold routing.

        Args:
            importance: [batch, seq_len] or [seq_len]
        Returns:
            routing: [batch, seq_len] or [seq_len]
        """
        # Handle 1D input
        squeeze_output = False
        if importance.dim() == 1:
            importance = importance.unsqueeze(0)
            squeeze_output = True

        # Update quantiles
        self.update_quantiles(importance)

        # Assign based on quantiles
        routing = torch.zeros_like(importance, dtype=torch.long)

        for i in range(self.n_subspaces):
            lower = self.running_quantiles[i]
            upper = self.running_quantiles[i + 1]

            mask = (importance >= lower) & (importance < upper)
            routing[mask] = i

        # Handle edge case: importance == max
        routing[importance >= self.running_quantiles[-1]] = self.n_subspaces - 1

        if squeeze_output:
            routing = routing.squeeze(0)

        return routing


# ============================================================================
# Unified Router Interface
# ============================================================================

class UnifiedRouter(nn.Module):
    """
    Unified interface for all routing strategies.

    Usage:
        router = UnifiedRouter(strategy='topk', n_subspaces=3, hidden_size=768)
        routing = router(importance, x)  # x only needed for gating
    """

    def __init__(self, strategy='topk', n_subspaces=3, hidden_size=None, device=None, **kwargs):
        super().__init__()
        self.strategy = strategy
        self.n_subspaces = n_subspaces
        self.device = device

        # Filter out device from kwargs for routers that don't need it
        # (Most routers don't explicitly accept device parameter)

        if strategy == 'topk':
            self.router = TopKRouter(n_subspaces, **kwargs)
        elif strategy == 'expert_choice':
            self.router = ExpertChoiceRouter(n_subspaces, **kwargs)
        elif strategy == 'sinkhorn':
            self.router = SinkhornRouter(n_subspaces, **kwargs)
        elif strategy == 'gating':
            assert hidden_size is not None, "hidden_size required for gating"
            self.router = GatingNetworkRouter(hidden_size, n_subspaces, **kwargs)
        elif strategy == 'adaptive':
            self.router = AdaptiveThresholdRouter(n_subspaces, **kwargs)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Move router to device if specified
        if device is not None:
            self.router = self.router.to(device)

    def forward(self, importance, x=None, hard=True):
        """
        Route tokens to subspaces.

        Args:
            importance: [batch, seq_len] - importance scores
            x: [batch, seq_len, hidden_size] - needed for gating strategy
            hard: Use hard routing (discrete assignment)
        Returns:
            routing: [batch, seq_len] (hard) or [batch, seq_len, n_subspaces] (soft)
        """
        if self.strategy == 'topk':
            if hard:
                return self.router.route_hard(importance)
            else:
                routing_weights, _ = self.router.route_soft_topk(importance)
                return routing_weights

        elif self.strategy == 'expert_choice':
            routing, weights = self.router.route(importance)
            return routing if hard else weights.unsqueeze(-1).expand(-1, -1, self.n_subspaces)

        elif self.strategy == 'sinkhorn':
            return self.router.route(importance, hard=hard)

        elif self.strategy == 'gating':
            assert x is not None, "x required for gating strategy"
            return self.router.route(x, hard=hard)

        elif self.strategy == 'adaptive':
            return self.router.route(importance)

    def route_hard(self, importance, x=None):
        """
        Hard routing: discrete assignment to subspaces.

        Args:
            importance: [batch, seq_len] - importance scores
            x: [batch, seq_len, hidden_size] - needed for gating strategy
        Returns:
            routing: [batch, seq_len] - subspace assignments (long tensor)
        """
        return self.forward(importance, x=x, hard=True)

    def route_soft(self, importance, x=None, temperature=1.0):
        """
        Soft routing: weighted combination of subspaces.

        Args:
            importance: [batch, seq_len] - importance scores
            x: [batch, seq_len, hidden_size] - needed for gating strategy
            temperature: temperature for softmax (not used by all strategies)
        Returns:
            routing: [batch, seq_len, n_subspaces] - routing weights
        """
        return self.forward(importance, x=x, hard=False)


# ============================================================================
# Comparison and Benchmarking
# ============================================================================

def compare_routing_strategies():
    """
    Compare different routing strategies.
    """
    batch_size = 2
    seq_len = 16
    hidden_size = 64
    n_subspaces = 3

    # Create dummy data
    importance = torch.randn(batch_size, seq_len).sigmoid()
    x = torch.randn(batch_size, seq_len, hidden_size)

    strategies = ['topk', 'expert_choice', 'sinkhorn', 'adaptive']

    print("="*80)
    print("Routing Strategy Comparison")
    print("="*80)
    print(f"Batch size: {batch_size}, Seq len: {seq_len}, Subspaces: {n_subspaces}\n")

    for strategy in strategies:
        router = UnifiedRouter(strategy=strategy, n_subspaces=n_subspaces, hidden_size=hidden_size)

        # Route
        routing = router(importance, x, hard=True)

        # Calculate load balance
        counts = torch.zeros(n_subspaces)
        for i in range(n_subspaces):
            counts[i] = (routing == i).sum().item()

        balance_score = counts.std().item() / (counts.mean().item() + 1e-10)

        print(f"{strategy:15s} | Distribution: {counts.tolist()} | Balance: {balance_score:.3f}")

    print("\n" + "="*80)
    print("Recommendations:")
    print("  - Best balance: 'sinkhorn' or 'adaptive'")
    print("  - Best performance: 'expert_choice'")
    print("  - Most flexible: 'gating' (learnable)")
    print("="*80)


if __name__ == "__main__":
    compare_routing_strategies()
