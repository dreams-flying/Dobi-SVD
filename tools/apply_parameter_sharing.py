#!/usr/bin/env python3
"""
Parameter Sharing Conversion Tool

Converts a multi-subspace dynamic routing model to use parameter sharing,
reducing model size while maintaining most of the quality benefits.

Strategies:
    - shared_uv: Share U and V matrices, only store different Sigma truncation
    - low_rank_adapter: Base SVD + small delta adjustments per subspace

Usage:
    python tools/apply_parameter_sharing.py \
        --model_path results/trained_model \
        --sharing_type shared_uv \
        --output results/shared_model
"""

import torch
import torch.nn as nn
import argparse
import json
import os
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.dynamic_subspace import MultiSubspaceSVDLayer
from transformers import AutoModelForCausalLM, AutoTokenizer


class SharedUVSubspaceLayer(nn.Module):
    """
    SVD layer with shared U,V matrices across subspaces.
    Only Sigma truncation parameters differ per subspace.
    """

    def __init__(self, original_layer):
        super().__init__()

        self.name = original_layer.name
        self.n_subspaces = original_layer.n_subspaces
        self.input_size = original_layer.input_size
        self.output_size = original_layer.output_size
        self.beta = original_layer.beta
        self.gammas = original_layer.gammas

        # Compute SVD once on the original weight
        with torch.no_grad():
            if hasattr(original_layer, 'weight'):
                weight = original_layer.weight.data
            else:
                # Reconstruct from first subspace as approximation
                weight = torch.randn(self.output_size, self.input_size, device=original_layer.device)

            U, S, V = torch.linalg.svd(weight, full_matrices=False)

            # Store shared U, V, S
            self.register_buffer('U', U)
            self.register_buffer('S', S)
            self.register_buffer('V', V)

        # Keep router (importance calculation and routing)
        self.router = original_layer.router

        # Bias
        if hasattr(original_layer, 'bias') and original_layer.bias is not None:
            self.register_buffer('bias', original_layer.bias)
        else:
            self.bias = None

    def forward(self, x):
        """
        Forward pass with shared U,V.

        Different subspaces use different truncation of the same SVD.
        """
        batch_size, seq_len, hidden_size = x.shape
        device = x.device

        # Reshape input
        x_flat = x.view(-1, hidden_size)  # [batch*seq, hidden]

        # Compute importance scores
        importance = self.router.compute_importance(x)  # [batch, seq]

        # Route tokens
        routing = self.router.route_tokens(importance, hard=True)  # [batch, seq]
        routing_flat = routing.view(-1)  # [batch*seq]

        # Compute soft routing weights for smooth combination
        routing_soft = self.router.route_tokens(importance, hard=False)  # [batch, seq, n_subspaces] or [batch, seq]

        # Handle different routing output formats
        if routing_soft.dim() == 2:
            # Hard routing was returned, convert to soft
            routing_weights = torch.zeros(batch_size, seq_len, self.n_subspaces, device=device)
            for k in range(self.n_subspaces):
                mask = (routing == k)
                routing_weights[mask, k] = 1.0
        else:
            routing_weights = routing_soft  # [batch, seq, n_subspaces]

        # Initialize output
        output = torch.zeros(batch_size * seq_len, self.output_size, device=device, dtype=x.dtype)

        # Process each subspace with different truncation
        for subspace_id, gamma in enumerate(self.gammas):
            # Get gamma value
            if isinstance(gamma, torch.Tensor):
                gamma_val = gamma.item()
            else:
                gamma_val = gamma

            # Compute truncation function
            sequence = torch.arange(1, len(self.S) + 1, device=device, dtype=torch.float32)
            trunc = 0.5 * torch.tanh(self.beta * (gamma_val - sequence)) + 0.5

            # Apply truncation to singular values
            S_truncated = self.S * trunc

            # SVD reconstruction: X @ V @ diag(S_truncated) @ U^T
            # For efficiency: (X @ V) @ diag(S_truncated) @ U^T
            x_v = x_flat @ self.V.T  # [batch*seq, rank]
            x_vs = x_v * S_truncated.unsqueeze(0)  # [batch*seq, rank]
            x_transformed = x_vs @ self.U.T  # [batch*seq, output_size]

            # Weight by routing probability
            weight = routing_weights[:, :, subspace_id].reshape(-1, 1)  # [batch*seq, 1]

            # Skip if weight is negligible
            if weight.abs().max() < 1e-6:
                continue

            # Accumulate weighted output
            output = output + weight * x_transformed

        # Reshape output
        output = output.view(batch_size, seq_len, self.output_size)

        # Add bias
        if self.bias is not None:
            output = output + self.bias

        return output

    def parameter_count(self):
        """Count parameters in this layer."""
        # U: output_size × rank
        # S: rank
        # V: input_size × rank
        # gammas: n_subspaces
        rank = self.S.shape[0]
        count = self.output_size * rank + rank + self.input_size * rank + self.n_subspaces

        if self.bias is not None:
            count += self.bias.numel()

        # Router parameters
        for param in self.router.parameters():
            count += param.numel()

        return count


class LowRankAdapterLayer(nn.Module):
    """
    SVD layer with base + low-rank adapters per subspace.
    More flexible than shared U,V but still parameter-efficient.
    """

    def __init__(self, original_layer, adapter_rank_ratio=0.1):
        super().__init__()

        self.name = original_layer.name
        self.n_subspaces = original_layer.n_subspaces
        self.input_size = original_layer.input_size
        self.output_size = original_layer.output_size
        self.adapter_rank_ratio = adapter_rank_ratio

        # Compute base SVD (using average gamma)
        with torch.no_grad():
            if hasattr(original_layer, 'weight'):
                weight = original_layer.weight.data
            else:
                weight = torch.randn(self.output_size, self.input_size, device=original_layer.device)

            # Base rank = average of gammas
            avg_gamma = sum([g.item() if isinstance(g, torch.Tensor) else g
                           for g in original_layer.gammas]) / len(original_layer.gammas)
            base_rank = int(avg_gamma)

            U_base, S_base, V_base = torch.svd_lowrank(weight, q=base_rank)

            self.register_buffer('U_base', U_base)
            self.register_buffer('S_base', S_base)
            self.register_buffer('V_base', V_base)

        # Low-rank adapters for each subspace
        adapter_rank = max(1, int(base_rank * adapter_rank_ratio))
        self.adapters = nn.ModuleList()

        for _ in range(self.n_subspaces):
            adapter = nn.ModuleDict({
                'U': nn.Parameter(torch.randn(self.output_size, adapter_rank) * 0.01),
                'S': nn.Parameter(torch.ones(adapter_rank)),
                'V': nn.Parameter(torch.randn(self.input_size, adapter_rank) * 0.01)
            })
            self.adapters.append(adapter)

        # Keep router
        self.router = original_layer.router

        # Bias
        if hasattr(original_layer, 'bias') and original_layer.bias is not None:
            self.register_buffer('bias', original_layer.bias)
        else:
            self.bias = None

    def forward(self, x):
        """Forward with base + adapters."""
        batch_size, seq_len, hidden_size = x.shape
        device = x.device

        x_flat = x.view(-1, hidden_size)

        # Compute importance and routing
        importance = self.router.compute_importance(x)
        routing = self.router.route_tokens(importance, hard=True)
        routing_soft = self.router.route_tokens(importance, hard=False)

        if routing_soft.dim() == 2:
            routing_weights = torch.zeros(batch_size, seq_len, self.n_subspaces, device=device)
            for k in range(self.n_subspaces):
                mask = (routing == k)
                routing_weights[mask, k] = 1.0
        else:
            routing_weights = routing_soft

        # Base transformation (shared)
        x_base = x_flat @ self.V_base * self.S_base.unsqueeze(0)
        x_base = x_base @ self.U_base.T

        # Initialize output with base
        output = x_base.view(batch_size, seq_len, self.output_size)

        # Add adapter deltas per subspace
        for subspace_id, adapter in enumerate(self.adapters):
            U_adapter = adapter['U']
            S_adapter = adapter['S']
            V_adapter = adapter['V']

            # Adapter transformation
            x_adapter = x_flat @ V_adapter * S_adapter.unsqueeze(0)
            x_adapter = x_adapter @ U_adapter.T
            x_adapter = x_adapter.view(batch_size, seq_len, self.output_size)

            # Weight by routing
            weight = routing_weights[:, :, subspace_id].unsqueeze(-1)
            output = output + weight * x_adapter

        if self.bias is not None:
            output = output + self.bias

        return output


def convert_model(model, sharing_type='shared_uv', adapter_rank_ratio=0.1):
    """
    Convert model to use parameter sharing.

    Args:
        model: Original model with MultiSubspaceSVDLayer
        sharing_type: 'shared_uv' or 'low_rank_adapter'
        adapter_rank_ratio: For low_rank_adapter, ratio of adapter rank to base rank

    Returns:
        Converted model
    """
    print(f"\nConverting model with strategy: {sharing_type}")

    converted_count = 0
    original_params = 0
    new_params = 0

    # Find and replace MultiSubspaceSVDLayer modules
    for name, module in model.named_modules():
        if isinstance(module, MultiSubspaceSVDLayer):
            print(f"\nConverting layer: {module.name}")

            # Count original parameters
            orig_param_count = sum(p.numel() for p in module.parameters())
            original_params += orig_param_count

            # Create new layer
            if sharing_type == 'shared_uv':
                new_layer = SharedUVSubspaceLayer(module)
            elif sharing_type == 'low_rank_adapter':
                new_layer = LowRankAdapterLayer(module, adapter_rank_ratio=adapter_rank_ratio)
            else:
                raise ValueError(f"Unknown sharing type: {sharing_type}")

            # Count new parameters
            new_param_count = sum(p.numel() for p in new_layer.parameters()) + \
                            sum(b.numel() for b in new_layer.buffers())
            new_params += new_param_count

            # Replace in model
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]

            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model

            setattr(parent, child_name, new_layer)

            converted_count += 1
            reduction = (1 - new_param_count / orig_param_count) * 100

            print(f"  Original params: {orig_param_count:,}")
            print(f"  New params: {new_param_count:,}")
            print(f"  Reduction: {reduction:.1f}%")

    print(f"\n✓ Converted {converted_count} layers")
    print(f"Total original parameters: {original_params:,}")
    print(f"Total new parameters: {new_params:,}")
    print(f"Overall reduction: {(1 - new_params / original_params) * 100:.1f}%")

    return model


def main():
    parser = argparse.ArgumentParser(description='Apply parameter sharing to dynamic routing model')

    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--sharing_type', type=str, default='shared_uv',
                       choices=['shared_uv', 'low_rank_adapter'],
                       help='Parameter sharing strategy')
    parser.add_argument('--adapter_rank_ratio', type=float, default=0.1,
                       help='For low_rank_adapter, adapter rank as ratio of base rank (default 0.1)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for converted model')
    parser.add_argument('--verify', action='store_true',
                       help='Verify conversion with sample forward pass')

    args = parser.parse_args()

    print("="*70)
    print("Parameter Sharing Conversion Tool")
    print("="*70)
    print(f"Model: {args.model_path}")
    print(f"Sharing type: {args.sharing_type}")
    print(f"Output: {args.output}")
    print("="*70)

    # Load model
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Convert
    model = convert_model(model, sharing_type=args.sharing_type,
                         adapter_rank_ratio=args.adapter_rank_ratio)

    # Verification
    if args.verify:
        print("\nVerifying conversion...")
        try:
            model.eval()
            with torch.no_grad():
                test_input = tokenizer("Hello world", return_tensors='pt')
                output = model(**test_input)
                print("  ✓ Forward pass successful")
                print(f"  Output shape: {output.logits.shape}")
        except Exception as e:
            print(f"  ✗ Verification failed: {e}")

    # Save
    print(f"\nSaving converted model to: {args.output}")
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    # Save conversion info
    info = {
        'sharing_type': args.sharing_type,
        'adapter_rank_ratio': args.adapter_rank_ratio if args.sharing_type == 'low_rank_adapter' else None,
        'original_model': args.model_path
    }

    with open(output_path / 'conversion_info.json', 'w') as f:
        json.dump(info, f, indent=2)

    print("\n" + "="*70)
    print("✓ Conversion complete!")
    print("="*70)
    print(f"\nNext steps:")
    print(f"1. (Recommended) Fine-tune on small dataset to recover quality")
    print(f"2. Evaluate model quality")
    print(f"3. Deploy optimized model")


if __name__ == '__main__':
    main()
