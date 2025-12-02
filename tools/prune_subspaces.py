#!/usr/bin/env python3
"""
Subspace Pruning Tool

Removes low-usage subspaces from trained dynamic routing models
to reduce parameters while maintaining quality.

Usage:
    python tools/prune_subspaces.py \
        --model_path results/trained_model \
        --threshold 0.1 \
        --output results/pruned_model
"""

import torch
import argparse
import json
import os
from pathlib import Path
import sys
from tqdm import tqdm
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.dynamic_subspace import MultiSubspaceSVDLayer
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def collect_usage_statistics(model, tokenizer, val_dataset, n_samples=500, device='cuda'):
    """
    Collect subspace usage statistics on validation data.

    Returns:
        subspace_stats: dict with usage ratios, importance scores, etc.
    """
    print("Collecting subspace usage statistics...")

    # Initialize counters
    layer_stats = {}

    model.eval()
    total_tokens = 0

    with torch.no_grad():
        for i, example in enumerate(tqdm(val_dataset.select(range(min(n_samples, len(val_dataset)))))):
            if i >= n_samples:
                break

            # Tokenize
            inputs = tokenizer(example['text'], return_tensors='pt', max_length=512, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Forward pass with hook to capture routing
            def routing_hook(module, input, output):
                if isinstance(module, MultiSubspaceSVDLayer):
                    layer_name = module.name
                    if layer_name not in layer_stats:
                        layer_stats[layer_name] = {
                            'subspace_counts': {k: 0 for k in range(module.n_subspaces)},
                            'subspace_importance': {k: [] for k in range(module.n_subspaces)},
                            'n_subspaces': module.n_subspaces,
                            'gammas': module.gammas
                        }

                    # Get routing decisions (stored during forward pass)
                    if hasattr(module, 'last_routing'):
                        routing = module.last_routing  # [batch, seq_len]
                        importance = module.last_importance  # [batch, seq_len]

                        for k in range(module.n_subspaces):
                            mask = (routing == k)
                            count = mask.sum().item()
                            layer_stats[layer_name]['subspace_counts'][k] += count

                            if count > 0:
                                avg_importance = importance[mask].mean().item()
                                layer_stats[layer_name]['subspace_importance'][k].append(avg_importance)

            # Register hooks
            hooks = []
            for module in model.modules():
                if isinstance(module, MultiSubspaceSVDLayer):
                    hooks.append(module.register_forward_hook(routing_hook))

            # Forward
            try:
                outputs = model(**inputs)
                total_tokens += inputs['input_ids'].numel()
            except Exception as e:
                print(f"Warning: Error in forward pass: {e}")
                continue

            # Remove hooks
            for hook in hooks:
                hook.remove()

    # Compute statistics
    stats = {}
    for layer_name, layer_stat in layer_stats.items():
        layer_total = sum(layer_stat['subspace_counts'].values())
        if layer_total == 0:
            continue

        stats[layer_name] = {
            'subspaces': {},
            'n_subspaces': layer_stat['n_subspaces']
        }

        for k in range(layer_stat['n_subspaces']):
            count = layer_stat['subspace_counts'][k]
            usage_ratio = count / layer_total if layer_total > 0 else 0.0

            importance_scores = layer_stat['subspace_importance'][k]
            avg_importance = np.mean(importance_scores) if importance_scores else 0.0

            stats[layer_name]['subspaces'][k] = {
                'usage_ratio': usage_ratio,
                'token_count': count,
                'avg_importance': avg_importance,
                'gamma': layer_stat['gammas'][k] if isinstance(layer_stat['gammas'], list) else layer_stat['gammas'][k].item()
            }

    return stats


def decide_pruning(stats, threshold=0.1, strategy='usage_based'):
    """
    Decide which subspaces to prune based on statistics.

    Args:
        stats: Usage statistics from collect_usage_statistics
        threshold: Pruning threshold (default 0.1 = prune if usage < 10%)
        strategy: 'usage_based' or 'importance_based'

    Returns:
        pruning_plan: dict mapping layer_name to list of subspaces to keep
    """
    pruning_plan = {}

    for layer_name, layer_stat in stats.items():
        subspaces_to_keep = []

        for k, subspace_stat in layer_stat['subspaces'].items():
            if strategy == 'usage_based':
                metric = subspace_stat['usage_ratio']
            elif strategy == 'importance_based':
                # Importance = usage * avg_importance
                metric = subspace_stat['usage_ratio'] * subspace_stat['avg_importance']
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            if metric >= threshold:
                subspaces_to_keep.append(k)

        # Always keep at least 1 subspace
        if len(subspaces_to_keep) == 0:
            # Keep the one with highest usage
            best_k = max(layer_stat['subspaces'].keys(),
                        key=lambda k: layer_stat['subspaces'][k]['usage_ratio'])
            subspaces_to_keep = [best_k]

        pruning_plan[layer_name] = sorted(subspaces_to_keep)

    return pruning_plan


def prune_model(model, pruning_plan):
    """
    Apply pruning plan to model.

    Removes unused subspaces from each layer.
    """
    print("\nApplying pruning plan...")

    total_removed = 0
    total_kept = 0

    for module in model.modules():
        if isinstance(module, MultiSubspaceSVDLayer):
            layer_name = module.name

            if layer_name not in pruning_plan:
                print(f"Warning: {layer_name} not in pruning plan, skipping")
                continue

            subspaces_to_keep = pruning_plan[layer_name]
            original_n = module.n_subspaces

            # Update module
            module.n_subspaces = len(subspaces_to_keep)
            module.gammas = [module.gammas[k] for k in subspaces_to_keep]

            # Update router thresholds if using threshold-based routing
            if hasattr(module.router, 'thresholds') and module.router.thresholds is not None:
                # Recompute thresholds for new number of subspaces
                n_new = len(subspaces_to_keep)
                new_thresholds = torch.linspace(0, 1, n_new + 1)[1:-1]
                module.router.thresholds = torch.nn.Parameter(new_thresholds)

            # Update advanced router if present
            if hasattr(module.router, 'advanced_router') and module.router.advanced_router is not None:
                module.router.advanced_router.n_subspaces = len(subspaces_to_keep)
                if hasattr(module.router.advanced_router.router, 'n_subspaces'):
                    module.router.advanced_router.router.n_subspaces = len(subspaces_to_keep)

            removed = original_n - len(subspaces_to_keep)
            total_removed += removed
            total_kept += len(subspaces_to_keep)

            print(f"  {layer_name}: {original_n} → {len(subspaces_to_keep)} subspaces "
                  f"(removed: {removed}, kept: {subspaces_to_keep})")

    print(f"\nTotal: Removed {total_removed} subspaces, Kept {total_kept} subspaces")

    return model


def main():
    parser = argparse.ArgumentParser(description='Prune low-usage subspaces from dynamic routing model')

    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--threshold', type=float, default=0.1,
                       help='Pruning threshold (usage ratio, default 0.1 = 10%%)')
    parser.add_argument('--strategy', type=str, default='usage_based',
                       choices=['usage_based', 'importance_based'],
                       help='Pruning strategy')
    parser.add_argument('--val_data', type=str, default=None,
                       help='Validation dataset (default: wikitext-2-raw-v1)')
    parser.add_argument('--n_samples', type=int, default=500,
                       help='Number of validation samples to use')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for pruned model')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (default: cuda)')

    args = parser.parse_args()

    print("="*70)
    print("Subspace Pruning Tool")
    print("="*70)
    print(f"Model: {args.model_path}")
    print(f"Threshold: {args.threshold}")
    print(f"Strategy: {args.strategy}")
    print(f"Output: {args.output}")
    print("="*70)

    # Load model
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = model.to(args.device)

    # Load validation data
    print("\nLoading validation data...")
    if args.val_data is None:
        val_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation')
    else:
        val_dataset = load_dataset('json', data_files=args.val_data, split='train')

    # Collect statistics
    stats = collect_usage_statistics(model, tokenizer, val_dataset,
                                     n_samples=args.n_samples, device=args.device)

    # Save statistics
    stats_path = Path(args.output) / 'pruning_statistics.json'
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to serializable format
    stats_serializable = {}
    for layer_name, layer_stat in stats.items():
        stats_serializable[layer_name] = {
            'n_subspaces': layer_stat['n_subspaces'],
            'subspaces': {str(k): v for k, v in layer_stat['subspaces'].items()}
        }

    with open(stats_path, 'w') as f:
        json.dump(stats_serializable, f, indent=2)

    print(f"\nStatistics saved to: {stats_path}")

    # Print statistics
    print("\n" + "="*70)
    print("Subspace Usage Statistics")
    print("="*70)
    for layer_name, layer_stat in stats.items():
        print(f"\n{layer_name}:")
        for k, subspace_stat in layer_stat['subspaces'].items():
            usage_pct = subspace_stat['usage_ratio'] * 100
            status = "KEEP" if subspace_stat['usage_ratio'] >= args.threshold else "PRUNE"
            print(f"  Subspace {k} (γ={subspace_stat['gamma']:3.0f}): "
                  f"{usage_pct:5.1f}% usage, importance={subspace_stat['avg_importance']:.3f} [{status}]")

    # Decide pruning
    pruning_plan = decide_pruning(stats, threshold=args.threshold, strategy=args.strategy)

    # Save pruning plan
    plan_path = Path(args.output) / 'pruning_plan.json'
    with open(plan_path, 'w') as f:
        json.dump(pruning_plan, f, indent=2)

    print(f"\nPruning plan saved to: {plan_path}")

    # Apply pruning
    model = prune_model(model, pruning_plan)

    # Save pruned model
    print(f"\nSaving pruned model to: {args.output}")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    print("\n" + "="*70)
    print("✓ Pruning complete!")
    print("="*70)
    print(f"\nNext steps:")
    print(f"1. Evaluate pruned model quality")
    print(f"2. (Optional) Fine-tune on small dataset to recover quality")
    print(f"3. Deploy optimized model")


if __name__ == '__main__':
    main()
