"""
Routing Analysis and Visualization Script

This script analyzes the routing behavior of MultiSubspaceSVDLayer models,
providing insights into:
1. Token distribution across subspaces
2. Per-layer routing patterns
3. Importance score distributions
4. FLOPs savings analysis
"""

import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn

from modules.dynamic_subspace import (
    MultiSubspaceSVDLayer,
    get_model_routing_statistics,
    reset_model_routing_statistics
)
from utils.datautils import prepare_train_loaders
from evaluate import evaluate_perplexity


def load_trained_model(model_path, config_path):
    """Load a trained model with MultiSubspaceSVDLayer."""
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"Loading model from: {model_path}")
    print(f"Config: {json.dumps(config, indent=2)}")

    # TODO: Implement model loading
    # This would require modifying weight_updater.py to support MultiSubspaceSVDLayer
    raise NotImplementedError("Model loading not yet implemented")


def analyze_routing_distribution(model, dataloader, n_samples=100):
    """
    Analyze routing distribution across the model.

    Returns:
        dict: Routing statistics per layer
    """
    model.eval()
    reset_model_routing_statistics(model)

    device = next(model.parameters()).device

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Analyzing routing")):
            if i >= n_samples:
                break

            input_ids = batch['input_ids'].to(device)
            model(input_ids=input_ids)

    # Collect statistics
    stats = get_model_routing_statistics(model)

    return stats


def plot_routing_distribution(stats, save_path):
    """
    Plot routing distribution for all layers.

    Args:
        stats: Dictionary of routing statistics per layer
        save_path: Path to save the plot
    """
    layer_names = list(stats.keys())
    n_layers = len(layer_names)
    n_subspaces = len(stats[layer_names[0]]['routing_distribution'])

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Stacked bar chart of routing distribution
    distributions = np.array([stats[name]['routing_distribution'] for name in layer_names])

    x = np.arange(n_layers)
    bottom = np.zeros(n_layers)

    colors = plt.cm.viridis(np.linspace(0, 1, n_subspaces))

    for i in range(n_subspaces):
        axes[0].bar(x, distributions[:, i], bottom=bottom, label=f'Subspace {i}', color=colors[i])
        bottom += distributions[:, i]

    axes[0].set_xlabel('Layer Index')
    axes[0].set_ylabel('Token Proportion')
    axes[0].set_title('Routing Distribution Across Layers')
    axes[0].legend()
    axes[0].set_xticks(x[::max(1, n_layers // 10)])
    axes[0].set_xticklabels(x[::max(1, n_layers // 10)])

    # Plot 2: Gamma values per layer
    gammas_array = np.array([stats[name]['gammas'] for name in layer_names])

    for i in range(n_subspaces):
        axes[1].plot(gammas_array[:, i], marker='o', label=f'Gamma {i}', color=colors[i])

    axes[1].set_xlabel('Layer Index')
    axes[1].set_ylabel('Gamma Value (Rank)')
    axes[1].set_title('Gamma Values Across Layers')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Routing distribution plot saved to: {save_path}")
    plt.close()


def plot_importance_distribution(stats, save_path):
    """
    Plot importance score distribution.

    Args:
        stats: Dictionary of routing statistics per layer
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    layer_names = list(stats.keys())
    n_plots = min(6, len(layer_names))

    for i in range(n_plots):
        layer_name = layer_names[i * len(layer_names) // n_plots]
        importance = stats[layer_name]['cached_importance']

        if importance is not None:
            axes[i].hist(importance.flatten(), bins=50, alpha=0.7, edgecolor='black')
            axes[i].set_xlabel('Importance Score')
            axes[i].set_ylabel('Frequency')
            axes[i].set_title(f'Layer {i * len(layer_names) // n_plots}: {layer_name.split(".")[-1]}')
            axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Importance distribution plot saved to: {save_path}")
    plt.close()


def calculate_flops_savings(stats, seq_len=2048):
    """
    Calculate FLOPs savings from dynamic routing.

    Args:
        stats: Routing statistics
        seq_len: Sequence length

    Returns:
        dict: FLOPs analysis
    """
    total_flops_baseline = 0
    total_flops_dynamic = 0

    for layer_name, layer_stats in stats.items():
        gammas = layer_stats['gammas']
        routing_dist = layer_stats['routing_distribution']

        # Baseline: use highest gamma for all tokens
        gamma_baseline = max(gammas)

        # Dynamic: weighted average based on routing
        gamma_dynamic = sum(g * d for g, d in zip(gammas, routing_dist))

        # FLOPs for SVD computation (approximate)
        # U, S, V = svd(x, rank) costs roughly O(n * rank^2) where n is hidden_size
        # Here we simplify to rank * seq_len
        flops_baseline = gamma_baseline * seq_len
        flops_dynamic = gamma_dynamic * seq_len

        total_flops_baseline += flops_baseline
        total_flops_dynamic += flops_dynamic

    savings_ratio = (total_flops_baseline - total_flops_dynamic) / total_flops_baseline
    speedup = total_flops_baseline / total_flops_dynamic

    return {
        'total_flops_baseline': total_flops_baseline,
        'total_flops_dynamic': total_flops_dynamic,
        'savings_ratio': savings_ratio,
        'speedup': speedup
    }


def print_summary_statistics(stats, flops_analysis):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("ROUTING ANALYSIS SUMMARY")
    print("="*80)

    # Overall routing distribution
    n_layers = len(stats)
    n_subspaces = len(list(stats.values())[0]['routing_distribution'])

    avg_distribution = np.mean([s['routing_distribution'] for s in stats.values()], axis=0)

    print(f"\nTotal Layers Analyzed: {n_layers}")
    print(f"Number of Subspaces: {n_subspaces}")
    print(f"\nAverage Routing Distribution:")
    for i, dist in enumerate(avg_distribution):
        print(f"  Subspace {i}: {dist:.2%}")

    # Gamma statistics
    all_gammas = np.array([s['gammas'] for s in stats.values()])
    print(f"\nGamma Statistics:")
    for i in range(n_subspaces):
        print(f"  Subspace {i}: mean={all_gammas[:, i].mean():.2f}, "
              f"std={all_gammas[:, i].std():.2f}, "
              f"min={all_gammas[:, i].min():.2f}, "
              f"max={all_gammas[:, i].max():.2f}")

    # FLOPs analysis
    print(f"\nFLOPs Analysis:")
    print(f"  Baseline FLOPs: {flops_analysis['total_flops_baseline']:.2e}")
    print(f"  Dynamic FLOPs: {flops_analysis['total_flops_dynamic']:.2e}")
    print(f"  Savings: {flops_analysis['savings_ratio']:.2%}")
    print(f"  Speedup: {flops_analysis['speedup']:.2f}x")

    print("="*80 + "\n")


def main(args):
    """Main analysis function."""

    # Load model and config
    if args.trained_model_path:
        model = load_trained_model(args.trained_model_path, args.config_path)
    else:
        print("Error: Please provide --trained_model_path")
        return

    # Load data
    model_id = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare dataset
    from utils.datautils import prepare_train_loaders
    DATASET_NAME = args.dataset
    data_cache_dir = Path(args.data_cache_dir)
    dataset_cache_dir = Path(args.dataset_cache_dir)

    _, tokenized_testdata = prepare_train_loaders(
        tokenizer, DATASET_NAME, data_cache_dir, dataset_cache_dir, args
    )

    # Create dataloader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(tokenized_testdata, batch_size=1, shuffle=False)

    # Analyze routing
    print("Analyzing routing behavior...")
    stats = analyze_routing_distribution(model, dataloader, n_samples=args.n_samples)

    # Calculate FLOPs savings
    flops_analysis = calculate_flops_savings(stats, seq_len=args.seq_len)

    # Print summary
    print_summary_statistics(stats, flops_analysis)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save statistics
    stats_output = {
        'layer_stats': {
            k: {
                'routing_distribution': v['routing_distribution'].tolist() if hasattr(v['routing_distribution'], 'tolist') else v['routing_distribution'],
                'gammas': v['gammas']
            }
            for k, v in stats.items()
        },
        'flops_analysis': flops_analysis
    }

    with open(output_dir / 'routing_analysis.json', 'w') as f:
        json.dump(stats_output, f, indent=4)

    print(f"Statistics saved to: {output_dir / 'routing_analysis.json'}")

    # Plot visualizations
    plot_routing_distribution(stats, output_dir / 'routing_distribution.png')
    plot_importance_distribution(stats, output_dir / 'importance_distribution.png')

    print(f"\nAnalysis complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze routing behavior of MultiSubspaceSVDLayer models")

    parser.add_argument('--trained_model_path', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--config_path', type=str, required=True,
                        help='Path to config.json')
    parser.add_argument('--model_id', type=str, default='facebook/opt-125m',
                        help='Original model ID for tokenizer')
    parser.add_argument('--dataset', type=str, default='wikitext',
                        help='Dataset for analysis')
    parser.add_argument('--data_cache_dir', type=str, default='./results/data_cache/wikitext')
    parser.add_argument('--dataset_cache_dir', type=str, default='./results/datasets/wikitext')
    parser.add_argument('--n_samples', type=int, default=100,
                        help='Number of samples to analyze')
    parser.add_argument('--seq_len', type=int, default=2048,
                        help='Sequence length')
    parser.add_argument('--output_dir', type=str, default='./analysis_results',
                        help='Output directory for analysis results')

    args = parser.parse_args()
    main(args)
