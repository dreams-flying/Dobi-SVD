#!/usr/bin/env python3
"""
Evaluation script for Matryoshka SVD models trained from SVD-LLM.

This script evaluates Matryoshka SVD models on:
1. Perplexity (wikitext2, c4, ptb)
2. Multi-rank evaluation (adaptive, r_min, r_mid, r_max)
3. Compression statistics

Usage:
    # Evaluate with adaptive rank
    CUDA_VISIBLE_DEVICES=0 python evaluate_matryoshka_svdllm.py \
        --checkpoint ./matryoshka_output/final \
        --eval_dataset wikitext2 \
        --eval_rank adaptive

    # Evaluate at fixed rank
    CUDA_VISIBLE_DEVICES=0 python evaluate_matryoshka_svdllm.py \
        --checkpoint ./matryoshka_output/final \
        --eval_dataset wikitext2 \
        --eval_rank 128

    # Multi-rank evaluation
    CUDA_VISIBLE_DEVICES=0 python evaluate_matryoshka_svdllm.py \
        --checkpoint ./matryoshka_output/final \
        --eval_dataset wikitext2 \
        --multi_rank_eval

Author: Claude
Date: 2025-12-16
"""

import argparse
import os
import sys
from pathlib import Path
import json
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM

# Add project to path
sys.path.append(str(Path(__file__).parent))

from modules.matryoshka_svd_layer import MatryoshkaSVDLayer
from utils.datautils import prepare_train_loaders


def load_matryoshka_model(
    checkpoint_path: str,
    base_model: Optional[str] = None,
    device: str = 'cuda'
) -> Tuple[nn.Module, AutoTokenizer, Optional[Dict]]:
    """
    Load Matryoshka SVD model from checkpoint.

    This function uses the custom matryoshka_model_utils.load_matryoshka_model()
    which properly reconstructs MatryoshkaSVDLayer instances from saved metadata.

    Args:
        checkpoint_path: Path to checkpoint directory
        base_model: Base model name for tokenizer (if not in checkpoint)
        device: Device to load model on

    Returns:
        model: Loaded model with MatryoshkaSVDLayer reconstructed
        tokenizer: Tokenizer
        config: Model configuration
    """
    from matryoshka_model_utils import load_matryoshka_model as load_matryoshka_custom

    # Use custom load function that reconstructs MatryoshkaSVDLayer
    model, tokenizer = load_matryoshka_custom(
        checkpoint_path=checkpoint_path,
        base_model=base_model,
        device=device,
        torch_dtype=torch.float32
    )

    # Extract Matryoshka layer info for config
    matryoshka_layers = []
    for name, module in model.named_modules():
        if isinstance(module, MatryoshkaSVDLayer):
            matryoshka_layers.append({
                'name': name,
                'r_min': module.r_min,
                'r_max': module.r_max,
                'predictor_mode': module.predictor_mode,
                'hard_inference': module.hard_inference
            })

    config = {
        'num_matryoshka_layers': len(matryoshka_layers),
        'matryoshka_layers': matryoshka_layers
    }

    return model, tokenizer, config


def set_model_rank(model: nn.Module, rank: Optional[int]):
    """
    Set fixed rank for all MatryoshkaSVDLayers in model.

    Args:
        model: Model with MatryoshkaSVDLayer instances
        rank: Fixed rank to use (None for adaptive/dynamic)
    """
    count = 0
    for module in model.modules():
        if isinstance(module, MatryoshkaSVDLayer):
            module.set_fixed_rank(rank)
            count += 1

    if rank is None:
        print(f"Set {count} layers to ADAPTIVE rank prediction")
    else:
        print(f"Set {count} layers to FIXED rank: {rank}")


def get_average_rank(model: nn.Module) -> float:
    """
    Get average effective rank across all MatryoshkaSVDLayers.

    Args:
        model: Model with MatryoshkaSVDLayer instances

    Returns:
        avg_rank: Average effective rank (or None if no layers)
    """
    ranks = []
    for module in model.modules():
        if isinstance(module, MatryoshkaSVDLayer):
            rank = module.get_effective_rank()
            if rank is not None:
                ranks.append(rank)

    return np.mean(ranks) if ranks else None


def evaluate_perplexity(
    model: nn.Module,
    dataset: torch.Tensor,
    eval_rank: str = 'adaptive',
    n_samples: int = 256,
    device: str = 'cuda'
) -> Tuple[float, Optional[float]]:
    """
    Evaluate perplexity on a dataset.

    Args:
        model: Model to evaluate
        dataset: Input IDs tensor of shape [n_samples, seq_len]
        eval_rank: 'adaptive' or integer for fixed rank
        n_samples: Number of samples to evaluate
        device: Device to use

    Returns:
        ppl: Perplexity score
        avg_rank: Average rank used (only for adaptive mode)
    """
    # Set evaluation rank
    if eval_rank == 'adaptive':
        set_model_rank(model, None)  # Enable dynamic prediction
        rank_str = "adaptive"
    else:
        fixed_rank = int(eval_rank)
        set_model_rank(model, fixed_rank)
        rank_str = str(fixed_rank)

    nsamples = min(dataset.size(0), n_samples)
    seqlen = dataset.size(1)

    print(f"\nEvaluating perplexity (rank={rank_str})...")
    print(f"  Samples: {nsamples}")
    print(f"  Sequence length: {seqlen}")

    nlls = []
    ranks_collected = []

    model.eval()

    with torch.no_grad():
        for i in tqdm(range(nsamples), desc=f"PPL (rank={rank_str})"):
            # Prepare input
            input_ids = dataset[i:i+1, :].to(device)

            # Forward pass
            try:
                outputs = model(input_ids=input_ids, use_cache=False)
                logits = outputs.logits

                # Compute loss for this sample
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()

                loss_fct = nn.CrossEntropyLoss(reduction='none')
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )

                # Average over sequence
                nll = loss.mean()

                if torch.isfinite(nll):
                    nlls.append(nll)

                    # Track average rank for adaptive mode
                    if eval_rank == 'adaptive':
                        avg_rank = get_average_rank(model)
                        if avg_rank is not None:
                            ranks_collected.append(avg_rank)

            except Exception as e:
                print(f"\nError processing sample {i}: {e}")
                continue

            # Clear cache periodically
            if (i + 1) % 10 == 0:
                torch.cuda.empty_cache()

    # Compute perplexity
    if nlls:
        ppl = torch.exp(torch.stack(nlls).mean())
        avg_rank = np.mean(ranks_collected) if ranks_collected else None
    else:
        print("⚠️  No valid samples, returning infinite perplexity")
        ppl = torch.tensor(float('inf'))
        avg_rank = None

    torch.cuda.empty_cache()

    return ppl.item(), avg_rank


def evaluate_multi_rank(
    model: nn.Module,
    dataset: torch.Tensor,
    n_samples: int,
    device: str = 'cuda'
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate perplexity at multiple rank levels.

    Args:
        model: Model to evaluate
        dataset: Dataset
        n_samples: Max samples
        device: Device

    Returns:
        results: Dictionary of {rank: {'perplexity': ppl, 'avg_rank': rank}}
    """
    # Determine rank range from first MatryoshkaSVDLayer
    r_min, r_max = None, None
    for module in model.modules():
        if isinstance(module, MatryoshkaSVDLayer):
            r_min, r_max = module.r_min, module.r_max
            break

    if r_min is None:
        print("⚠️  No MatryoshkaSVDLayer found, cannot do multi-rank eval")
        return {}

    # Define ranks to test
    r_mid = (r_min + r_max) // 2
    ranks_to_test = [
        ('adaptive', 'adaptive'),
        (f'r_max={r_max}', r_max),
        (f'r_mid={r_mid}', r_mid),
        (f'r_min={r_min}', r_min)
    ]

    results = {}

    for rank_name, rank_value in ranks_to_test:
        print(f"\n{'='*80}")
        print(f"Evaluating: {rank_name}")
        print(f"{'='*80}")

        ppl, avg_rank = evaluate_perplexity(
            model, dataset, eval_rank=rank_value,
            n_samples=n_samples, device=device
        )

        results[rank_name] = {
            'perplexity': ppl,
            'avg_rank': avg_rank,
            'target_rank': rank_value
        }

        print(f"\nResults:")
        print(f"  Perplexity: {ppl:.4f}")
        if avg_rank is not None:
            print(f"  Average rank: {avg_rank:.2f}")

    return results


def print_compression_stats(model: nn.Module):
    """Print compression statistics for all Matryoshka SVD layers."""
    print(f"\n{'='*80}")
    print(f"Compression Statistics")
    print(f"{'='*80}")

    total_original = 0
    total_compressed = 0
    layer_count = 0

    layer_stats = []

    for name, module in model.named_modules():
        if isinstance(module, MatryoshkaSVDLayer):
            # Original parameters
            original_params = module.out_features * module.in_features

            # Effective rank
            avg_rank = module.get_effective_rank()
            if avg_rank is None or avg_rank == 0:
                avg_rank = (module.r_min + module.r_max) / 2

            # Compressed parameters: V_proj + U_proj
            # V: [in_features, r] + U: [r, out_features]
            compressed_params = (module.in_features * avg_rank +
                                avg_rank * module.out_features)

            compression_ratio = compressed_params / original_params

            total_original += original_params
            total_compressed += compressed_params
            layer_count += 1

            layer_stats.append({
                'name': name,
                'original': original_params,
                'compressed': compressed_params,
                'ratio': compression_ratio,
                'rank': avg_rank,
                'r_max': module.r_max
            })

    # Print layer-wise stats (sample first few)
    print(f"\nSample layers (first 3):")
    for stat in layer_stats[:3]:
        print(f"\n{stat['name']}:")
        print(f"  Original params:     {stat['original']:>12,}")
        print(f"  Compressed params:   {stat['compressed']:>12,.0f}")
        print(f"  Compression ratio:   {stat['ratio']:>12.2%}")
        print(f"  Effective rank:      {stat['rank']:>12.1f} / {stat['r_max']}")

    # Print overall stats
    if layer_count > 0:
        overall_ratio = total_compressed / total_original
        print(f"\n{'='*80}")
        print(f"Overall Compression")
        print(f"{'='*80}")
        print(f"  Total layers:        {layer_count:>12}")
        print(f"  Total original:      {total_original:>12,}")
        print(f"  Total compressed:    {total_compressed:>12,.0f}")
        print(f"  Overall ratio:       {overall_ratio:>12.2%}")
        print(f"  Parameter reduction: {(1 - overall_ratio):>12.2%}")
        print(f"  Memory saved:        {(total_original - total_compressed)/1e6:>12.1f}M params")


def main(args):
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model, tokenizer, config = load_matryoshka_model(
        args.checkpoint,
        base_model=args.base_model,
        device=device
    )

    # Print compression stats
    print_compression_stats(model)

    # Prepare dataset
    print(f"\n{'='*80}")
    print(f"Preparing Dataset: {args.eval_dataset}")
    print(f"{'='*80}")

    path_head_folder = Path(args.path_head_folder)
    data_cache_dir = path_head_folder / 'data_cache'
    dataset_cache_dir = path_head_folder / 'dataset_cache'
    data_cache_dir.mkdir(parents=True, exist_ok=True)
    dataset_cache_dir.mkdir(parents=True, exist_ok=True)

    tokenized_traindata, tokenized_valdata = prepare_train_loaders(
        tokenizer=tokenizer,
        DATASET_NAME=args.eval_dataset,
        data_cache_dir=data_cache_dir,
        dataset_cache_dir=dataset_cache_dir,
        args=args
    )

    # Convert to tensor
    input_ids = torch.cat([
        sample["input_ids"].unsqueeze(0)
        for sample in tokenized_valdata
    ], dim=0)

    print(f"Dataset loaded: {input_ids.shape}")

    # Evaluate
    if args.multi_rank_eval:
        # Multi-rank evaluation
        results = evaluate_multi_rank(
            model, input_ids, args.n_eval_samples, device
        )

        # Print summary
        print(f"\n{'='*80}")
        print(f"Multi-Rank Evaluation Summary")
        print(f"{'='*80}")
        print(f"Dataset: {args.eval_dataset}\n")

        for rank_name, metrics in results.items():
            print(f"{rank_name}:")
            print(f"  Perplexity:  {metrics['perplexity']:>8.4f}")
            if metrics['avg_rank'] is not None:
                print(f"  Avg rank:    {metrics['avg_rank']:>8.2f}")
            print()

        # Save results
        if args.save_results:
            output_file = Path(args.checkpoint).parent / f"eval_{args.eval_dataset}_multi_rank.json"
            with open(output_file, 'w') as f:
                # Convert to serializable format
                results_serializable = {
                    k: {
                        'perplexity': float(v['perplexity']),
                        'avg_rank': float(v['avg_rank']) if v['avg_rank'] is not None else None,
                        'target_rank': v['target_rank'] if isinstance(v['target_rank'], str) else int(v['target_rank'])
                    }
                    for k, v in results.items()
                }
                json.dump(results_serializable, f, indent=2)
            print(f"Results saved to: {output_file}")

    else:
        # Single rank evaluation
        ppl, avg_rank = evaluate_perplexity(
            model, input_ids,
            eval_rank=args.eval_rank,
            n_samples=args.n_eval_samples,
            device=device
        )

        print(f"\n{'='*80}")
        print(f"Evaluation Results")
        print(f"{'='*80}")
        print(f"Dataset:     {args.eval_dataset}")
        print(f"Rank:        {args.eval_rank}")
        print(f"Perplexity:  {ppl:.4f}")
        if avg_rank is not None:
            print(f"Avg rank:    {avg_rank:.2f}")

        # Save results
        if args.save_results:
            output_file = Path(args.checkpoint).parent / f"eval_{args.eval_dataset}_rank{args.eval_rank}.json"
            result = {
                'dataset': args.eval_dataset,
                'eval_rank': args.eval_rank,
                'perplexity': float(ppl),
                'avg_rank': float(avg_rank) if avg_rank is not None else None
            }
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResults saved to: {output_file}")

    print(f"\n{'='*80}")
    print(f"Evaluation Complete")
    print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate Matryoshka SVD models from SVD-LLM training'
    )

    # Model arguments
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (HuggingFace format directory)'
    )
    parser.add_argument(
        '--base_model',
        type=str,
        default=None,
        help='Base model name for tokenizer (e.g., meta-llama/Llama-2-7b-hf). '
             'Only needed if tokenizer not saved in checkpoint.'
    )

    # Evaluation arguments
    parser.add_argument(
        '--eval_dataset',
        type=str,
        default='wikitext2',
        choices=['wikitext2', 'c4', 'ptb'],
        help='Evaluation dataset'
    )
    parser.add_argument(
        '--eval_rank',
        default='adaptive',
        help='Evaluation rank: "adaptive" or integer (e.g., 32, 64, 128, 256)'
    )
    parser.add_argument(
        '--multi_rank_eval',
        action='store_true',
        help='Evaluate at multiple rank levels (r_min, r_mid, r_max, adaptive)'
    )
    parser.add_argument(
        '--n_eval_samples',
        type=int,
        default=256,
        help='Number of evaluation samples'
    )

    # Dataset arguments
    parser.add_argument(
        '--n_train_samples',
        type=int,
        default=256,
        help='Number of training samples (for data loading compatibility)'
    )
    parser.add_argument(
        '--seq_len',
        type=int,
        default=2048,
        help='Sequence length'
    )
    parser.add_argument(
        '--SAVE',
        action='store_true',
        help='Save generated dataset'
    )
    parser.add_argument(
        '--RECREATE',
        action='store_true',
        help='Regenerate dataset'
    )

    # Path arguments
    parser.add_argument(
        '--path_head_folder',
        type=str,
        default='./',
        help='Path for data and dataset cache'
    )

    # System arguments
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU device ID'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed'
    )
    parser.add_argument(
        '--save_results',
        action='store_true',
        help='Save evaluation results to JSON'
    )

    args = parser.parse_args()

    main(args)
