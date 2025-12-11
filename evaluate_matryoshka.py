"""
Evaluation script for Matryoshka SVD compressed models.

This script evaluates Matryoshka SVD models on:
1. Perplexity (wikitext2, c4, ptb)
2. Accuracy (arc_easy, arc_challenge, openbookqa, winogrande, hellaswag, piqa, mathqa)

Key features:
- Multi-rank evaluation (test at r_min, r_mid, r_max)
- Compression ratio reporting
- Average rank statistics

Usage:
    # Evaluate perplexity on wikitext2
    python evaluate_matryoshka.py \
        --model_path ./output_matryoshka/best_checkpoint.pt \
        --eval_metric ppl \
        --eval_dataset wikitext2 \
        --eval_rank adaptive

    # Evaluate at fixed rank
    python evaluate_matryoshka.py \
        --model_path ./output_matryoshka/best_checkpoint.pt \
        --eval_metric ppl \
        --eval_dataset wikitext2 \
        --eval_rank 32

    # Evaluate accuracy on commonsense tasks
    python evaluate_matryoshka.py \
        --model_path ./output_matryoshka/best_checkpoint.pt \
        --eval_metric accuracy \
        --eval_dataset arc_easy \
        --eval_rank adaptive

Author: Claude (Anthropic)
Date: 2025-12-11
"""

import argparse
import os
import sys
from pathlib import Path
import json

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project to path
sys.path.append(str(Path(__file__).parent))

from modules.matryoshka_svd import MatryoshkaSVDLayer, replace_linear_with_matryoshka_svd
from utils.datautils import prepare_train_loaders


def load_matryoshka_model(checkpoint_path, base_model=None):
    """
    Load Matryoshka SVD compressed model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint .pt file
        base_model: Optional base model name (if not in checkpoint)

    Returns:
        model: Loaded model
        tokenizer: Tokenizer
        args: Training arguments from checkpoint
    """
    print(f"\n{'='*80}")
    print(f"Loading Matryoshka SVD model")
    print(f"{'='*80}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Get training args
    if 'args' in checkpoint:
        train_args = argparse.Namespace(**checkpoint['args'])
        model_name = train_args.model
    else:
        if base_model is None:
            raise ValueError("Checkpoint doesn't contain args, must provide base_model")
        model_name = base_model
        train_args = None

    print(f"Base model: {model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load base model architecture
    print(f"Loading base model architecture...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )

    # Apply Matryoshka SVD replacement to match checkpoint structure
    # Extract parameters from training args or use defaults
    if train_args is not None:
        r_max = getattr(train_args, 'r_max', 64)
        r_min = getattr(train_args, 'r_min', 16)
        importance_strategy = getattr(train_args, 'importance_strategy', 'norm')
        temperature = getattr(train_args, 'temperature', 0.1)
        target_layers = getattr(train_args, 'target_layers', 'q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj').split(',')
    else:
        # Use defaults
        r_max = 64
        r_min = 16
        importance_strategy = 'norm'
        temperature = 0.1
        target_layers = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']

    print(f"\nApplying Matryoshka SVD replacement...")
    print(f"  r_max={r_max}, r_min={r_min}")
    print(f"  importance_strategy={importance_strategy}")
    print(f"  target_layers={target_layers}")

    # Replace linear layers with Matryoshka SVD layers
    # IMPORTANT: Do NOT compute SVD - we'll load weights from checkpoint
    model = replace_linear_with_matryoshka_svd(
        model,
        target_layers=target_layers,
        r_max=r_max,
        r_min=r_min,
        importance_strategy=importance_strategy,
        temperature=temperature,
        verbose=False,
        aggressive_memory_saving=True
    )

    print(f"\nLoading checkpoint weights...")
    # Load state dict with weights_only for security
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        print(f"✓ Model weights loaded successfully")
    except RuntimeError as e:
        print(f"Warning: Some keys mismatch, trying non-strict loading...")
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"✓ Model weights loaded (non-strict)")

    # Print Matryoshka SVD layer statistics
    print(f"\nMatryoshka SVD layers:")
    for name, module in model.named_modules():
        if isinstance(module, MatryoshkaSVDLayer):
            print(f"  {name}:")
            print(f"    - Rank range: [{module.r_min}, {module.r_max}]")
            print(f"    - Avg rank: {module.avg_rank_tracker.item():.1f}")
            print(f"    - Strategy: {module.importance_predictor.strategy}")

    return model, tokenizer, train_args


def set_fixed_rank_all_layers(model, fixed_rank):
    """
    Set all Matryoshka SVD layers to use a fixed rank.

    Args:
        model: Model with Matryoshka SVD layers
        fixed_rank: Fixed rank to use for all layers
    """
    for module in model.modules():
        if isinstance(module, MatryoshkaSVDLayer):
            # Calculate target importance to achieve desired rank
            target_importance = (fixed_rank - module.r_min) / (module.r_max - module.r_min)
            target_importance = max(0.0, min(1.0, target_importance))

            # Monkey-patch the importance predictor
            original_forward = module.importance_predictor.forward

            def fixed_importance_forward(x, attention_scores=None, imp=target_importance):
                batch, seq_len = x.shape[0], x.shape[1]
                return torch.full((batch, seq_len), imp, device=x.device, dtype=x.dtype)

            module.importance_predictor.forward = fixed_importance_forward
            module._original_forward = original_forward


def restore_adaptive_rank(model):
    """Restore adaptive rank selection for all layers."""
    for module in model.modules():
        if isinstance(module, MatryoshkaSVDLayer):
            if hasattr(module, '_original_forward'):
                module.importance_predictor.forward = module._original_forward
                delattr(module, '_original_forward')


def evaluate_perplexity(model, dataset, limit, eval_rank='adaptive'):
    """
    Evaluate perplexity on a dataset.

    Args:
        model: Model to evaluate
        dataset: Input IDs tensor of shape [batch, seq_len]
        limit: Maximum number of samples to evaluate
        eval_rank: 'adaptive' or integer for fixed rank

    Returns:
        ppl: Perplexity score
        avg_rank: Average rank used (only for adaptive)
    """
    nsamples, seqlen = dataset.size()

    # Set evaluation rank
    if eval_rank != 'adaptive':
        set_fixed_rank_all_layers(model, int(eval_rank))

    nlls = []
    ranks = []

    model.eval()

    for i in tqdm(range(min(nsamples, limit)), desc=f"Evaluating (rank={eval_rank})"):
        with torch.no_grad():
            input_ids = dataset[i:i+1, :-1].to(model.device)
            labels = dataset[i:i+1, 1:].contiguous()

            # Forward pass
            logits = model(input_ids=input_ids, use_cache=False)[0]

            if torch.isfinite(logits).all():
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()

                loss_fct = nn.CrossEntropyLoss(reduction='none')
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                nlls.append(loss)

                # Track average rank for adaptive mode
                if eval_rank == 'adaptive':
                    total_rank = 0
                    count = 0
                    for module in model.modules():
                        if isinstance(module, MatryoshkaSVDLayer):
                            total_rank += module.avg_rank_tracker.item()
                            count += 1
                    if count > 0:
                        ranks.append(total_rank / count)

            # Clear cache periodically
            if (i + 1) % 10 == 0:
                torch.cuda.empty_cache()

    # Restore adaptive rank
    if eval_rank != 'adaptive':
        restore_adaptive_rank(model)

    ppl = torch.exp(torch.stack(nlls).sum() / (len(nlls) * seqlen))
    avg_rank = np.mean(ranks) if ranks else None

    torch.cuda.empty_cache()

    return ppl.item(), avg_rank


def evaluate_commonsense(model, tokenizer, eval_dataset_name, eval_rank='adaptive'):
    """
    Evaluate accuracy on commonsense reasoning tasks.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        eval_dataset_name: Task name (e.g., 'arc_easy')
        eval_rank: 'adaptive' or integer for fixed rank

    Returns:
        results: Evaluation results dictionary
    """
    import lm_eval
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM

    # Set evaluation rank
    if eval_rank != 'adaptive':
        set_fixed_rank_all_layers(model, int(eval_rank))

    hflm = HFLM(pretrained=model, tokenizer=tokenizer)

    results = evaluator.simple_evaluate(
        model=hflm,
        tasks=[eval_dataset_name]
    )

    # Restore adaptive rank
    if eval_rank != 'adaptive':
        restore_adaptive_rank(model)

    return results['results']


def evaluate_multi_rank(model, dataset, limit, ranks_to_test):
    """
    Evaluate perplexity at multiple rank levels.

    Args:
        model: Model to evaluate
        dataset: Dataset
        limit: Max samples
        ranks_to_test: List of ranks to test (can include 'adaptive')

    Returns:
        results: Dictionary of {rank: (ppl, avg_rank)}
    """
    results = {}

    for rank in ranks_to_test:
        print(f"\n{'='*80}")
        print(f"Evaluating at rank: {rank}")
        print(f"{'='*80}")

        ppl, avg_rank = evaluate_perplexity(model, dataset, limit, eval_rank=rank)
        results[str(rank)] = {
            'perplexity': ppl,
            'avg_rank': avg_rank
        }

        print(f"Perplexity: {ppl:.4f}")
        if avg_rank is not None:
            print(f"Average rank: {avg_rank:.1f}")

    return results


def print_compression_stats(model):
    """Print compression statistics for all Matryoshka SVD layers."""
    print(f"\n{'='*80}")
    print(f"Compression Statistics")
    print(f"{'='*80}")

    total_original = 0
    total_compressed = 0
    layer_count = 0

    for name, module in model.named_modules():
        if isinstance(module, MatryoshkaSVDLayer):
            original_params = module.output_size * module.input_size
            avg_rank = module.avg_rank_tracker.item()
            if avg_rank == 0:
                avg_rank = (module.r_min + module.r_max) / 2

            compressed_params = module.output_size * avg_rank + avg_rank + avg_rank * module.input_size
            compression_ratio = compressed_params / original_params

            total_original += original_params
            total_compressed += compressed_params
            layer_count += 1

            print(f"{name}:")
            print(f"  Original params: {original_params:,}")
            print(f"  Compressed params: {compressed_params:,.0f}")
            print(f"  Compression ratio: {compression_ratio:.2%}")
            print(f"  Avg rank: {avg_rank:.1f} / {module.r_max}")

    if layer_count > 0:
        overall_ratio = total_compressed / total_original
        print(f"\nOverall:")
        print(f"  Total original: {total_original:,}")
        print(f"  Total compressed: {total_compressed:,.0f}")
        print(f"  Overall compression: {overall_ratio:.2%}")
        print(f"  Parameter reduction: {(1 - overall_ratio):.2%}")


def main(args):
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load model
    model, tokenizer, train_args = load_matryoshka_model(
        args.model_path,
        base_model=args.base_model
    )

    # Move to GPU
    model.to(device)

    # Print compression stats
    print_compression_stats(model)

    # Evaluation
    if args.eval_metric == "ppl":
        valid_datasets = {"wikitext2", "c4", "ptb"}
        if args.eval_dataset not in valid_datasets:
            raise ValueError(f"eval_dataset must be one of {valid_datasets}")

        # Prepare dataset
        DATASET_NAME = args.eval_dataset
        path_head_folder = Path(args.path_head_folder)
        data_cache_dir = path_head_folder / 'data_cache' / DATASET_NAME
        data_cache_dir.mkdir(parents=True, exist_ok=True)
        dataset_cache_dir = path_head_folder / 'datasets' / DATASET_NAME
        dataset_cache_dir.mkdir(parents=True, exist_ok=True)

        # Use train_args if available, otherwise use eval args
        eval_args = train_args if train_args is not None else args

        tokenized_traindata, tokenized_valdata = prepare_train_loaders(
            tokenizer, DATASET_NAME, data_cache_dir, dataset_cache_dir, eval_args
        )

        input_ids = torch.cat([_["input_ids"].unsqueeze(0) for _ in tokenized_valdata], 0)

        # Multi-rank evaluation
        if args.multi_rank_eval:
            # Get r_min and r_max from first Matryoshka layer
            r_min, r_max = None, None
            for module in model.modules():
                if isinstance(module, MatryoshkaSVDLayer):
                    r_min, r_max = module.r_min, module.r_max
                    break

            if r_min is not None:
                r_mid = (r_min + r_max) // 2
                ranks_to_test = ['adaptive', r_min, r_mid, r_max]
            else:
                ranks_to_test = ['adaptive']

            results = evaluate_multi_rank(model, input_ids, args.n_eval_samples, ranks_to_test)

            # Print summary
            print(f"\n{'='*80}")
            print(f"Multi-Rank Evaluation Summary")
            print(f"{'='*80}")
            for rank, metrics in results.items():
                print(f"Rank {rank}:")
                print(f"  Perplexity: {metrics['perplexity']:.4f}")
                if metrics['avg_rank'] is not None:
                    print(f"  Avg rank: {metrics['avg_rank']:.1f}")

            # Save results
            if args.save_results:
                output_path = Path(args.model_path).parent / f"eval_results_{DATASET_NAME}.json"
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"\nResults saved to: {output_path}")

        else:
            # Single rank evaluation
            ppl, avg_rank = evaluate_perplexity(
                model, input_ids, args.n_eval_samples, eval_rank=args.eval_rank
            )

            print(f"\n{'='*80}")
            print(f"Evaluation Results")
            print(f"{'='*80}")
            print(f"Dataset: {DATASET_NAME}")
            print(f"Rank: {args.eval_rank}")
            print(f"Perplexity: {ppl:.4f}")
            if avg_rank is not None:
                print(f"Average rank: {avg_rank:.1f}")

    elif args.eval_metric == "accuracy":
        valid_datasets = {
            "arc_easy", "arc_challenge", "openbookqa",
            "winogrande", "hellaswag", "piqa", "mathqa"
        }
        if args.eval_dataset not in valid_datasets:
            raise ValueError(f"eval_dataset must be one of {valid_datasets}")

        results = evaluate_commonsense(
            model, tokenizer, args.eval_dataset, eval_rank=args.eval_rank
        )

        print(f"\n{'='*80}")
        print(f"Accuracy Results")
        print(f"{'='*80}")
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Matryoshka SVD compressed models')

    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint (.pt file)')
    parser.add_argument('--base_model', type=str, default=None,
                       help='Base model name (if not in checkpoint)')

    # Evaluation arguments
    parser.add_argument('--eval_metric', type=str, default='ppl',
                       choices=['ppl', 'accuracy'],
                       help='Evaluation metric')
    parser.add_argument('--eval_dataset', type=str, default='wikitext2',
                       choices=['wikitext2', 'c4', 'ptb', 'arc_easy', 'arc_challenge',
                               'openbookqa', 'winogrande', 'hellaswag', 'piqa', 'mathqa'],
                       help='Evaluation dataset')
    parser.add_argument('--eval_rank', default='adaptive',
                       help='Evaluation rank: "adaptive" or integer (e.g., 32, 64)')
    parser.add_argument('--multi_rank_eval', action='store_true',
                       help='Evaluate at multiple rank levels (r_min, r_mid, r_max, adaptive)')

    # Dataset arguments
    parser.add_argument('--n_train_samples', type=int, default=256,
                       help='Number of training samples (for data loading)')
    parser.add_argument('--n_eval_samples', type=int, default=256,
                       help='Number of evaluation samples')
    parser.add_argument('--seq_len', type=int, default=512,
                       help='Sequence length')
    parser.add_argument('--SAVE', action='store_true', default=False,
                       help='Save generated dataset')
    parser.add_argument('--RECREATE', action='store_true', default=False,
                       help='Regenerate dataset')
    parser.add_argument('--DO_SAMPLE', action='store_true', default=False,
                       help='Sample dataset')

    # Path arguments
    parser.add_argument('--path_head_folder', type=str, default='./',
                       help='Path for data and dataset cache')
    parser.add_argument('--path_head_folder_output', type=str, default='./results',
                       help='Output path')

    # System arguments
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed')
    parser.add_argument('--save_results', action='store_true',
                       help='Save evaluation results to JSON')

    args = parser.parse_args()

    main(args)
