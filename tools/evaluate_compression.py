#!/usr/bin/env python3
"""
Compression Evaluation Tool

Comprehensively evaluates compressed model vs original:
- Parameter count and compression ratio
- Perplexity on test data
- Inference latency
- Memory usage

Usage:
    python tools/evaluate_compression.py \
        --original_model /path/to/original \
        --compressed_model results/optimized_model \
        --test_data data/test.json \
        --output evaluation/report.json
"""

import torch
import argparse
import json
import os
import sys
from pathlib import Path
import time
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def count_parameters(model):
    """Count total and trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total': total_params,
        'trainable': trainable_params,
        'total_millions': total_params / 1e6,
        'trainable_millions': trainable_params / 1e6
    }


def calculate_model_size(model):
    """Calculate model size in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024 / 1024

    return {
        'param_size_mb': param_size / 1024 / 1024,
        'buffer_size_mb': buffer_size / 1024 / 1024,
        'total_size_mb': size_mb
    }


def evaluate_perplexity(model, tokenizer, dataset, device='cuda', max_samples=500):
    """Evaluate perplexity on dataset."""
    print(f"Evaluating perplexity on {len(dataset)} samples...")

    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for i, example in enumerate(tqdm(dataset.select(range(min(max_samples, len(dataset)))))):
            if i >= max_samples:
                break

            # Tokenize
            inputs = tokenizer(
                example['text'],
                return_tensors='pt',
                max_length=512,
                truncation=True
            )

            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Forward
            try:
                outputs = model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss

                total_loss += loss.item() * inputs['input_ids'].numel()
                total_tokens += inputs['input_ids'].numel()

            except Exception as e:
                print(f"Warning: Error processing example {i}: {e}")
                continue

    if total_tokens == 0:
        return None

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)

    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'total_tokens': total_tokens
    }


def measure_latency(model, tokenizer, device='cuda', n_runs=100, seq_len=512):
    """Measure inference latency."""
    print(f"Measuring latency over {n_runs} runs...")

    model.eval()

    # Prepare test input
    test_text = "The quick brown fox jumps over the lazy dog. " * 50
    inputs = tokenizer(
        test_text,
        return_tensors='pt',
        max_length=seq_len,
        truncation=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(**inputs)

    if device == 'cuda':
        torch.cuda.synchronize()

    # Measure
    latencies = []

    with torch.no_grad():
        for _ in tqdm(range(n_runs), desc='Measuring latency'):
            start = time.time()
            _ = model(**inputs)

            if device == 'cuda':
                torch.cuda.synchronize()

            end = time.time()
            latencies.append((end - start) * 1000)  # Convert to ms

    return {
        'mean_ms': np.mean(latencies),
        'std_ms': np.std(latencies),
        'min_ms': np.min(latencies),
        'max_ms': np.max(latencies),
        'median_ms': np.median(latencies),
        'seq_len': seq_len,
        'n_runs': n_runs
    }


def measure_memory(model, tokenizer, device='cuda'):
    """Measure peak memory usage."""
    if device != 'cuda':
        return None

    print("Measuring memory usage...")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    model.eval()

    # Test forward pass
    test_text = "The quick brown fox jumps over the lazy dog. " * 50
    inputs = tokenizer(
        test_text,
        return_tensors='pt',
        max_length=512,
        truncation=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        _ = model(**inputs)

    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB

    return {
        'peak_memory_mb': peak_memory
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate compression')

    parser.add_argument('--original_model', type=str, required=True,
                       help='Path to original model')
    parser.add_argument('--compressed_model', type=str, required=True,
                       help='Path to compressed model')
    parser.add_argument('--test_data', type=str, default=None,
                       help='Test dataset (default: wikitext)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for evaluation results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (default cuda)')
    parser.add_argument('--max_samples', type=int, default=500,
                       help='Max samples for perplexity evaluation (default 500)')
    parser.add_argument('--latency_runs', type=int, default=100,
                       help='Number of runs for latency measurement (default 100)')

    args = parser.parse_args()

    print("="*70)
    print("Compression Evaluation Tool")
    print("="*70)
    print(f"Original model: {args.original_model}")
    print(f"Compressed model: {args.compressed_model}")
    print(f"Output: {args.output}")
    print("="*70)

    results = {
        'original_model': args.original_model,
        'compressed_model': args.compressed_model
    }

    # Load test dataset
    print("\nLoading test dataset...")
    if args.test_data is None:
        test_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    else:
        test_dataset = load_dataset('json', data_files=args.test_data, split='train')

    print(f"Test dataset size: {len(test_dataset)}")

    # Evaluate original model
    print("\n" + "="*70)
    print("Evaluating ORIGINAL model")
    print("="*70)

    print("\nLoading original model...")
    original_model = AutoModelForCausalLM.from_pretrained(args.original_model)
    original_tokenizer = AutoTokenizer.from_pretrained(args.original_model)

    if original_tokenizer.pad_token is None:
        original_tokenizer.pad_token = original_tokenizer.eos_token

    original_model = original_model.to(args.device)

    # Count parameters
    print("\nCounting parameters...")
    original_params = count_parameters(original_model)
    original_size = calculate_model_size(original_model)

    print(f"  Total parameters: {original_params['total_millions']:.2f}M")
    print(f"  Model size: {original_size['total_size_mb']:.2f} MB")

    # Evaluate perplexity
    original_ppl = evaluate_perplexity(
        original_model,
        original_tokenizer,
        test_dataset,
        device=args.device,
        max_samples=args.max_samples
    )

    if original_ppl:
        print(f"  Perplexity: {original_ppl['perplexity']:.2f}")

    # Measure latency
    original_latency = measure_latency(
        original_model,
        original_tokenizer,
        device=args.device,
        n_runs=args.latency_runs
    )

    print(f"  Latency: {original_latency['mean_ms']:.2f} ± {original_latency['std_ms']:.2f} ms")

    # Measure memory
    if args.device == 'cuda':
        original_memory = measure_memory(original_model, original_tokenizer, device=args.device)
        print(f"  Peak memory: {original_memory['peak_memory_mb']:.2f} MB")
    else:
        original_memory = None

    results['original'] = {
        'parameters': original_params,
        'size': original_size,
        'perplexity': original_ppl,
        'latency': original_latency,
        'memory': original_memory
    }

    # Clear memory
    del original_model
    torch.cuda.empty_cache() if args.device == 'cuda' else None

    # Evaluate compressed model
    print("\n" + "="*70)
    print("Evaluating COMPRESSED model")
    print("="*70)

    print("\nLoading compressed model...")
    compressed_model = AutoModelForCausalLM.from_pretrained(args.compressed_model)
    compressed_tokenizer = AutoTokenizer.from_pretrained(args.compressed_model)

    if compressed_tokenizer.pad_token is None:
        compressed_tokenizer.pad_token = compressed_tokenizer.eos_token

    compressed_model = compressed_model.to(args.device)

    # Count parameters
    print("\nCounting parameters...")
    compressed_params = count_parameters(compressed_model)
    compressed_size = calculate_model_size(compressed_model)

    print(f"  Total parameters: {compressed_params['total_millions']:.2f}M")
    print(f"  Model size: {compressed_size['total_size_mb']:.2f} MB")

    # Evaluate perplexity
    compressed_ppl = evaluate_perplexity(
        compressed_model,
        compressed_tokenizer,
        test_dataset,
        device=args.device,
        max_samples=args.max_samples
    )

    if compressed_ppl:
        print(f"  Perplexity: {compressed_ppl['perplexity']:.2f}")

    # Measure latency
    compressed_latency = measure_latency(
        compressed_model,
        compressed_tokenizer,
        device=args.device,
        n_runs=args.latency_runs
    )

    print(f"  Latency: {compressed_latency['mean_ms']:.2f} ± {compressed_latency['std_ms']:.2f} ms")

    # Measure memory
    if args.device == 'cuda':
        compressed_memory = measure_memory(compressed_model, compressed_tokenizer, device=args.device)
        print(f"  Peak memory: {compressed_memory['peak_memory_mb']:.2f} MB")
    else:
        compressed_memory = None

    results['compressed'] = {
        'parameters': compressed_params,
        'size': compressed_size,
        'perplexity': compressed_ppl,
        'latency': compressed_latency,
        'memory': compressed_memory
    }

    # Calculate improvements
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)

    # Compression ratio
    param_reduction = 1 - (compressed_params['total'] / original_params['total'])
    size_reduction = 1 - (compressed_size['total_size_mb'] / original_size['total_size_mb'])

    print(f"\nParameter reduction: {param_reduction*100:.1f}%")
    print(f"  Original: {original_params['total_millions']:.2f}M")
    print(f"  Compressed: {compressed_params['total_millions']:.2f}M")

    print(f"\nSize reduction: {size_reduction*100:.1f}%")
    print(f"  Original: {original_size['total_size_mb']:.2f} MB")
    print(f"  Compressed: {compressed_size['total_size_mb']:.2f} MB")

    # Perplexity change
    if original_ppl and compressed_ppl:
        ppl_delta = (compressed_ppl['perplexity'] - original_ppl['perplexity']) / original_ppl['perplexity'] * 100
        print(f"\nPerplexity change: {ppl_delta:+.1f}%")
        print(f"  Original: {original_ppl['perplexity']:.2f}")
        print(f"  Compressed: {compressed_ppl['perplexity']:.2f}")
    else:
        ppl_delta = None

    # Latency speedup
    latency_speedup = original_latency['mean_ms'] / compressed_latency['mean_ms']
    print(f"\nLatency speedup: {latency_speedup:.2f}x")
    print(f"  Original: {original_latency['mean_ms']:.2f} ms")
    print(f"  Compressed: {compressed_latency['mean_ms']:.2f} ms")

    # Memory reduction
    if original_memory and compressed_memory:
        memory_reduction = 1 - (compressed_memory['peak_memory_mb'] / original_memory['peak_memory_mb'])
        print(f"\nMemory reduction: {memory_reduction*100:.1f}%")
        print(f"  Original: {original_memory['peak_memory_mb']:.2f} MB")
        print(f"  Compressed: {compressed_memory['peak_memory_mb']:.2f} MB")
    else:
        memory_reduction = None

    results['comparison'] = {
        'compression_ratio': param_reduction,
        'size_reduction': size_reduction,
        'perplexity_delta': ppl_delta,
        'latency_speedup': latency_speedup,
        'memory_reduction': memory_reduction
    }

    # Save results
    print(f"\nSaving evaluation results to: {args.output}")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*70)
    print("✓ Evaluation complete!")
    print("="*70)

    print(f"\nResults summary:")
    print(f"  Compression: {param_reduction*100:.1f}%")
    print(f"  Quality impact: {ppl_delta:+.1f}%" if ppl_delta else "  Quality: N/A")
    print(f"  Speedup: {latency_speedup:.2f}x")

    print(f"\nDetailed results saved to: {args.output}")


if __name__ == '__main__':
    main()
