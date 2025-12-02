#!/usr/bin/env python3
"""
Quick Pruning Tool

One-command tool for quick deployment optimization.
Combines pruning, optional fine-tuning, and evaluation.

Usage:
    # Quick pruning without fine-tuning
    python tools/quick_prune.py \
        --model_path results/trained_model \
        --threshold 0.1 \
        --output results/deployed_model

    # With auto fine-tuning
    python tools/quick_prune.py \
        --model_path results/trained_model \
        --threshold 0.1 \
        --auto_finetune \
        --output results/deployed_model
"""

import torch
import argparse
import json
import os
from pathlib import Path
import sys
import subprocess
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_command(cmd, description):
    """Run a command and print status."""
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}\n")

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - start_time

    if result.returncode == 0:
        print(f"\n✓ {description} completed in {elapsed:.1f}s")
        return True
    else:
        print(f"\n✗ {description} failed (exit code: {result.returncode})")
        return False


def main():
    parser = argparse.ArgumentParser(description='Quick pruning for deployment')

    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--threshold', type=float, default=0.1,
                       help='Pruning threshold (default 0.1 = 10%%)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for deployed model')

    # Optional fine-tuning
    parser.add_argument('--auto_finetune', action='store_true',
                       help='Automatically fine-tune after pruning')
    parser.add_argument('--finetune_epochs', type=int, default=3,
                       help='Number of fine-tuning epochs (default 3)')
    parser.add_argument('--finetune_samples', type=int, default=5000,
                       help='Number of training samples for fine-tuning (default 5000)')
    parser.add_argument('--finetune_lr', type=float, default=1e-5,
                       help='Fine-tuning learning rate (default 1e-5)')

    # Evaluation
    parser.add_argument('--skip_eval', action='store_true',
                       help='Skip evaluation')

    # Data
    parser.add_argument('--val_data', type=str, default=None,
                       help='Validation dataset (default: wikitext)')
    parser.add_argument('--train_data', type=str, default=None,
                       help='Training dataset for fine-tuning (default: wikitext)')

    args = parser.parse_args()

    print("="*70)
    print("Quick Pruning Tool")
    print("="*70)
    print(f"Model: {args.model_path}")
    print(f"Threshold: {args.threshold}")
    print(f"Auto fine-tune: {args.auto_finetune}")
    print(f"Output: {args.output}")
    print("="*70)

    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Prune subspaces
    print("\n" + "="*70)
    print("STEP 1: Pruning subspaces")
    print("="*70)

    prune_cmd = [
        'python', 'tools/prune_subspaces.py',
        '--model_path', args.model_path,
        '--threshold', str(args.threshold),
        '--output', str(output_path / 'pruned')
    ]

    if args.val_data:
        prune_cmd.extend(['--val_data', args.val_data])

    if not run_command(prune_cmd, "Subspace pruning"):
        print("\n✗ Pruning failed, aborting")
        return 1

    current_model = str(output_path / 'pruned')

    # Step 2: Fine-tune (optional)
    if args.auto_finetune:
        print("\n" + "="*70)
        print("STEP 2: Fine-tuning pruned model")
        print("="*70)

        finetune_cmd = [
            'python', 'tools/finetune_optimized.py',
            '--model_path', current_model,
            '--n_epochs', str(args.finetune_epochs),
            '--lr', str(args.finetune_lr),
            '--max_samples', str(args.finetune_samples),
            '--output', str(output_path / 'finetuned')
        ]

        if args.train_data:
            finetune_cmd.extend(['--train_data', args.train_data])

        if not run_command(finetune_cmd, "Fine-tuning"):
            print("\n⚠ Fine-tuning failed, using pruned model without fine-tuning")
        else:
            current_model = str(output_path / 'finetuned')

    # Step 3: Evaluate (optional)
    if not args.skip_eval:
        print("\n" + "="*70)
        print("STEP 3: Evaluating optimized model")
        print("="*70)

        eval_cmd = [
            'python', 'tools/evaluate_compression.py',
            '--original_model', args.model_path,
            '--compressed_model', current_model,
            '--output', str(output_path / 'evaluation.json')
        ]

        if args.val_data:
            eval_cmd.extend(['--test_data', args.val_data])

        run_command(eval_cmd, "Evaluation")

    # Step 4: Copy final model to output
    print("\n" + "="*70)
    print("STEP 4: Finalizing deployment model")
    print("="*70)

    if current_model != args.output:
        import shutil
        final_output = Path(args.output) / 'final'
        if final_output.exists():
            shutil.rmtree(final_output)

        shutil.copytree(current_model, final_output)
        print(f"✓ Final model copied to: {final_output}")

    # Generate summary
    summary = {
        'original_model': args.model_path,
        'pruning_threshold': args.threshold,
        'fine_tuned': args.auto_finetune,
        'output_model': str(Path(args.output) / 'final'),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    # Load pruning statistics
    pruning_stats_path = output_path / 'pruned' / 'pruning_statistics.json'
    if pruning_stats_path.exists():
        with open(pruning_stats_path, 'r') as f:
            summary['pruning_statistics'] = json.load(f)

    # Load evaluation results
    eval_path = output_path / 'evaluation.json'
    if eval_path.exists():
        with open(eval_path, 'r') as f:
            summary['evaluation'] = json.load(f)

    # Save summary
    summary_path = output_path / 'deployment_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Deployment summary saved to: {summary_path}")

    # Final report
    print("\n" + "="*70)
    print("✓ DEPLOYMENT COMPLETE")
    print("="*70)
    print(f"\nFinal model: {Path(args.output) / 'final'}")
    print(f"Summary: {summary_path}")

    if not args.skip_eval and eval_path.exists():
        print(f"Evaluation: {eval_path}")

    print("\nNext steps:")
    print("1. Test model quality on your tasks")
    print("2. Export to ONNX for production deployment")
    print("3. Measure inference latency")

    print("\n" + "="*70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
