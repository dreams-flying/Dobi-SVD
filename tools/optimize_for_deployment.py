#!/usr/bin/env python3
"""
End-to-End Deployment Optimization Tool

Automatically applies the best combination of optimization strategies
based on target compression ratio and quality requirements.

Strategies:
    - conservative: 30-50% compression, minimal quality loss
    - balanced: 50-70% compression, acceptable quality loss
    - aggressive: 70-80%+ compression, higher quality loss acceptable

Usage:
    python tools/optimize_for_deployment.py \
        --model_path results/trained_model \
        --strategy balanced \
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


def get_strategy_config(strategy):
    """Get optimization configuration for a strategy."""

    configs = {
        'conservative': {
            'description': '30-50% compression, <2% quality loss',
            'pruning_threshold': 0.05,  # Only prune very low usage
            'apply_parameter_sharing': False,
            'quantization': False,
            'finetune_epochs': 2,
            'target_compression': 0.4
        },
        'balanced': {
            'description': '50-70% compression, 2-5% quality loss',
            'pruning_threshold': 0.1,  # Prune <10% usage
            'apply_parameter_sharing': True,
            'sharing_type': 'shared_uv',
            'quantization': False,
            'finetune_epochs': 3,
            'target_compression': 0.6
        },
        'aggressive': {
            'description': '70-80%+ compression, 5-10% quality loss',
            'pruning_threshold': 0.15,  # Prune <15% usage
            'apply_parameter_sharing': True,
            'sharing_type': 'shared_uv',
            'quantization': True,
            'finetune_epochs': 5,
            'target_compression': 0.75
        }
    }

    return configs.get(strategy, configs['balanced'])


def main():
    parser = argparse.ArgumentParser(description='End-to-end deployment optimization')

    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--strategy', type=str, default='balanced',
                       choices=['conservative', 'balanced', 'aggressive'],
                       help='Optimization strategy')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for deployed model')

    # Override options
    parser.add_argument('--target_compression', type=float, default=None,
                       help='Target compression ratio (overrides strategy default)')
    parser.add_argument('--skip_finetune', action='store_true',
                       help='Skip fine-tuning step')
    parser.add_argument('--enable_quantization', action='store_true',
                       help='Force enable quantization')

    # Data
    parser.add_argument('--val_data', type=str, default=None,
                       help='Validation dataset')
    parser.add_argument('--train_data', type=str, default=None,
                       help='Training dataset for fine-tuning')

    args = parser.parse_args()

    # Get strategy configuration
    config = get_strategy_config(args.strategy)

    if args.target_compression is not None:
        config['target_compression'] = args.target_compression

    if args.enable_quantization:
        config['quantization'] = True

    print("="*70)
    print("End-to-End Deployment Optimization")
    print("="*70)
    print(f"Model: {args.model_path}")
    print(f"Strategy: {args.strategy}")
    print(f"Description: {config['description']}")
    print(f"Target compression: {config['target_compression']*100:.0f}%")
    print(f"Output: {args.output}")
    print("\nOptimization plan:")
    print(f"  1. Prune subspaces (threshold={config['pruning_threshold']})")
    if config['apply_parameter_sharing']:
        print(f"  2. Apply parameter sharing (type={config.get('sharing_type', 'shared_uv')})")
    if config['quantization']:
        print(f"  3. Mixed-precision quantization")
    if not args.skip_finetune:
        print(f"  4. Fine-tune ({config['finetune_epochs']} epochs)")
    print(f"  5. Evaluate and export")
    print("="*70)

    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    current_model = args.model_path
    optimization_log = []

    # Step 1: Prune subspaces
    print("\n" + "="*70)
    print("STEP 1: Pruning subspaces")
    print("="*70)

    pruned_path = output_path / '01_pruned'
    prune_cmd = [
        'python', 'tools/prune_subspaces.py',
        '--model_path', current_model,
        '--threshold', str(config['pruning_threshold']),
        '--output', str(pruned_path)
    ]

    if args.val_data:
        prune_cmd.extend(['--val_data', args.val_data])

    if run_command(prune_cmd, "Subspace pruning"):
        current_model = str(pruned_path)
        optimization_log.append({
            'step': 'pruning',
            'threshold': config['pruning_threshold'],
            'success': True
        })
    else:
        print("\n⚠ Pruning failed, continuing without pruning")
        optimization_log.append({
            'step': 'pruning',
            'success': False
        })

    # Step 2: Parameter sharing (optional)
    if config.get('apply_parameter_sharing', False):
        print("\n" + "="*70)
        print("STEP 2: Applying parameter sharing")
        print("="*70)

        shared_path = output_path / '02_shared'
        sharing_cmd = [
            'python', 'tools/apply_parameter_sharing.py',
            '--model_path', current_model,
            '--sharing_type', config.get('sharing_type', 'shared_uv'),
            '--output', str(shared_path),
            '--verify'
        ]

        if run_command(sharing_cmd, "Parameter sharing"):
            current_model = str(shared_path)
            optimization_log.append({
                'step': 'parameter_sharing',
                'sharing_type': config.get('sharing_type'),
                'success': True
            })
        else:
            print("\n⚠ Parameter sharing failed, continuing without it")
            optimization_log.append({
                'step': 'parameter_sharing',
                'success': False
            })

    # Step 3: Quantization (optional)
    if config.get('quantization', False):
        print("\n" + "="*70)
        print("STEP 3: Mixed-precision quantization")
        print("="*70)

        quantized_path = output_path / '03_quantized'
        quant_cmd = [
            'python', 'tools/mixed_precision_quantize.py',
            '--model_path', current_model,
            '--output', str(quantized_path)
        ]

        # This tool may not exist yet, so we'll just log it as planned
        print(f"  (Quantization tool to be implemented)")
        print(f"  Planned: INT8 for high-usage subspaces, INT4 for low-usage")

        optimization_log.append({
            'step': 'quantization',
            'success': False,
            'note': 'Tool not yet implemented'
        })

    # Step 4: Fine-tune (optional)
    if not args.skip_finetune:
        print("\n" + "="*70)
        print("STEP 4: Fine-tuning")
        print("="*70)

        finetuned_path = output_path / '04_finetuned'
        finetune_cmd = [
            'python', 'tools/finetune_optimized.py',
            '--model_path', current_model,
            '--n_epochs', str(config['finetune_epochs']),
            '--lr', '1e-5',
            '--output', str(finetuned_path)
        ]

        if args.train_data:
            finetune_cmd.extend(['--train_data', args.train_data])

        if run_command(finetune_cmd, "Fine-tuning"):
            current_model = str(finetuned_path)
            optimization_log.append({
                'step': 'finetuning',
                'epochs': config['finetune_epochs'],
                'success': True
            })
        else:
            print("\n⚠ Fine-tuning failed, using model without fine-tuning")
            optimization_log.append({
                'step': 'finetuning',
                'success': False
            })

    # Step 5: Evaluate
    print("\n" + "="*70)
    print("STEP 5: Final evaluation")
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

    # Copy final model
    print("\n" + "="*70)
    print("Finalizing deployment model")
    print("="*70)

    import shutil
    final_path = output_path / 'final'
    if final_path.exists():
        shutil.rmtree(final_path)
    shutil.copytree(current_model, final_path)

    print(f"✓ Final model: {final_path}")

    # Generate comprehensive report
    report = {
        'original_model': args.model_path,
        'strategy': args.strategy,
        'strategy_config': config,
        'optimization_log': optimization_log,
        'final_model': str(final_path),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    # Load evaluation if exists
    eval_path = output_path / 'evaluation.json'
    if eval_path.exists():
        with open(eval_path, 'r') as f:
            report['evaluation'] = json.load(f)

    report_path = output_path / 'optimization_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"✓ Optimization report: {report_path}")

    # Print summary
    print("\n" + "="*70)
    print("✓ OPTIMIZATION COMPLETE")
    print("="*70)
    print(f"\nStrategy: {args.strategy}")
    print(f"Final model: {final_path}")

    if 'evaluation' in report:
        eval_data = report['evaluation']
        if 'compression_ratio' in eval_data:
            print(f"Compression ratio: {eval_data['compression_ratio']*100:.1f}%")
        if 'perplexity_delta' in eval_data:
            print(f"Perplexity change: {eval_data['perplexity_delta']:+.1f}%")

    print("\nOptimization steps completed:")
    for i, log_entry in enumerate(optimization_log, 1):
        status = "✓" if log_entry['success'] else "✗"
        print(f"  {status} {log_entry['step']}")

    print("\nNext steps:")
    print("1. Test model on your specific tasks")
    print("2. Export to ONNX: python tools/export_for_deployment.py")
    print("3. Deploy to production")

    print("\n" + "="*70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
