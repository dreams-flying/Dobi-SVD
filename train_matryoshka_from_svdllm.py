"""
Train Matryoshka SVD on top of SVD-LLM's whitened decomposition.

This script:
1. Loads U, V matrices from SVD-LLM output
2. Creates MatryoshkaSVDLayer with per-token rank predictor
3. Trains using multi-scale training strategy
4. Evaluates at different compression levels

Author: Claude
Date: 2025-12-15
"""

import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import random
import os
from typing import Dict, List, Optional
from pathlib import Path

from modules.matryoshka_svd_layer import MatryoshkaSVDLayer, RankPredictor


class MatryoshkaTrainer(Trainer):
    """
    Custom Trainer for Matryoshka SVD with multi-scale training.
    """

    def __init__(
        self,
        *args,
        r_max: int = 256,
        r_min: int = 64,
        lambda_rank: float = 0.001,
        multiscale_frequency: float = 0.5,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.r_max = r_max
        self.r_min = r_min
        self.lambda_rank = lambda_rank
        self.multiscale_frequency = multiscale_frequency

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute loss with multi-scale training and rank regularization.
        """
        # Multi-scale training
        use_multiscale = random.random() < self.multiscale_frequency

        if use_multiscale and self.model.training:
            # Sample multiple ranks and average their losses
            sampled_ranks = [
                self.r_min,
                self.r_max,
                random.randint(self.r_min + 1, self.r_max - 1)
            ]

            total_loss = 0
            for rank in sampled_ranks:
                # Set fixed rank for all MatryoshkaSVDLayers
                set_model_rank(model, rank)

                # Forward pass
                outputs = model(**inputs)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                total_loss += loss

                # Clear fixed rank
                set_model_rank(model, None)

            main_loss = total_loss / len(sampled_ranks)
            outputs = None  # We don't return outputs in multiscale mode

        else:
            # Regular forward (dynamic rank prediction)
            set_model_rank(model, None)
            outputs = model(**inputs)
            main_loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]

        # Add rank regularization (encourage lower average rank)
        rank_reg = compute_rank_regularization(model)
        total_loss = main_loss + self.lambda_rank * rank_reg

        return (total_loss, outputs) if return_outputs else total_loss


def set_model_rank(model: nn.Module, rank: Optional[float]):
    """Set fixed rank for all MatryoshkaSVDLayers in model."""
    for module in model.modules():
        if isinstance(module, MatryoshkaSVDLayer):
            module.set_fixed_rank(rank)


def compute_rank_regularization(model: nn.Module) -> torch.Tensor:
    """
    Compute rank regularization term.

    Encourages the model to use lower average rank.
    """
    total_rank = torch.tensor(0.0, device=next(model.parameters()).device)
    count = 0

    for module in model.modules():
        if isinstance(module, MatryoshkaSVDLayer):
            if module.use_rank_predictor and module.rank_predictor is not None:
                # Get average predicted rank (normalized to [0, 1])
                avg_rank = module.get_effective_rank()
                normalized = (avg_rank - module.r_min) / (module.r_max - module.r_min)
                total_rank += normalized
                count += 1

    return total_rank / max(count, 1)


def load_svdllm_model(
    model_name: str,
    svd_model_path: str,
    r_max: int,
    r_min: int,
    device: str = 'cuda'
) -> nn.Module:
    """
    Load SVD-LLM compressed model and convert to Matryoshka.

    Args:
        model_name: Original model name (e.g., 'meta-llama/Llama-2-7b-hf')
        svd_model_path: Path to SVD-LLM output model
        r_max: Maximum rank for Matryoshka
        r_min: Minimum rank for Matryoshka
        device: Device to load on

    Returns:
        Model with MatryoshkaSVDLayers
    """
    print(f"Loading SVD-LLM model from {svd_model_path}...")

    # Convert to absolute path if relative
    svd_model_path = os.path.abspath(svd_model_path)

    # Check if path exists
    if not os.path.exists(svd_model_path):
        raise ValueError(f"SVD model path does not exist: {svd_model_path}")

    model = None

    # Try loading as SVD-LLM format first
    try:
        import sys
        svdllm_path = '/home/user/SVD-LLM'
        if os.path.exists(svdllm_path) and svdllm_path not in sys.path:
            sys.path.insert(0, svdllm_path)

        from component.svd_llama import SVD_LlamaForCausalLM
        print("  → Loading as SVD-LLM format...")
        model = SVD_LlamaForCausalLM.from_pretrained(
            svd_model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        print("  ✓ Successfully loaded SVD-LLM model")

    except ImportError as e:
        print(f"  ✗ SVD-LLM components not available: {e}")
        print("  → Attempting to load as standard HuggingFace model...")

    except Exception as e:
        print(f"  ✗ Error loading as SVD-LLM format: {e}")
        print("  → Attempting to load as standard HuggingFace model...")

    # Fallback to standard loading
    if model is None:
        try:
            print(f"  → Loading from local path: {svd_model_path}")
            model = AutoModelForCausalLM.from_pretrained(
                svd_model_path,
                trust_remote_code=True,
                local_files_only=True
            )
            print("  ✓ Successfully loaded as standard model")
        except Exception as e:
            print(f"  ✗ Failed to load model: {e}")
            raise ValueError(
                f"Could not load model from {svd_model_path}. "
                f"Please ensure:\n"
                f"  1. Path exists and contains model files\n"
                f"  2. SVD-LLM is installed if using SVD-LLM format\n"
                f"  3. Model is in HuggingFace format\n"
                f"Error: {e}"
            )

    # Convert SVD layers to Matryoshka layers
    print("Converting to Matryoshka layers...")
    model = convert_to_matryoshka(model, r_max, r_min)

    return model.to(device)


def convert_to_matryoshka(
    model: nn.Module,
    r_max: int,
    r_min: int
) -> nn.Module:
    """
    Convert SVD-LLM layers to MatryoshkaSVDLayers.

    This function finds all U/V projection pairs and replaces them
    with MatryoshkaSVDLayer.
    """
    # This is a simplified version - actual implementation depends on
    # SVD-LLM's model structure

    # For now, we'll add rank predictors to existing SVD layers
    # without replacing them, since SVD-LLM already has the right structure

    for name, module in model.named_modules():
        # Look for SVD_LlamaMLP or SVD_LlamaAttention modules
        if hasattr(module, 'up_v_proj') and hasattr(module, 'up_u_proj'):
            # This is an SVD layer, add rank predictor
            in_features = module.up_v_proj.in_features
            current_rank = module.up_v_proj.out_features

            # Add rank predictor
            module.rank_predictor = RankPredictor(
                hidden_dim=in_features,
                r_max=min(r_max, current_rank),
                r_min=r_min
            )

            # Add gating mechanism
            module.use_matryoshka = True
            module.gating_tau = 0.1

            # Monkey-patch the forward method to use Matryoshka gating
            original_forward = module.forward

            def matryoshka_forward(self, x):
                if hasattr(self, 'use_matryoshka') and self.use_matryoshka:
                    # Use Matryoshka gating
                    return matryoshka_mlp_forward(self, x)
                else:
                    # Use original forward
                    return original_forward(x)

            module.forward = lambda x: matryoshka_forward(module, x)

    return model


def matryoshka_mlp_forward(module, x):
    """
    Matryoshka forward for MLP module.

    Applies soft gating based on predicted rank.
    """
    # Predict rank
    if hasattr(module, 'rank_predictor'):
        rank = module.rank_predictor(x)  # [batch, seq, 1]
    else:
        rank = torch.tensor(module.up_v_proj.out_features, device=x.device)

    # V projections
    up = module.up_v_proj(x)
    gate = module.gate_v_proj(x)

    # Compute soft gating
    r_max = up.shape[-1]
    positions = torch.arange(1, r_max + 1, device=x.device, dtype=x.dtype)
    diff = rank - positions.unsqueeze(0).unsqueeze(0)
    gates = torch.sigmoid(diff / module.gating_tau)

    # Apply gating
    up = up * gates
    gate = gate * gates

    # Apply activation
    gate = module.act_fn(gate)

    # U projections
    down = module.down_v_proj(gate * up)
    down = down * gates  # Gate again before U projection
    output = module.down_u_proj(down)

    return output


def prepare_dataset(dataset_name: str, tokenizer, max_length: int = 512):
    """Prepare training dataset."""
    if dataset_name == 'wikitext2':
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
        column = 'text'
    elif dataset_name == 'c4':
        dataset = load_dataset('allenai/c4', 'en', streaming=True)
        column = 'text'
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    def tokenize_function(examples):
        return tokenizer(
            examples[column],
            truncation=True,
            max_length=max_length,
            padding='max_length'
        )

    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset['train'].column_names
    )

    return tokenized


def main():
    parser = argparse.ArgumentParser(
        description='Train Matryoshka SVD on top of SVD-LLM'
    )

    # Model arguments
    parser.add_argument(
        '--model',
        type=str,
        default='meta-llama/Llama-2-7b-hf',
        help='Original model name'
    )
    parser.add_argument(
        '--svdllm_model',
        type=str,
        required=True,
        help='Path to SVD-LLM compressed model'
    )

    # Matryoshka arguments
    parser.add_argument('--r_max', type=int, default=256, help='Maximum rank')
    parser.add_argument('--r_min', type=int, default=64, help='Minimum rank')
    parser.add_argument(
        '--lambda_rank',
        type=float,
        default=0.001,
        help='Rank regularization weight'
    )
    parser.add_argument(
        '--multiscale_frequency',
        type=float,
        default=0.5,
        help='Frequency of multi-scale training (0-1)'
    )
    parser.add_argument(
        '--freeze_uv',
        action='store_true',
        help='Freeze U and V, only train rank predictor'
    )

    # Training arguments
    parser.add_argument('--dataset', type=str, default='wikitext2')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_train_epochs', type=int, default=1)
    parser.add_argument('--per_device_train_batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--save_steps', type=int, default=500)
    parser.add_argument('--eval_steps', type=int, default=500)

    args = parser.parse_args()

    # Load model
    print("=" * 80)
    print("Step 1: Loading SVD-LLM model")
    print("=" * 80)

    model = load_svdllm_model(
        model_name=args.model,
        svd_model_path=args.svdllm_model,
        r_max=args.r_max,
        r_min=args.r_min
    )

    # Freeze U and V if requested
    if args.freeze_uv:
        print("\nFreezing U and V projections...")
        for name, param in model.named_parameters():
            if '_u_proj' in name or '_v_proj' in name:
                param.requires_grad = False
            elif 'rank_predictor' in name:
                param.requires_grad = True

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare dataset
    print("\n" + "=" * 80)
    print("Step 2: Preparing dataset")
    print("=" * 80)

    dataset = prepare_dataset(args.dataset, tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        evaluation_strategy='steps',
        save_strategy='steps',
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=False,  # Usually not needed for inference-only compression
        dataloader_num_workers=4,
        remove_unused_columns=False
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Create trainer
    print("\n" + "=" * 80)
    print("Step 3: Starting Matryoshka training")
    print("=" * 80)
    print(f"  r_max: {args.r_max}")
    print(f"  r_min: {args.r_min}")
    print(f"  Rank regularization: {args.lambda_rank}")
    print(f"  Multi-scale frequency: {args.multiscale_frequency}")
    print(f"  Freeze U/V: {args.freeze_uv}")
    print("=" * 80)

    trainer = MatryoshkaTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset.get('validation', dataset.get('test')),
        data_collator=data_collator,
        r_max=args.r_max,
        r_min=args.r_min,
        lambda_rank=args.lambda_rank,
        multiscale_frequency=args.multiscale_frequency
    )

    # Train
    trainer.train()

    # Save final model
    print("\nSaving final model...")
    trainer.save_model(os.path.join(args.output_dir, 'final'))

    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)
    print(f"Model saved to: {args.output_dir}/final")
    print("\nTo evaluate at different ranks:")
    print(f"  python evaluate_matryoshka.py \\")
    print(f"    --checkpoint {args.output_dir}/final \\")
    print(f"    --target_rank 64")


if __name__ == '__main__':
    main()
