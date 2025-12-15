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
) -> tuple:
    """
    Load SVD-LLM compressed model and convert to Matryoshka.

    Args:
        model_name: Original model name (e.g., 'meta-llama/Llama-2-7b-hf')
        svd_model_path: Path to SVD-LLM output .pt file
        r_max: Maximum rank for Matryoshka
        r_min: Minimum rank for Matryoshka
        device: Device to load on

    Returns:
        Tuple of (model, tokenizer) with MatryoshkaSVDLayers
    """
    print(f"Loading SVD-LLM model from {svd_model_path}...")
    print(f"  Model name: {model_name}")
    print(f"  Matryoshka rank range: [{r_min}, {r_max}]")

    # Convert to absolute path if relative
    svd_model_path = os.path.abspath(svd_model_path)

    # Check if path exists
    if not os.path.exists(svd_model_path):
        raise ValueError(f"SVD model path does not exist: {svd_model_path}")

    # Add SVD-LLM to path for custom layer definitions (BEFORE torch.load!)
    import sys

    # Try to detect SVD-LLM path from model path
    # e.g., /data1/user/SVD-LLM/svd_llm_output/model.pt -> /data1/user/SVD-LLM
    potential_paths = []

    # Method 1: Extract from model path
    model_dir = os.path.dirname(svd_model_path)
    if 'SVD-LLM' in model_dir:
        svdllm_candidate = model_dir.split('SVD-LLM')[0] + 'SVD-LLM'
        potential_paths.append(svdllm_candidate)

    # Method 2: Check parent directories
    parent = os.path.dirname(model_dir)
    potential_paths.append(parent)
    potential_paths.append(os.path.join(parent, 'SVD-LLM'))

    # Method 3: Common locations
    potential_paths.extend([
        '/home/user/SVD-LLM',
        '/root/SVD-LLM',
        os.path.expanduser('~/SVD-LLM')
    ])

    svdllm_path = None
    for path in potential_paths:
        if os.path.exists(path) and os.path.exists(os.path.join(path, 'component')):
            svdllm_path = path
            break

    if svdllm_path is None:
        raise ValueError(
            f"Could not find SVD-LLM installation.\n"
            f"Searched in: {potential_paths}\n"
            f"Please ensure SVD-LLM is installed and the 'component' directory exists.\n"
            f"You can clone it from: https://github.com/AIoT-MLSys-Lab/SVD-LLM"
        )

    print(f"  → Found SVD-LLM at: {svdllm_path}")

    if svdllm_path not in sys.path:
        sys.path.insert(0, svdllm_path)

    # Load the .pt file saved by SVD-LLM
    try:
        print("  → Loading .pt file...")
        pruned_dict = torch.load(svd_model_path, weights_only=False, map_location='cpu')
        tokenizer = pruned_dict['tokenizer']
        model = pruned_dict['model']
        print("  ✓ Successfully loaded SVD-LLM model from .pt file")

        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  → Model parameters: {total_params / 1e9:.2f}B")

    except Exception as e:
        raise ValueError(
            f"Could not load SVD-LLM model from {svd_model_path}.\n"
            f"Expected format: .pt file with dict containing 'model' and 'tokenizer' keys.\n"
            f"This file should be generated by SVDLLM.py step 2.\n"
            f"Error: {e}"
        )

    # Convert SVD layers to Matryoshka layers
    print("Converting to Matryoshka layers...")
    model = convert_to_matryoshka(model, r_max, r_min)

    return model.to(device), tokenizer


def convert_to_matryoshka(
    model: nn.Module,
    r_max: int,
    r_min: int
) -> nn.Module:
    """
    Convert SVD-LLM layers to MatryoshkaSVDLayers.

    This function finds SVD_LlamaAttention and SVD_LlamaMLP layers,
    wraps each U/V pair in a MatryoshkaSVDLayer with rank prediction.

    Args:
        model: Model with SVD_LlamaAttention/SVD_LlamaMLP layers
        r_max: Maximum rank for Matryoshka
        r_min: Minimum rank for Matryoshka

    Returns:
        Model with MatryoshkaSVDLayers
    """
    # SVD-LLM path should already be in sys.path from load_svdllm_model()
    try:
        from component.svd_llama import SVD_LlamaAttention, SVD_LlamaMLP
    except ImportError as e:
        print(f"Warning: Could not import SVD_LlamaAttention/SVD_LlamaMLP: {e}")
        print("Skipping Matryoshka conversion")
        return model

    conversion_count = 0

    # Iterate through model layers
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    else:
        print("Warning: Could not find model.model.layers")
        return model

    for layer_idx, layer in enumerate(layers):
        # Convert attention
        if hasattr(layer, 'self_attn') and isinstance(layer.self_attn, SVD_LlamaAttention):
            attn = layer.self_attn
            hidden_dim = attn.hidden_size

            # Get current rank from existing projections
            current_rank = attn.q_v_proj.out_features
            effective_r_max = min(r_max, current_rank)

            # Wrap each Q/K/V/O projection pair in MatryoshkaSVDLayer
            attn.q_matryoshka = MatryoshkaSVDLayer(
                u_proj=attn.q_u_proj,
                v_proj=attn.q_v_proj,
                r_max=effective_r_max,
                r_min=r_min,
                use_rank_predictor=True,
                gating_tau=0.1
            )

            attn.k_matryoshka = MatryoshkaSVDLayer(
                u_proj=attn.k_u_proj,
                v_proj=attn.k_v_proj,
                r_max=effective_r_max,
                r_min=r_min,
                use_rank_predictor=True,
                gating_tau=0.1
            )

            attn.v_matryoshka = MatryoshkaSVDLayer(
                u_proj=attn.v_u_proj,
                v_proj=attn.v_v_proj,
                r_max=effective_r_max,
                r_min=r_min,
                use_rank_predictor=True,
                gating_tau=0.1
            )

            attn.o_matryoshka = MatryoshkaSVDLayer(
                u_proj=attn.o_u_proj,
                v_proj=attn.o_v_proj,
                r_max=effective_r_max,
                r_min=r_min,
                use_rank_predictor=True,
                gating_tau=0.1
            )

            # Replace forward method
            attn.original_forward = attn.forward
            attn.forward = lambda *args, **kwargs: matryoshka_attention_forward(attn, *args, **kwargs)

            conversion_count += 4  # 4 projections

        # Convert MLP
        if hasattr(layer, 'mlp') and isinstance(layer.mlp, SVD_LlamaMLP):
            mlp = layer.mlp
            hidden_dim = mlp.gate_v_proj.in_features

            # Get current rank from existing projections
            current_rank = mlp.gate_v_proj.out_features
            effective_r_max = min(r_max, current_rank)

            # Wrap each gate/up/down projection pair in MatryoshkaSVDLayer
            mlp.gate_matryoshka = MatryoshkaSVDLayer(
                u_proj=mlp.gate_u_proj,
                v_proj=mlp.gate_v_proj,
                r_max=effective_r_max,
                r_min=r_min,
                use_rank_predictor=True,
                gating_tau=0.1
            )

            mlp.up_matryoshka = MatryoshkaSVDLayer(
                u_proj=mlp.up_u_proj,
                v_proj=mlp.up_v_proj,
                r_max=effective_r_max,
                r_min=r_min,
                use_rank_predictor=True,
                gating_tau=0.1
            )

            mlp.down_matryoshka = MatryoshkaSVDLayer(
                u_proj=mlp.down_u_proj,
                v_proj=mlp.down_v_proj,
                r_max=effective_r_max,
                r_min=r_min,
                use_rank_predictor=True,
                gating_tau=0.1
            )

            # Replace forward method
            mlp.original_forward = mlp.forward
            mlp.forward = lambda x: matryoshka_mlp_forward(mlp, x)

            conversion_count += 3  # 3 projections

    print(f"  ✓ Converted {conversion_count} projections to Matryoshka layers")
    return model


def matryoshka_attention_forward(attn, hidden_states, attention_mask=None, position_ids=None, **kwargs):
    """
    Matryoshka forward for attention module.

    Uses MatryoshkaSVDLayers for Q/K/V/O projections.
    """
    bsz, q_len, _ = hidden_states.size()

    # Use Matryoshka layers for projections
    query_states = attn.q_matryoshka(hidden_states)
    key_states = attn.k_matryoshka(hidden_states)
    value_states = attn.v_matryoshka(hidden_states)

    # Reshape for multi-head attention
    query_states = query_states.view(bsz, q_len, attn.num_heads, attn.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, attn.num_heads, attn.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, attn.num_heads, attn.head_dim).transpose(1, 2)

    # Apply rotary embeddings
    kv_seq_len = key_states.shape[-2]
    cos, sin = attn.rotary_emb(value_states, seq_len=kv_seq_len)

    # Import apply_rotary_pos_emb (path should already be in sys.path)
    from component.svd_llama import apply_rotary_pos_emb

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    # Attention computation
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / (attn.head_dim ** 0.5)

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)

    # Reshape and output projection
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, attn.hidden_size)

    # Use Matryoshka layer for output projection
    attn_output = attn.o_matryoshka(attn_output)

    return (attn_output,)


def matryoshka_mlp_forward(mlp, x):
    """
    Matryoshka forward for MLP module.

    Uses MatryoshkaSVDLayers for gate/up/down projections.
    """
    # Use Matryoshka layers instead of direct projections
    gate = mlp.gate_matryoshka(x)
    up = mlp.up_matryoshka(x)

    # Apply activation
    gate = mlp.act_fn(gate)

    # Down projection
    output = mlp.down_matryoshka(gate * up)

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

    model, tokenizer = load_svdllm_model(
        model_name=args.model,
        svd_model_path=args.svdllm_model,
        r_max=args.r_max,
        r_min=args.r_min
    )

    # Set pad token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Freeze U and V if requested
    if args.freeze_uv:
        print("\nFreezing U and V projections...")
        for name, param in model.named_parameters():
            if '_u_proj' in name or '_v_proj' in name:
                param.requires_grad = False
            elif 'rank_predictor' in name:
                param.requires_grad = True

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
