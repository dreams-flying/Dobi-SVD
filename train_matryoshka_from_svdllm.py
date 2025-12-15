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
from datasets import load_dataset, Dataset
import random
import os
from typing import Dict, List, Optional
from pathlib import Path

from modules.matryoshka_svd_layer import MatryoshkaSVDLayer, RankPredictor
from utils.datautils import prepare_train_loaders


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


def create_matryoshka_layer_from_svd(
    u_proj: nn.Linear,
    v_proj: nn.Linear,
    r_max: int,
    r_min: int,
    gating_tau: float = 0.1
) -> MatryoshkaSVDLayer:
    """
    Create MatryoshkaSVDLayer and copy weights from existing SVD projections.

    Args:
        u_proj: Existing U projection from SVD-LLM
        v_proj: Existing V projection from SVD-LLM
        r_max: Maximum rank for Matryoshka
        r_min: Minimum rank
        gating_tau: Temperature for soft gating

    Returns:
        MatryoshkaSVDLayer with copied weights
    """
    # Get dimensions from existing projections
    in_features = v_proj.in_features
    out_features = u_proj.out_features
    current_rank = v_proj.out_features

    # Effective r_max cannot exceed current rank
    effective_r_max = min(r_max, current_rank)

    # Create Matryoshka layer
    matryoshka = MatryoshkaSVDLayer(
        in_features=in_features,
        out_features=out_features,
        r_max=effective_r_max,
        r_min=r_min,
        use_rank_predictor=True,
        gating_tau=gating_tau,
        bias=False
    )

    # Copy weights from SVD-LLM projections
    # V projection: [in_features, r_max]
    matryoshka.v_proj.weight.data = v_proj.weight.data[:effective_r_max, :].clone()

    # U projection: [out_features, r_max]
    matryoshka.u_proj.weight.data = u_proj.weight.data[:, :effective_r_max].clone()

    return matryoshka


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

            # Create Matryoshka layers from SVD projections
            attn.q_matryoshka = create_matryoshka_layer_from_svd(
                attn.q_u_proj, attn.q_v_proj, r_max, r_min
            )
            attn.k_matryoshka = create_matryoshka_layer_from_svd(
                attn.k_u_proj, attn.k_v_proj, r_max, r_min
            )
            attn.v_matryoshka = create_matryoshka_layer_from_svd(
                attn.v_u_proj, attn.v_v_proj, r_max, r_min
            )
            attn.o_matryoshka = create_matryoshka_layer_from_svd(
                attn.o_u_proj, attn.o_v_proj, r_max, r_min
            )

            # Replace forward method
            attn.original_forward = attn.forward
            attn.forward = lambda *args, **kwargs: matryoshka_attention_forward(attn, *args, **kwargs)

            conversion_count += 4  # 4 projections

        # Convert MLP
        if hasattr(layer, 'mlp') and isinstance(layer.mlp, SVD_LlamaMLP):
            mlp = layer.mlp

            # Create Matryoshka layers from SVD projections
            mlp.gate_matryoshka = create_matryoshka_layer_from_svd(
                mlp.gate_u_proj, mlp.gate_v_proj, r_max, r_min
            )
            mlp.up_matryoshka = create_matryoshka_layer_from_svd(
                mlp.up_u_proj, mlp.up_v_proj, r_max, r_min
            )
            mlp.down_matryoshka = create_matryoshka_layer_from_svd(
                mlp.down_u_proj, mlp.down_v_proj, r_max, r_min
            )

            # Replace forward method
            mlp.original_forward = mlp.forward
            mlp.forward = lambda x: matryoshka_mlp_forward(mlp, x)

            conversion_count += 3  # 3 projections

    print(f"  ✓ Converted {conversion_count} projections to Matryoshka layers")
    return model


def matryoshka_attention_forward(attn, hidden_states, attention_mask=None, position_ids=None,
                                 past_key_value=None, output_attentions=False, use_cache=False, **kwargs):
    """
    Matryoshka forward for attention module.

    Uses MatryoshkaSVDLayers for Q/K/V/O projections.

    Returns:
        Tuple of (attn_output, attn_weights, past_key_value)
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

    # Handle past_key_value for caching
    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    # Apply rotary embeddings
    cos, sin = attn.rotary_emb(value_states, seq_len=kv_seq_len)

    # Import apply_rotary_pos_emb (path should already be in sys.path)
    from component.svd_llama import apply_rotary_pos_emb

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    # Concatenate with past_key_value if provided
    if past_key_value is not None:
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    # Update past_key_value for next iteration
    past_key_value = (key_states, value_states) if use_cache else None

    # Attention computation
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / (attn.head_dim ** 0.5)

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
        attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device))

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)

    # Reshape and output projection
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, attn.hidden_size)

    # Use Matryoshka layer for output projection
    attn_output = attn.o_matryoshka(attn_output)

    # Return None for attn_weights if not requested
    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


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

    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='wikitext2',
                       help='Dataset name: wikitext2, c4, ptb')
    parser.add_argument('--seq_len', type=int, default=2048,
                       help='Sequence length for training')
    parser.add_argument('--n_train_samples', type=int, default=256,
                       help='Number of training samples')
    parser.add_argument('--n_eval_samples', type=int, default=128,
                       help='Number of evaluation samples')
    parser.add_argument('--data_cache_dir', type=str, default='./data_cache',
                       help='Directory for cached tokenized data')
    parser.add_argument('--dataset_cache_dir', type=str, default='./dataset_cache',
                       help='Directory for cached raw datasets')
    parser.add_argument('--SAVE', action='store_true',
                       help='Save processed datasets to cache')
    parser.add_argument('--RECREATE', action='store_true',
                       help='Recreate datasets even if cache exists')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed for dataset sampling')

    # Training arguments
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

    # Create cache directories
    data_cache_dir = Path(args.data_cache_dir)
    dataset_cache_dir = Path(args.dataset_cache_dir)
    data_cache_dir.mkdir(parents=True, exist_ok=True)
    dataset_cache_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets using prepare_train_loaders
    tokenized_traindata, tokenized_valdata = prepare_train_loaders(
        tokenizer=tokenizer,
        DATASET_NAME=args.dataset,
        data_cache_dir=data_cache_dir,
        dataset_cache_dir=dataset_cache_dir,
        args=args
    )

    # Convert to HuggingFace Dataset format
    train_dataset = Dataset.from_list(tokenized_traindata)
    eval_dataset = Dataset.from_list(tokenized_valdata)

    print(f"  ✓ Train samples: {len(train_dataset)}")
    print(f"  ✓ Eval samples: {len(eval_dataset)}")

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
        gradient_checkpointing=False,
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
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
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
