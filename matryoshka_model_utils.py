#!/usr/bin/env python3
"""
Utilities for saving and loading Matryoshka SVD models.

The standard HuggingFace save_pretrained() loses custom module structure.
These utilities preserve MatryoshkaSVDLayer information across save/load cycles.
"""

import torch
import torch.nn as nn
import json
from pathlib import Path
from typing import Dict, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from modules.matryoshka_svd_layer import MatryoshkaSVDLayer


def save_matryoshka_model(
    model: nn.Module,
    tokenizer,
    output_dir: str,
    safe_serialization: bool = True
):
    """
    Save Matryoshka SVD model with metadata to preserve custom layers.

    Args:
        model: Model with MatryoshkaSVDLayer instances
        tokenizer: Tokenizer to save
        output_dir: Directory to save model
        safe_serialization: Whether to use safetensors format
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Saving Matryoshka SVD Model")
    print(f"{'='*80}")

    # 1. Save the base model (this will save weights but lose custom structure)
    print("Saving model weights...")
    model.save_pretrained(output_dir, safe_serialization=safe_serialization)

    # 2. Save tokenizer
    print("Saving tokenizer...")
    tokenizer.save_pretrained(output_dir)

    # 3. IMPORTANT: Save Matryoshka layer metadata
    print("Saving Matryoshka layer metadata...")
    matryoshka_metadata = {
        'matryoshka_layers': [],
        'version': '1.0'
    }

    for name, module in model.named_modules():
        if isinstance(module, MatryoshkaSVDLayer):
            # Extract gumbel parameters if using dimension_wise predictor
            use_gumbel = False
            gumbel_tau = 1.0
            gumbel_hard = False

            if (module.use_rank_predictor and
                module.predictor_mode == 'dimension_wise' and
                hasattr(module.rank_predictor, 'use_gumbel')):
                use_gumbel = module.rank_predictor.use_gumbel
                gumbel_tau = module.rank_predictor.gumbel_tau
                gumbel_hard = module.rank_predictor.gumbel_hard

            layer_info = {
                'name': name,
                'in_features': module.in_features,
                'out_features': module.out_features,
                'r_max': module.r_max,
                'r_min': module.r_min,
                'use_rank_predictor': module.use_rank_predictor,
                'predictor_mode': module.predictor_mode,
                'gating_tau': module.gating_tau,
                'use_gumbel': use_gumbel,
                'gumbel_tau': gumbel_tau,
                'gumbel_hard': gumbel_hard,
                'hard_inference': module.hard_inference,
                'bias': module.bias is not None
            }
            matryoshka_metadata['matryoshka_layers'].append(layer_info)

    metadata_path = output_dir / 'matryoshka_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(matryoshka_metadata, f, indent=2)

    print(f"  Saved metadata for {len(matryoshka_metadata['matryoshka_layers'])} layers")
    print(f"  Metadata file: {metadata_path}")

    print(f"\n{'='*80}")
    print(f"Model saved successfully to: {output_dir}")
    print(f"{'='*80}")
    print(f"Files saved:")
    print(f"  - config.json")
    print(f"  - model.safetensors (or pytorch_model.bin)")
    print(f"  - matryoshka_metadata.json  ← CRITICAL for reconstruction")
    print(f"  - tokenizer files")


def load_matryoshka_model(
    checkpoint_path: str,
    base_model: Optional[str] = None,
    device: str = 'cuda',
    torch_dtype = torch.float32
):
    """
    Load Matryoshka SVD model and reconstruct custom layers.

    Args:
        checkpoint_path: Path to saved checkpoint
        base_model: Base model name (for tokenizer fallback)
        device: Device to load model on
        torch_dtype: Data type for model weights

    Returns:
        (model, tokenizer) tuple
    """
    checkpoint_path = Path(checkpoint_path)

    print(f"\n{'='*80}")
    print(f"Loading Matryoshka SVD Model")
    print(f"{'='*80}")
    print(f"Checkpoint: {checkpoint_path}")

    # 1. Check for metadata file
    metadata_path = checkpoint_path / 'matryoshka_metadata.json'
    if not metadata_path.exists():
        print(f"\n⚠️  WARNING: matryoshka_metadata.json not found!")
        print(f"This checkpoint may not have been saved with save_matryoshka_model().")
        print(f"Will attempt to load as standard model, but MatryoshkaSVDLayer will be lost.")

        # Load as standard model
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            device_map=None
        )

        # Load tokenizer with fallback
        try:
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        except:
            if base_model:
                tokenizer = AutoTokenizer.from_pretrained(base_model)
            else:
                config_path = checkpoint_path / 'config.json'
                with open(config_path, 'r') as f:
                    config = json.load(f)
                base_model = config.get('_name_or_path', 'gpt2')
                tokenizer = AutoTokenizer.from_pretrained(base_model)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model.to(device)
        model.eval()

        return model, tokenizer

    # 2. Load metadata
    print(f"\nLoading Matryoshka metadata...")
    import json
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    matryoshka_layers_info = metadata['matryoshka_layers']
    print(f"  Found metadata for {len(matryoshka_layers_info)} Matryoshka layers")

    # 3. Load the checkpoint state dict BEFORE loading the model
    # This preserves all MatryoshkaSVDLayer weights that would otherwise be dropped
    print(f"\nLoading checkpoint state dict...")

    # Try different weight file formats
    if (checkpoint_path / 'pytorch_model.bin').exists():
        checkpoint_state = torch.load(
            checkpoint_path / 'pytorch_model.bin',
            map_location='cpu'
        )
        print(f"  Loaded from pytorch_model.bin")
    elif (checkpoint_path / 'model.safetensors').exists():
        from safetensors.torch import load_file
        checkpoint_state = load_file(checkpoint_path / 'model.safetensors')
        print(f"  Loaded from model.safetensors")
    else:
        # Try sharded safetensors
        index_file = checkpoint_path / 'model.safetensors.index.json'
        if index_file.exists():
            from safetensors.torch import load_file
            with open(index_file, 'r') as f:
                index = json.load(f)
            weight_map = index['weight_map']
            checkpoint_state = {}
            shard_files = set(weight_map.values())
            for shard_file in shard_files:
                shard_path = checkpoint_path / shard_file
                shard_state = load_file(shard_path)
                checkpoint_state.update(shard_state)
            print(f"  Loaded from {len(shard_files)} safetensors shards")
        else:
            raise FileNotFoundError(f"No weight files found in {checkpoint_path}")

    print(f"  Total tensors in checkpoint: {len(checkpoint_state)}")

    # 4. Load base model (this will have Linear layers, but we'll replace them)
    print(f"\nLoading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map=None
    )
    print(f"  Base model loaded (MatryoshkaSVDLayer weights will be added next)")

    # 5. Replace Linear layers with MatryoshkaSVDLayer
    print(f"\nReconstructing MatryoshkaSVDLayer instances...")
    layers_reconstructed = 0

    for layer_info in matryoshka_layers_info:
        name = layer_info['name']

        # Navigate to the parent module
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            if part.isdigit():
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part)

        # Create MatryoshkaSVDLayer
        matryoshka_layer = MatryoshkaSVDLayer(
            in_features=layer_info['in_features'],
            out_features=layer_info['out_features'],
            r_max=layer_info['r_max'],
            r_min=layer_info['r_min'],
            use_rank_predictor=layer_info['use_rank_predictor'],
            predictor_mode=layer_info['predictor_mode'],
            gating_tau=layer_info['gating_tau'],
            use_gumbel=layer_info.get('use_gumbel', False),
            gumbel_tau=layer_info.get('gumbel_tau', 1.0),
            gumbel_hard=layer_info.get('gumbel_hard', False),
            hard_inference=layer_info.get('hard_inference', True),
            bias=layer_info['bias']
        )

        # Extract state dict for this layer from the full checkpoint state
        prefix = name + '.'
        layer_state = {}
        for key, value in checkpoint_state.items():
            if key.startswith(prefix):
                # Remove prefix to get local key
                local_key = key[len(prefix):]
                layer_state[local_key] = value

        # Load the state into MatryoshkaSVDLayer
        if layer_state:
            missing, unexpected = matryoshka_layer.load_state_dict(layer_state, strict=False)
            if missing:
                print(f"  ⚠️  {name}: Missing keys: {missing[:3]}...")  # Show first 3
        else:
            print(f"  ⚠️  {name}: No weights found in checkpoint!")

        # Replace the Linear layer with MatryoshkaSVDLayer
        setattr(parent, parts[-1], matryoshka_layer)
        layers_reconstructed += 1

        print(f"  ✅ {name}: Linear → MatryoshkaSVDLayer (r={layer_info['r_min']}-{layer_info['r_max']})")

    print(f"\nReconstructed {layers_reconstructed}/{len(matryoshka_layers_info)} layers")

    # CRITICAL: Patch forward methods to use MatryoshkaSVDLayer
    print(f"\nPatching forward methods to use Matryoshka layers...")
    forward_patched = 0

    # Import forward functions from training script
    try:
        import sys
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)

        # IMPORTANT: Add SVD-LLM to path (needed for component.svd_llama imports in forward functions)
        svdllm_paths = [
            '/home/user/SVD-LLM',
            '/root/SVD-LLM',
            os.path.expanduser('~/SVD-LLM'),
            '/data1/lichangqun/SVD-LLM'
        ]
        svdllm_found = False
        for svdllm_path in svdllm_paths:
            if os.path.exists(svdllm_path) and os.path.exists(os.path.join(svdllm_path, 'component')):
                if svdllm_path not in sys.path:
                    sys.path.insert(0, svdllm_path)
                print(f"  Found SVD-LLM at: {svdllm_path}")
                svdllm_found = True
                break

        if not svdllm_found:
            print(f"  ⚠️  Warning: SVD-LLM not found in common paths")
            print(f"     Forward methods may fail if they need component.svd_llama")

        from train_matryoshka_from_svdllm import (
            matryoshka_attention_forward,
            matryoshka_mlp_forward
        )

        # Patch attention and MLP forward methods
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            for layer_idx, layer in enumerate(model.model.layers):
                # Patch attention forward
                if hasattr(layer, 'self_attn'):
                    attn = layer.self_attn
                    # Check if this attention has matryoshka layers
                    if (hasattr(attn, 'q_matryoshka') and
                        hasattr(attn, 'k_matryoshka') and
                        hasattr(attn, 'v_matryoshka') and
                        hasattr(attn, 'o_matryoshka')):
                        # Patch forward method
                        attn.original_forward = attn.forward
                        attn.forward = lambda *args, _attn=attn, **kwargs: matryoshka_attention_forward(_attn, *args, **kwargs)
                        forward_patched += 1

                # Patch MLP forward
                if hasattr(layer, 'mlp'):
                    mlp = layer.mlp
                    # Check if this MLP has matryoshka layers
                    if (hasattr(mlp, 'gate_matryoshka') and
                        hasattr(mlp, 'up_matryoshka') and
                        hasattr(mlp, 'down_matryoshka')):
                        # Patch forward method
                        mlp.original_forward = mlp.forward
                        mlp.forward = lambda x, _mlp=mlp: matryoshka_mlp_forward(_mlp, x)
                        forward_patched += 1

        print(f"  ✅ Patched {forward_patched} attention/MLP forward methods")

    except Exception as e:
        print(f"  ⚠️  Warning: Could not patch forward methods: {e}")
        print(f"     The model may not work correctly!")
        import traceback
        traceback.print_exc()

    # 6. Load tokenizer
    print(f"\nLoading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        print(f"  Loaded from checkpoint")
    except:
        if base_model is None:
            config_path = checkpoint_path / 'config.json'
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                base_model = config.get('_name_or_path', None)

        if base_model:
            tokenizer = AutoTokenizer.from_pretrained(base_model)
            print(f"  Loaded from base model: {base_model}")
        else:
            raise ValueError("Cannot load tokenizer. Please provide --base_model")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 7. Move to device
    model.to(device)
    model.eval()

    # 8. Verify reconstruction
    print(f"\n{'='*80}")
    print(f"Model Loaded Successfully")
    print(f"{'='*80}")

    matryoshka_count = sum(
        1 for m in model.modules() if isinstance(m, MatryoshkaSVDLayer)
    )

    print(f"MatryoshkaSVDLayer instances: {matryoshka_count}")

    if matryoshka_count == 0:
        print(f"⚠️  WARNING: No MatryoshkaSVDLayer found after reconstruction!")
    else:
        print(f"✅ Successfully reconstructed Matryoshka model")

        # Show first layer info
        for module in model.modules():
            if isinstance(module, MatryoshkaSVDLayer):
                print(f"\nMatryoshka Configuration:")
                print(f"  Rank range: [{module.r_min}, {module.r_max}]")
                print(f"  Predictor mode: {module.predictor_mode}")
                print(f"  Hard inference: {module.hard_inference}")
                break

    return model, tokenizer


def get_matryoshka_layer_names(model: nn.Module) -> List[str]:
    """Get names of all MatryoshkaSVDLayer instances in model."""
    return [
        name for name, module in model.named_modules()
        if isinstance(module, MatryoshkaSVDLayer)
    ]


def verify_matryoshka_structure(checkpoint_path: str) -> Dict:
    """
    Verify if a checkpoint has Matryoshka structure metadata.

    Returns:
        Dictionary with verification results
    """
    checkpoint_path = Path(checkpoint_path)

    result = {
        'has_metadata': False,
        'num_layers': 0,
        'layer_names': [],
        'config_exists': False,
        'weights_exist': False
    }

    # Check metadata
    metadata_path = checkpoint_path / 'matryoshka_metadata.json'
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        result['has_metadata'] = True
        result['num_layers'] = len(metadata['matryoshka_layers'])
        result['layer_names'] = [l['name'] for l in metadata['matryoshka_layers']]

    # Check config
    result['config_exists'] = (checkpoint_path / 'config.json').exists()

    # Check weights
    result['weights_exist'] = (
        (checkpoint_path / 'pytorch_model.bin').exists() or
        (checkpoint_path / 'model.safetensors').exists() or
        any(checkpoint_path.glob('model-*.safetensors'))
    )

    return result
