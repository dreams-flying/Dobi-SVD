#!/usr/bin/env python3
"""
Diagnostic script to check if MatryoshkaSVDLayer weights are being loaded correctly.
"""

import torch
from pathlib import Path
from safetensors.torch import load_file
import json

checkpoint_path = Path("matryoshka_output0/final")

print("="*80)
print("Weight Loading Diagnostic")
print("="*80)

# 1. Check metadata
metadata_path = checkpoint_path / "matryoshka_metadata.json"
if not metadata_path.exists():
    print(f"❌ No matryoshka_metadata.json found!")
    exit(1)

with open(metadata_path) as f:
    metadata = json.load(f)

print(f"\n✅ Found metadata with {len(metadata['matryoshka_layers'])} layers")

# 2. Load checkpoint state
print(f"\nLoading checkpoint weights...")
checkpoint_state = {}

# Try safetensors
safetensor_files = list(checkpoint_path.glob("*.safetensors"))
if safetensor_files:
    for shard in safetensor_files:
        checkpoint_state.update(load_file(shard))
    print(f"  Loaded from {len(safetensor_files)} safetensors files")
else:
    print(f"  ❌ No safetensors files found")

print(f"  Total keys in checkpoint: {len(checkpoint_state)}")

# 3. Check if matryoshka weights exist
print(f"\n{'='*80}")
print("Checking Matryoshka Layer Weights")
print("="*80)

matryoshka_layers_info = metadata['matryoshka_layers']
sample_layers = matryoshka_layers_info[:5]  # Check first 5

for layer_info in sample_layers:
    name = layer_info['name']
    print(f"\n{name}:")

    # Expected keys for MatryoshkaSVDLayer
    expected_keys = [
        f"{name}.v_proj.weight",
        f"{name}.u_proj.weight",
    ]

    if layer_info['use_rank_predictor']:
        expected_keys.extend([
            f"{name}.rank_predictor.fc.weight",
            f"{name}.rank_predictor.fc.bias",
        ])

    # Check which keys exist
    found = []
    missing = []

    for key in expected_keys:
        if key in checkpoint_state:
            found.append(key)
            # Show shape
            print(f"  ✅ {key}: {checkpoint_state[key].shape}")
        else:
            missing.append(key)

    if missing:
        print(f"  ❌ Missing keys: {missing}")

        # Check what keys DO exist for this layer
        layer_keys = [k for k in checkpoint_state.keys() if k.startswith(name)]
        if layer_keys:
            print(f"  📋 Actually found: {layer_keys[:5]}")  # Show first 5
        else:
            print(f"  ⚠️  NO keys found for this layer at all!")

# 4. Check if standard Linear weights exist instead
print(f"\n{'='*80}")
print("Checking for Standard Linear Weights (should NOT exist)")
print("="*80)

# Check if we have standard Linear weights (q_proj, k_proj, etc.) instead of matryoshka
standard_keys = [k for k in checkpoint_state.keys() if any(x in k for x in ['q_proj.weight', 'k_proj.weight', 'v_proj.weight', 'o_proj.weight'])]

if standard_keys:
    print(f"⚠️  Found {len(standard_keys)} standard Linear layer weights!")
    print(f"  Examples: {standard_keys[:5]}")
    print(f"\n  This suggests the checkpoint was saved WITHOUT MatryoshkaSVDLayer!")
else:
    print(f"✅ No standard Linear weights found (good - means we have MatryoshkaSVDLayer)")

# 5. Summary
print(f"\n{'='*80}")
print("Summary")
print("="*80)

matryoshka_weight_count = sum(1 for k in checkpoint_state.keys() if 'matryoshka' in k or ('v_proj' in k and 'u_proj' in k.replace('v_proj', '')))
print(f"Matryoshka-related weights: ~{matryoshka_weight_count}")
print(f"Standard Linear weights: {len(standard_keys)}")

if len(standard_keys) > 0:
    print(f"\n❌ PROBLEM: Checkpoint contains standard Linear weights, not MatryoshkaSVDLayer!")
    print(f"   This means the model was saved with save_pretrained() instead of save_matryoshka_model()")
    print(f"   You need to retrain or use a checkpoint that was saved with save_matryoshka_model()")
else:
    print(f"\n✅ Checkpoint appears to have MatryoshkaSVDLayer weights")
