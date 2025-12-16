#!/usr/bin/env python3
"""
Diagnostic script to check why MatryoshkaSVDLayer is not found in the loaded model.

Usage:
    python diagnose_model_loading.py --checkpoint /path/to/checkpoint
"""

import argparse
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoConfig

from modules.matryoshka_svd_layer import MatryoshkaSVDLayer


def diagnose_checkpoint(checkpoint_path):
    """Diagnose why MatryoshkaSVDLayer is not being loaded."""
    print("="*80)
    print("MODEL LOADING DIAGNOSTIC")
    print("="*80)
    print(f"Checkpoint: {checkpoint_path}\n")

    checkpoint_path = Path(checkpoint_path)

    # Step 1: Check files
    print("[Step 1] Checking checkpoint files...")
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint path does not exist: {checkpoint_path}")
        return

    files = list(checkpoint_path.glob("*"))
    print(f"Files in checkpoint ({len(files)}):")
    for f in sorted(files)[:20]:  # Show first 20
        print(f"  {f.name}")

    # Step 2: Check config
    print("\n[Step 2] Checking config.json...")
    config_path = checkpoint_path / "config.json"
    if config_path.exists():
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)

        print(f"Model type: {config.get('model_type', 'unknown')}")
        print(f"Architecture: {config.get('architectures', 'unknown')}")

        # Check for custom modules
        if 'auto_map' in config:
            print(f"Custom modules (auto_map): {config['auto_map']}")
        else:
            print("⚠️  No auto_map found (custom modules may not load)")
    else:
        print("❌ config.json not found")

    # Step 3: Load model
    print("\n[Step 3] Loading model with AutoModelForCausalLM...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            trust_remote_code=True  # Allow custom code
        )
        print("✅ Model loaded")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Step 4: Check model structure
    print("\n[Step 4] Analyzing model structure...")

    # Count all module types
    module_types = {}
    for name, module in model.named_modules():
        module_type = type(module).__name__
        module_types[module_type] = module_types.get(module_type, 0) + 1

    print(f"Module types found ({len(module_types)} unique types):")
    for mod_type, count in sorted(module_types.items(), key=lambda x: -x[1])[:20]:
        marker = "🎯" if mod_type == "MatryoshkaSVDLayer" else "  "
        print(f"{marker} {mod_type:40s}: {count:4d}")

    # Step 5: Look for MatryoshkaSVDLayer specifically
    print("\n[Step 5] Searching for MatryoshkaSVDLayer instances...")
    matryoshka_layers = []
    for name, module in model.named_modules():
        if isinstance(module, MatryoshkaSVDLayer):
            matryoshka_layers.append(name)

    if matryoshka_layers:
        print(f"✅ Found {len(matryoshka_layers)} MatryoshkaSVDLayer instances:")
        for name in matryoshka_layers[:5]:
            print(f"  {name}")
    else:
        print("❌ NO MatryoshkaSVDLayer instances found!")
        print("\nPossible causes:")
        print("1. Model was saved without MatryoshkaSVDLayer (using HuggingFace Trainer)")
        print("2. Custom modules not registered in config.json (missing auto_map)")
        print("3. Model was saved with original Linear layers, not Matryoshka layers")

    # Step 6: Check for Linear layers that should be Matryoshka
    print("\n[Step 6] Checking attention/MLP layers...")
    attention_layers = []
    for name, module in model.named_modules():
        if any(x in name for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                                     'gate_proj', 'up_proj', 'down_proj']):
            module_type = type(module).__name__
            attention_layers.append((name, module_type))

    print(f"Found {len(attention_layers)} projection layers:")
    for name, mod_type in attention_layers[:10]:
        marker = "✅" if mod_type == "MatryoshkaSVDLayer" else "⚠️ "
        print(f"{marker} {name:50s}: {mod_type}")

    # Step 7: Recommendation
    print("\n" + "="*80)
    print("DIAGNOSIS SUMMARY")
    print("="*80)

    if matryoshka_layers:
        print("✅ Model has MatryoshkaSVDLayer - should work correctly")
    else:
        print("❌ Model does NOT have MatryoshkaSVDLayer")
        print("\nRECOMMENDED FIX:")
        print("The model was saved using HuggingFace's save_pretrained(), which")
        print("converts custom modules back to their base types (Linear).")
        print("\nYou need to:")
        print("1. Save the model using torch.save() to preserve custom classes, OR")
        print("2. Use a custom loading function that reconstructs MatryoshkaSVDLayer")
        print("\nSee: https://github.com/huggingface/transformers/issues/XXXX")


def main():
    parser = argparse.ArgumentParser(description='Diagnose model loading issues')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    args = parser.parse_args()

    diagnose_checkpoint(args.checkpoint)


if __name__ == '__main__':
    main()
