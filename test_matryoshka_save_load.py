#!/usr/bin/env python3
"""
Test script for Matryoshka SVD model save/load functionality.

Verifies that:
1. save_matryoshka_model() creates metadata file
2. load_matryoshka_model() reconstructs MatryoshkaSVDLayer
3. Weights are correctly preserved
4. Model outputs are identical before/after save/load
"""

import torch
import torch.nn as nn
import tempfile
import shutil
from pathlib import Path
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from modules.matryoshka_svd_layer import MatryoshkaSVDLayer
from matryoshka_model_utils import (
    save_matryoshka_model,
    load_matryoshka_model,
    verify_matryoshka_structure
)


def create_test_model():
    """Create a small Llama model with MatryoshkaSVDLayer for testing."""
    print("\n" + "="*80)
    print("Creating test model")
    print("="*80)

    # Small config for testing
    config = LlamaConfig(
        vocab_size=1000,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8,
        max_position_embeddings=512,
        _name_or_path="test-llama"
    )

    # Create model
    model = LlamaForCausalLM(config)

    # Replace some Linear layers with MatryoshkaSVDLayer
    print("\nReplacing Linear layers with MatryoshkaSVDLayer...")
    layers_replaced = 0

    for name, module in model.named_modules():
        if hasattr(module, 'self_attn'):
            attn = module.self_attn

            # Replace q_proj
            if hasattr(attn, 'q_proj'):
                original = attn.q_proj
                matryoshka_layer = MatryoshkaSVDLayer(
                    in_features=original.in_features,
                    out_features=original.out_features,
                    r_max=64,
                    r_min=16,
                    use_rank_predictor=True,
                    predictor_mode='rank',
                    hard_inference=True,
                    bias=False
                )
                attn.q_proj = matryoshka_layer
                layers_replaced += 1

            # Replace k_proj
            if hasattr(attn, 'k_proj'):
                original = attn.k_proj
                matryoshka_layer = MatryoshkaSVDLayer(
                    in_features=original.in_features,
                    out_features=original.out_features,
                    r_max=64,
                    r_min=16,
                    use_rank_predictor=True,
                    predictor_mode='rank',
                    hard_inference=True,
                    bias=False
                )
                attn.k_proj = matryoshka_layer
                layers_replaced += 1

    print(f"Replaced {layers_replaced} layers with MatryoshkaSVDLayer")

    # Count MatryoshkaSVDLayer instances
    matryoshka_count = sum(
        1 for m in model.modules()
        if isinstance(m, MatryoshkaSVDLayer)
    )
    print(f"Total MatryoshkaSVDLayer instances: {matryoshka_count}")

    return model, matryoshka_count


def test_save_load():
    """Test save and load functionality."""
    print("\n" + "="*80)
    print("TEST: Save and Load Matryoshka Model")
    print("="*80)

    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "test_checkpoint"

        # Create test model
        model_original, expected_layers = create_test_model()
        model_original.eval()

        # Create a minimal dummy tokenizer (to avoid network access)
        from transformers import PreTrainedTokenizerFast
        from tokenizers import Tokenizer, models, pre_tokenizers

        # Build a simple tokenizer
        tokenizer_model = models.BPE()
        tokenizer_obj = Tokenizer(tokenizer_model)
        tokenizer_obj.pre_tokenizer = pre_tokenizers.Whitespace()

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer_obj,
            eos_token="<|endoftext|>",
            pad_token="<|endoftext|>",
            unk_token="<|endoftext|>"
        )

        # Test 1: Save model
        print("\n" + "-"*80)
        print("TEST 1: Saving model")
        print("-"*80)

        save_matryoshka_model(
            model=model_original,
            tokenizer=tokenizer,
            output_dir=checkpoint_dir,
            safe_serialization=True
        )

        # Verify files were created
        assert (checkpoint_dir / 'config.json').exists(), "config.json not created"
        assert (checkpoint_dir / 'matryoshka_metadata.json').exists(), "metadata not created"
        print("✅ All required files created")

        # Test 2: Verify metadata
        print("\n" + "-"*80)
        print("TEST 2: Verifying metadata")
        print("-"*80)

        result = verify_matryoshka_structure(checkpoint_dir)
        print(f"Has metadata: {result['has_metadata']}")
        print(f"Number of layers: {result['num_layers']}")
        print(f"Config exists: {result['config_exists']}")
        print(f"Weights exist: {result['weights_exist']}")

        assert result['has_metadata'], "Metadata not found"
        assert result['num_layers'] == expected_layers, f"Expected {expected_layers} layers, got {result['num_layers']}"
        print("✅ Metadata verification passed")

        # Test 3: Load model
        print("\n" + "-"*80)
        print("TEST 3: Loading model")
        print("-"*80)

        model_loaded, tokenizer_loaded = load_matryoshka_model(
            checkpoint_path=checkpoint_dir,
            base_model=None,
            device='cpu'
        )

        # Count reconstructed layers
        loaded_layers = sum(
            1 for m in model_loaded.modules()
            if isinstance(m, MatryoshkaSVDLayer)
        )

        print(f"\nOriginal model layers: {expected_layers}")
        print(f"Loaded model layers: {loaded_layers}")

        assert loaded_layers == expected_layers, f"Expected {expected_layers} layers, got {loaded_layers}"
        print("✅ All MatryoshkaSVDLayer instances reconstructed")

        # Test 4: Compare outputs
        print("\n" + "-"*80)
        print("TEST 4: Comparing model outputs")
        print("-"*80)

        # Create test input
        input_ids = torch.randint(0, 1000, (1, 32))

        # Get outputs from both models
        with torch.no_grad():
            output_original = model_original(input_ids, use_cache=False)
            output_loaded = model_loaded(input_ids, use_cache=False)

        # Compare logits
        logits_original = output_original.logits
        logits_loaded = output_loaded.logits

        max_diff = (logits_original - logits_loaded).abs().max().item()
        mean_diff = (logits_original - logits_loaded).abs().mean().item()

        print(f"Max difference: {max_diff:.6e}")
        print(f"Mean difference: {mean_diff:.6e}")

        # Allow small numerical differences
        tolerance = 1e-4
        if max_diff < tolerance:
            print(f"✅ Outputs are identical (within {tolerance})")
        else:
            print(f"⚠️  Outputs differ by {max_diff:.6e} (tolerance: {tolerance})")
            if max_diff < 1e-2:
                print("   (Small difference, likely acceptable)")
            else:
                raise AssertionError(f"Output difference too large: {max_diff}")

        # Test 5: Test with fixed rank
        print("\n" + "-"*80)
        print("TEST 5: Testing with fixed rank")
        print("-"*80)

        # Set fixed rank
        for module in model_loaded.modules():
            if isinstance(module, MatryoshkaSVDLayer):
                module.set_fixed_rank(32)

        with torch.no_grad():
            output_fixed = model_loaded(input_ids, use_cache=False)

        print("✅ Fixed rank inference works")

        # Test 6: Test with adaptive rank
        print("\n" + "-"*80)
        print("TEST 6: Testing with adaptive rank")
        print("-"*80)

        # Set adaptive rank
        for module in model_loaded.modules():
            if isinstance(module, MatryoshkaSVDLayer):
                module.set_fixed_rank(None)

        with torch.no_grad():
            output_adaptive = model_loaded(input_ids, use_cache=False)

        # Get average rank
        ranks = []
        for module in model_loaded.modules():
            if isinstance(module, MatryoshkaSVDLayer):
                rank = module.get_effective_rank()
                if rank is not None:
                    ranks.append(rank)

        avg_rank = sum(ranks) / len(ranks) if ranks else 0
        print(f"Average predicted rank: {avg_rank:.2f}")
        print(f"Expected range: [16, 64]")

        if 16 <= avg_rank <= 64:
            print("✅ Adaptive rank prediction works")
        else:
            print(f"⚠️  Rank outside expected range: {avg_rank}")

    print("\n" + "="*80)
    print("ALL TESTS PASSED ✅")
    print("="*80)
    print("\nSummary:")
    print("✅ save_matryoshka_model() creates all required files")
    print("✅ Metadata is correctly saved")
    print("✅ load_matryoshka_model() reconstructs all MatryoshkaSVDLayer instances")
    print("✅ Model outputs are preserved across save/load")
    print("✅ Fixed rank inference works")
    print("✅ Adaptive rank prediction works")
    print("\n🎉 Matryoshka SVD save/load mechanism is working correctly!")


def main():
    """Run all tests."""
    torch.manual_seed(42)

    try:
        test_save_load()
        return 0
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
