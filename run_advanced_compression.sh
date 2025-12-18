#!/bin/bash
#
# Quick start script for advanced compression (NO knowledge distillation)
#
# Recommended: Self-distillation + Two-stage + Layer-wise + Learned temperature
# Expected: Loss 7.0 → 3.7 (comparable to distillation, no extra memory!)
#

# Configuration
MATRYOSHKA_MODEL_PATH="./matryoshka_model_checkpoint"
BASE_MODEL="meta-llama/Llama-2-7b-hf"
OUTPUT_DIR="./matryoshka_advanced_output"

echo "========================================"
echo "Advanced Compression (NO Distillation)"
echo "========================================"
echo ""
echo "Benefits:"
echo "  ✅ No teacher model needed (1x memory)"
echo "  ✅ Loss 7.0 → 3.5-4.0"
echo "  ✅ Comparable to knowledge distillation"
echo ""
echo "========================================"
echo ""

# Check if checkpoint exists
if [ ! -d "$MATRYOSHKA_MODEL_PATH" ]; then
    echo "❌ Error: Matryoshka checkpoint not found at: $MATRYOSHKA_MODEL_PATH"
    echo ""
    echo "Please set MATRYOSHKA_MODEL_PATH to your checkpoint directory"
    echo "Example: MATRYOSHKA_MODEL_PATH=/path/to/checkpoint ./run_advanced_compression.sh"
    exit 1
fi

echo "✅ Found checkpoint: $MATRYOSHKA_MODEL_PATH"
echo ""

# Show options
echo "Available strategies:"
echo ""
echo "1. Self-Distillation Only (Simple, Fast)"
echo "   - Use full-rank as teacher"
echo "   - Expected loss: ~4.2"
echo "   - Time: 1.3x"
echo ""
echo "2. Two-Stage Training (Simple, No Overhead)"
echo "   - Stage 1: Train predictor"
echo "   - Stage 2: Joint training"
echo "   - Expected loss: ~4.7"
echo "   - Time: 1x"
echo ""
echo "3. Best Combination (Recommended) ⭐"
echo "   - Self-distillation + Two-stage + Layer-wise + Learned temp"
echo "   - Expected loss: ~3.7"
echo "   - Time: 1.3x"
echo "   - Memory: 1x (same as baseline)"
echo ""
echo "4. Ultra Mode (All Strategies)"
echo "   - Self-distill + Reconstruction + Two-stage + Layer-wise + Learned temp"
echo "   - Expected loss: ~3.5"
echo "   - Time: 1.5x"
echo ""
echo "========================================"
echo ""

read -p "Choose option (1-4): " option
echo ""

case $option in
    1)
        echo "Running Option 1: Self-Distillation Only"
        echo "Expected: Loss 7.0 → 4.2"
        echo ""

        python train_advanced_compression.py \
          --matryoshka_model_path "$MATRYOSHKA_MODEL_PATH" \
          --base_model_name "$BASE_MODEL" \
          --output_dir "${OUTPUT_DIR}_self_distill" \
          --num_train_epochs 5 \
          --per_device_train_batch_size 4 \
          --gradient_accumulation_steps 4 \
          --strategy self_distill \
          --self_distill_alpha 1.0 \
          --self_distill_beta 0.3
        ;;

    2)
        echo "Running Option 2: Two-Stage Training"
        echo "Expected: Loss 7.0 → 4.7"
        echo ""

        python train_advanced_compression.py \
          --matryoshka_model_path "$MATRYOSHKA_MODEL_PATH" \
          --base_model_name "$BASE_MODEL" \
          --output_dir "${OUTPUT_DIR}_two_stage" \
          --num_train_epochs 5 \
          --per_device_train_batch_size 4 \
          --gradient_accumulation_steps 4 \
          --strategy self_distill \
          --use_two_stage
        ;;

    3)
        echo "Running Option 3: Best Combination (Recommended) ⭐"
        echo "Expected: Loss 7.0 → 3.7"
        echo ""
        echo "This combines:"
        echo "  ✅ Self-distillation"
        echo "  ✅ Two-stage training"
        echo "  ✅ Layer-wise adaptive rank"
        echo "  ✅ Learned temperature scheduling"
        echo ""

        python train_advanced_compression.py \
          --matryoshka_model_path "$MATRYOSHKA_MODEL_PATH" \
          --base_model_name "$BASE_MODEL" \
          --output_dir "${OUTPUT_DIR}_best_combo" \
          --num_train_epochs 5 \
          --per_device_train_batch_size 4 \
          --gradient_accumulation_steps 4 \
          --strategy self_distill \
          --use_two_stage \
          --use_layerwise_rank \
          --use_learned_temp \
          --self_distill_alpha 1.0 \
          --self_distill_beta 0.3 \
          --initial_tau 5.0 \
          --final_tau 0.5 \
          --r_min 256 \
          --r_max 512
        ;;

    4)
        echo "Running Option 4: Ultra Mode (All Strategies)"
        echo "Expected: Loss 7.0 → 3.5"
        echo ""
        echo "This combines:"
        echo "  ✅ Self-distillation"
        echo "  ✅ Reconstruction loss"
        echo "  ✅ Two-stage training"
        echo "  ✅ Layer-wise adaptive rank"
        echo "  ✅ Learned temperature scheduling"
        echo ""

        python train_advanced_compression.py \
          --matryoshka_model_path "$MATRYOSHKA_MODEL_PATH" \
          --base_model_name "$BASE_MODEL" \
          --output_dir "${OUTPUT_DIR}_ultra" \
          --num_train_epochs 5 \
          --per_device_train_batch_size 4 \
          --gradient_accumulation_steps 4 \
          --strategy both \
          --use_two_stage \
          --use_layerwise_rank \
          --use_learned_temp \
          --self_distill_alpha 1.0 \
          --self_distill_beta 0.3 \
          --recon_weight 0.1 \
          --initial_tau 5.0 \
          --final_tau 0.5 \
          --r_min 256 \
          --r_max 512
        ;;

    *)
        echo "❌ Invalid option. Please run again and choose 1-4."
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "Training Completed!"
echo "========================================"
echo ""
echo "Check final loss:"
echo "  - If loss < 4.0: Excellent! ✅✅"
echo "  - If loss 4.0-5.0: Good! ✅"
echo "  - If loss > 5.0: Try option 4 (Ultra Mode)"
echo ""
echo "Compare with baseline:"
echo "  - Baseline (freeze_uv removed): loss ~7.0"
echo "  - Your result: check output above"
echo ""
echo "Next steps:"
echo "  1. Evaluate PPL on test set"
echo "  2. Compare with knowledge distillation if memory allows"
echo "  3. See ADVANCED_COMPRESSION_WITHOUT_DISTILLATION.md for details"
echo ""
echo "========================================"
