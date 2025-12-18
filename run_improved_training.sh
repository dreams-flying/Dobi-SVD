#!/bin/bash
#
# Quick start script for improved training strategy
#
# This script demonstrates the simplest approach to break the loss plateau:
# - Parameter group learning rates
# - Optional progressive rank training
#
# Expected: Loss should decrease from ~7.0 to ~4.2-4.5
#

# Configuration
MATRYOSHKA_MODEL_PATH="./matryoshka_model_checkpoint"  # Change this to your checkpoint path
BASE_MODEL="meta-llama/Llama-2-7b-hf"  # Change if using different base model
OUTPUT_DIR="./matryoshka_improved_output"

echo "========================================"
echo "Improved Matryoshka Training"
echo "========================================"
echo ""
echo "Strategy: Parameter Group Learning Rates + Progressive Rank"
echo ""
echo "This should break the loss plateau at 7.0"
echo "Expected final loss: 4.2-4.5"
echo ""
echo "========================================"
echo ""

# Check if checkpoint exists
if [ ! -d "$MATRYOSHKA_MODEL_PATH" ]; then
    echo "❌ Error: Matryoshka checkpoint not found at: $MATRYOSHKA_MODEL_PATH"
    echo ""
    echo "Please set MATRYOSHKA_MODEL_PATH to your checkpoint directory"
    echo "Example: MATRYOSHKA_MODEL_PATH=/path/to/checkpoint ./run_improved_training.sh"
    exit 1
fi

echo "✅ Found checkpoint: $MATRYOSHKA_MODEL_PATH"
echo ""

# Option 1: Parameter groups only (simpler, faster to test)
echo "Option 1: Parameter Groups Only (simpler)"
echo "Command:"
echo ""
echo "python train_with_improved_strategy.py \\"
echo "  --matryoshka_model_path $MATRYOSHKA_MODEL_PATH \\"
echo "  --base_model_name $BASE_MODEL \\"
echo "  --output_dir ${OUTPUT_DIR}_param_groups \\"
echo "  --num_train_epochs 5 \\"
echo "  --per_device_train_batch_size 4 \\"
echo "  --gradient_accumulation_steps 4 \\"
echo "  --rank_predictor_lr 1e-4 \\"
echo "  --uv_lr 1e-6 \\"
echo "  --other_lr 5e-5"
echo ""
echo "========================================"
echo ""

# Option 2: Parameter groups + progressive rank (better, recommended)
echo "Option 2: Parameter Groups + Progressive Rank (recommended)"
echo "Command:"
echo ""
echo "python train_with_improved_strategy.py \\"
echo "  --matryoshka_model_path $MATRYOSHKA_MODEL_PATH \\"
echo "  --base_model_name $BASE_MODEL \\"
echo "  --output_dir ${OUTPUT_DIR}_progressive \\"
echo "  --num_train_epochs 5 \\"
echo "  --per_device_train_batch_size 4 \\"
echo "  --gradient_accumulation_steps 4 \\"
echo "  --rank_predictor_lr 1e-4 \\"
echo "  --uv_lr 1e-6 \\"
echo "  --other_lr 5e-5 \\"
echo "  --use_progressive_rank \\"
echo "  --r_max 512 \\"
echo "  --r_min 256"
echo ""
echo "========================================"
echo ""

# Prompt user
read -p "Choose option (1 or 2): " option

if [ "$option" == "1" ]; then
    echo ""
    echo "Running Option 1: Parameter Groups Only"
    echo ""

    python train_with_improved_strategy.py \
      --matryoshka_model_path "$MATRYOSHKA_MODEL_PATH" \
      --base_model_name "$BASE_MODEL" \
      --output_dir "${OUTPUT_DIR}_param_groups" \
      --num_train_epochs 5 \
      --per_device_train_batch_size 4 \
      --gradient_accumulation_steps 4 \
      --rank_predictor_lr 1e-4 \
      --uv_lr 1e-6 \
      --other_lr 5e-5

elif [ "$option" == "2" ]; then
    echo ""
    echo "Running Option 2: Parameter Groups + Progressive Rank (Recommended)"
    echo ""

    python train_with_improved_strategy.py \
      --matryoshka_model_path "$MATRYOSHKA_MODEL_PATH" \
      --base_model_name "$BASE_MODEL" \
      --output_dir "${OUTPUT_DIR}_progressive" \
      --num_train_epochs 5 \
      --per_device_train_batch_size 4 \
      --gradient_accumulation_steps 4 \
      --rank_predictor_lr 1e-4 \
      --uv_lr 1e-6 \
      --other_lr 5e-5 \
      --use_progressive_rank \
      --r_max 512 \
      --r_min 256

else
    echo "❌ Invalid option. Please run again and choose 1 or 2."
    exit 1
fi

echo ""
echo "========================================"
echo "Training completed!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Check final loss (should be < 5.0)"
echo "2. If loss is still high, try knowledge distillation"
echo "3. See TRAINING_STRATEGIES_TO_BREAK_PLATEAU.md for details"
echo ""
