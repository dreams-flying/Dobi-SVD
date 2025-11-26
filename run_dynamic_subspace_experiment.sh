#!/bin/bash

# Dynamic Subspace Routing Experiment Script
# This script runs experiments with different configurations

# Set environment variables
export CUDA_VISIBLE_DEVICES=0

# Base configuration
MODEL_ID="facebook/opt-125m"
TARGET_RATIO=0.5
SEQ_LEN=2048
N_TRAIN_SAMPLES=256
N_EVAL_SAMPLES=128
N_EPOCHS=5
DATASET="wikitext"

# Create results directory
mkdir -p results/training_output
mkdir -p results/data_cache
mkdir -p results/datasets

echo "=================================================="
echo "Dynamic Subspace Routing Experiments"
echo "=================================================="

# Experiment 1: 3 subspaces, norm-based routing (baseline)
echo ""
echo "[Experiment 1] 3 subspaces, norm-based routing"
python svd_trainer_dynamic.py \
    --model_id ${MODEL_ID} \
    --target_ratio ${TARGET_RATIO} \
    --seq_len ${SEQ_LEN} \
    --n_train_samples ${N_TRAIN_SAMPLES} \
    --n_eval_samples ${N_EVAL_SAMPLES} \
    --n_train_epochs ${N_EPOCHS} \
    --training_dataset ${DATASET} \
    --n_subspaces 3 \
    --routing_strategy norm \
    --lambda_balance 0.01 \
    --gamma_multipliers 0.5 1.0 1.5 \
    --path_head_folder ./results \
    --path_head_folder_output ./results

# Experiment 2: 3 subspaces, learnable routing
echo ""
echo "[Experiment 2] 3 subspaces, learnable routing"
python svd_trainer_dynamic.py \
    --model_id ${MODEL_ID} \
    --target_ratio ${TARGET_RATIO} \
    --seq_len ${SEQ_LEN} \
    --n_train_samples ${N_TRAIN_SAMPLES} \
    --n_eval_samples ${N_EVAL_SAMPLES} \
    --n_train_epochs ${N_EPOCHS} \
    --training_dataset ${DATASET} \
    --n_subspaces 3 \
    --routing_strategy learned \
    --learnable_thresholds \
    --lambda_balance 0.01 \
    --gamma_multipliers 0.5 1.0 1.5 \
    --path_head_folder ./results \
    --path_head_folder_output ./results

# Experiment 3: 5 subspaces, norm-based routing
echo ""
echo "[Experiment 3] 5 subspaces, norm-based routing"
python svd_trainer_dynamic.py \
    --model_id ${MODEL_ID} \
    --target_ratio ${TARGET_RATIO} \
    --seq_len ${SEQ_LEN} \
    --n_train_samples ${N_TRAIN_SAMPLES} \
    --n_eval_samples ${N_EVAL_SAMPLES} \
    --n_train_epochs ${N_EPOCHS} \
    --training_dataset ${DATASET} \
    --n_subspaces 5 \
    --routing_strategy norm \
    --lambda_balance 0.01 \
    --gamma_multipliers 0.3 0.6 1.0 1.4 1.8 \
    --path_head_folder ./results \
    --path_head_folder_output ./results

# Experiment 4: Soft routing (differentiable)
echo ""
echo "[Experiment 4] 3 subspaces, soft routing"
python svd_trainer_dynamic.py \
    --model_id ${MODEL_ID} \
    --target_ratio ${TARGET_RATIO} \
    --seq_len ${SEQ_LEN} \
    --n_train_samples ${N_TRAIN_SAMPLES} \
    --n_eval_samples ${N_EVAL_SAMPLES} \
    --n_train_epochs ${N_EPOCHS} \
    --training_dataset ${DATASET} \
    --n_subspaces 3 \
    --routing_strategy norm \
    --use_soft_routing \
    --routing_temperature 0.5 \
    --lambda_balance 0.01 \
    --gamma_multipliers 0.5 1.0 1.5 \
    --path_head_folder ./results \
    --path_head_folder_output ./results

echo ""
echo "=================================================="
echo "All experiments completed!"
echo "Results saved in: ./results/training_output/"
echo "=================================================="
