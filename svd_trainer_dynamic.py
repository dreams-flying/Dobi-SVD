"""
Dynamic Subspace Routing SVD Trainer

This is an extended version of svd_trainer.py that supports MultiSubspaceSVDLayer
with dynamic token-wise routing across multiple subspaces.

Key differences from original trainer:
1. Uses MultiSubspaceSVDLayer instead of SVDTransformLayer
2. Initializes multiple gamma values (low/mid/high) per layer
3. Adds load balance loss to encourage balanced routing
4. Supports multi-stage training (subspace-only, routing-only, joint)
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torch.distributions as dist
from transformers import Trainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import DataCollatorForLanguageModeling
from accelerate import Accelerator
import numpy as np
import math
import random
from tqdm import tqdm
from pathlib import Path
import gc
import json
from datetime import datetime

# Fix for CUDA cusolver errors in multi-GPU training
# Use magma backend which is more stable for SVD operations
try:
    torch.backends.cuda.preferred_linalg_library('magma')
    print("Using MAGMA backend for linear algebra operations")
except:
    try:
        torch.backends.cuda.preferred_linalg_library('cusolver')
        print("Using cuSOLVER backend (default)")
    except:
        print("Warning: Could not set preferred linalg library")

from utils.datautils import prepare_train_loaders
from evaluate import evaluate_perplexity
from modules.dynamic_subspace import (
    MultiSubspaceSVDLayer,
    SharedParamMultiSubspaceSVDLayer,
    compute_load_balance_loss,
    get_model_routing_statistics,
    reset_model_routing_statistics
)


def main(args):
    # setting random seed of numpy and torch
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # setting device
    print(f"Target Ratio: {args.target_ratio}")
    print(f"Number of Subspaces: {args.n_subspaces}")
    print(f"Routing Strategy: {args.routing_strategy}")
    gpu_count = torch.cuda.device_count()
    print(f"Visible GPU count: {gpu_count}")
    NGPUS = gpu_count
    DEV_GPU = torch.device('cuda:0')
    DEV_CPU= torch.device('cpu')

    target_compression_ratio = args.target_ratio

    accelerator = Accelerator()

    SEQ_LEN = args.seq_len

    NSAMPLES_per_GPU_train = math.ceil(args.n_train_samples/NGPUS)
    NSAMPLES_per_GPU_val = math.ceil(args.n_eval_samples/NGPUS)


    SAVE = args.SAVE
    RECREATE = args.RECREATE
    DO_SAMPLE = args.DO_SAMPLE
    remapping = args.remapping

    NSAMPLES_train = args.n_train_samples
    NSAMPLES_val = args.n_eval_samples


    BETA = args.BETA

    # 训练设置, TA is TrainArgument
    TA_num_train_epochs = args.n_train_epochs
    TA_warmup_steps = args.warmup_steps
    TA_gradient_accumulation_steps = args.gradient_accumulation_steps
    save_epoch_num = args.save_epoch_num


    lambda_reg = args.lambda_reg
    lambda_balance = args.lambda_balance  # Load balance loss weight

    # 优化器设置：
    scheduler_lr=args.scheduler_lr
    scheduler_step_size= math.ceil(NSAMPLES_per_GPU_train / TA_gradient_accumulation_steps)
    scheduler_gamma=args.scheduler_gamma
    scheduler_min_lr =args.scheduler_min_lr

    # Dynamic routing specific settings
    n_subspaces = args.n_subspaces
    routing_strategy = args.routing_strategy
    use_soft_routing = args.use_soft_routing
    routing_temperature = args.routing_temperature
    learnable_thresholds = args.learnable_thresholds

    # Advanced routing settings
    advanced_routing = args.advanced_routing
    advanced_routing_kwargs = {}
    if advanced_routing is not None:
        print(f"Using advanced routing strategy: {advanced_routing}")
        if advanced_routing == 'topk':
            advanced_routing_kwargs['top_k'] = args.advanced_routing_topk
            advanced_routing_kwargs['capacity_factor'] = args.advanced_routing_capacity
        elif advanced_routing == 'expert_choice':
            # ExpertChoiceRouter uses tokens_per_expert, not top_k
            # If None, it will auto-calculate as seq_len / n_subspaces
            # We don't set it here - let the router handle it automatically
            pass
        elif advanced_routing == 'sinkhorn':
            advanced_routing_kwargs['sinkhorn_iters'] = args.advanced_routing_sinkhorn_iters
        # gating and adaptive don't need extra kwargs
    else:
        advanced_routing_kwargs = None

   # load model
    model_load_dtype = torch.float16
    computeSVD_dtype = torch.float32

    model_id = args.model_id
    print("processing model: ", model_id.split('/')[-1])

    model_no_svd_layer_dic = {}
    lower_id = model_id.split('/')[-1]

    if "llama" in lower_id or "Llama" in lower_id:
        model_no_svd_layer_dic[lower_id] = ['lm_head']
    elif "opt" in lower_id:
        model_no_svd_layer_dic[lower_id] = ['project_out', 'project_in']

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=model_load_dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Set Path
    DATASET_NAME = args.training_dataset
    path_head_folder = Path(args.path_head_folder)
    path_head_folder_output = Path(args.path_head_folder_output)
    model_folder = path_head_folder / 'models' / lower_id
    data_cache_dir = path_head_folder / 'data_cache' / DATASET_NAME
    data_cache_dir.mkdir(parents=True, exist_ok=True)
    dataset_cache_dir = path_head_folder / 'datasets' / DATASET_NAME
    dataset_cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir = path_head_folder_output / 'training_output' / lower_id
    output_dir.mkdir(parents=True, exist_ok=True)
    print("The output will be saved in: ", output_dir)


    # prepared dataset
    tokenized_traindata, tokenized_valdata = prepare_train_loaders(tokenizer, DATASET_NAME, data_cache_dir, dataset_cache_dir, args)


    # evaluate ppl of original model
    print("Start evaluating the original model's PPL.")
    val_input_ids = torch.cat([_["input_ids"].unsqueeze(0) for _ in tokenized_valdata], 0)
    model.to(DEV_GPU)
    orig_PPL = evaluate_perplexity(model, val_input_ids, NSAMPLES_val)
    model.to(DEV_CPU)
    print(f"Original Perplexity: {orig_PPL}")



    # transform model to MultiSubspaceSVDLayer
    print(f"Transforming model to MultiSubspaceSVDLayer with {n_subspaces} subspaces...")
    model_ori_weight_size = torch.tensor(0)

    for name, module in tqdm(model.named_modules(), desc="Add Multi-Subspace SVD attribute to modules"):
        if isinstance(module, nn.Linear) and all(x not in name for x in model_no_svd_layer_dic[lower_id]):
            parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
            attr_name = name.rsplit('.', 1)[-1]
            if parent_name != '':
                parent = dict(model.named_modules())[parent_name]
            else:
                parent = model
            RANK_RATIO = int(min(module.in_features, module.out_features)/SEQ_LEN)

            # CRITICAL: Prevent division by zero
            if RANK_RATIO == 0:
                print(f"[WARNING] RANK_RATIO = 0 for layer {name} (in_features={module.in_features}, out_features={module.out_features}, SEQ_LEN={SEQ_LEN})")
                print(f"  Setting RANK_RATIO = 1 to prevent NaN gamma values")
                RANK_RATIO = 1

            # Calculate base gamma
            if remapping:
                gamma_base = (1/RANK_RATIO)*target_compression_ratio*min(module.in_features, module.out_features)
            else:
                gamma_base =(1/RANK_RATIO)*target_compression_ratio*module.in_features*module.out_features/(module.in_features+module.out_features)

            # CRITICAL: Check if gamma_base is valid
            if not (0 < gamma_base < 10000):  # Sanity check
                print(f"[ERROR] Invalid gamma_base = {gamma_base} for layer {name}")
                print(f"  RANK_RATIO={RANK_RATIO}, target_compression_ratio={target_compression_ratio}")
                print(f"  in_features={module.in_features}, out_features={module.out_features}")
                raise ValueError(f"Invalid gamma_base computed for layer {name}")

            # Initialize multiple gammas for different subspaces
            # Strategy: Use layer-aware initialization for better compression
            if args.use_layer_aware_gamma if hasattr(args, 'use_layer_aware_gamma') else False:
                from utils.training_optimizers import compute_adaptive_gamma_multipliers
                gamma_multipliers = compute_adaptive_gamma_multipliers(
                    name,
                    args.gamma_multipliers if hasattr(args, 'gamma_multipliers') else [0.5, 1.0, 1.5],
                    n_subspaces
                )
                print(f"[Adaptive] Layer {name}: using adjusted multipliers {[f'{m:.2f}' for m in gamma_multipliers]}")
            else:
                gamma_multipliers = args.gamma_multipliers if hasattr(args, 'gamma_multipliers') else [0.5, 1.0, 1.5]

            gammas = [gamma_base * mult for mult in gamma_multipliers[:n_subspaces]]

            print(f"Layer {name}: gammas = {[f'{g:.2f}' for g in gammas]}")

            weight_size = torch.tensor(module.in_features * module.out_features)
            model_ori_weight_size += weight_size

            # Choose layer type based on parameter sharing setting
            use_shared = args.use_shared_params if hasattr(args, 'use_shared_params') else False

            if use_shared:
                # Use parameter-shared version (saves ~66% VRAM)
                use_grad_ckpt = args.use_gradient_checkpointing if hasattr(args, 'use_gradient_checkpointing') else False
                print(f"[SharedParam] Creating SharedParamMultiSubspaceSVDLayer for {name}")
                if use_grad_ckpt:
                    print(f"[SharedParam] Gradient checkpointing ENABLED for {name}")
                NewLayer = SharedParamMultiSubspaceSVDLayer(
                    gammas=gammas,
                    n_subspaces=n_subspaces,
                    SEQ_LEN=SEQ_LEN,
                    beta=BETA,
                    input_size=module.in_features,
                    output_size=module.out_features,
                    weight_size=weight_size,
                    weight=module.weight,
                    bias=module.bias,
                    name=name,
                    device=model.device,
                    routing_strategy=routing_strategy,
                    learnable_thresholds=learnable_thresholds,
                    use_soft_routing=use_soft_routing,
                    routing_temperature=routing_temperature,
                    advanced_routing=advanced_routing,
                    advanced_routing_kwargs=advanced_routing_kwargs,
                    svd_rank=args.shared_svd_rank if hasattr(args, 'shared_svd_rank') else None,
                    use_gradient_checkpointing=use_grad_ckpt
                )
            else:
                # Use standard version (independent SVD per subspace)
                NewLayer = MultiSubspaceSVDLayer(
                    gammas=gammas,
                    n_subspaces=n_subspaces,
                    SEQ_LEN=SEQ_LEN,
                    beta=BETA,
                    input_size=module.in_features,
                    output_size=module.out_features,
                    weight_size=weight_size,
                    weight=module.weight,
                    bias=module.bias,
                    name=name,
                    device=model.device,
                    routing_strategy=routing_strategy,
                    learnable_thresholds=learnable_thresholds,
                    use_soft_routing=use_soft_routing,
                    routing_temperature=routing_temperature,
                    advanced_routing=advanced_routing,
                    advanced_routing_kwargs=advanced_routing_kwargs
                )
            setattr(parent, attr_name, NewLayer)
            del module

    model.register_buffer('ori_weight_size', model_ori_weight_size)
    model.register_buffer('epoch_cnt', torch.tensor(0))
    model.register_buffer('BEST_loss', torch.tensor(float('inf'), dtype=computeSVD_dtype, device = model.device))
    gc.collect()
    print("transform done")


    # frozen other paramters
    now = datetime.now()
    formatted_time = now.strftime("%m%d-%H:%M:%S")
    experiment_name = f"DynamicSubspace-{n_subspaces}subspace-{routing_strategy}-{target_compression_ratio}_{DATASET_NAME}_{SEQ_LEN}_{formatted_time}"
    if remapping:
        experiment_name = "Remapping-" + experiment_name
    else:
        experiment_name = "Noremapping-" + experiment_name

    TA_tarined_model_output_dir = output_dir / experiment_name
    TA_tarined_model_output_dir.mkdir(parents=True, exist_ok=True)

    # Save experiment config
    config_dict = {
        'model_id': model_id,
        'target_compression_ratio': target_compression_ratio,
        'n_subspaces': n_subspaces,
        'routing_strategy': routing_strategy,
        'learnable_thresholds': learnable_thresholds,
        'use_soft_routing': use_soft_routing,
        'routing_temperature': routing_temperature,
        'lambda_reg': lambda_reg,
        'lambda_balance': lambda_balance,
        'gamma_multipliers': gamma_multipliers[:n_subspaces],
        'advanced_routing': advanced_routing,
        'advanced_routing_kwargs': advanced_routing_kwargs,
        'original_ppl': orig_PPL
    }
    with open(TA_tarined_model_output_dir / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=4)

    # Set trainable parameters
    for name, param in model.named_parameters():
        param.requires_grad = False

    for module in model.modules():
        # Handle both MultiSubspaceSVDLayer and SharedParamMultiSubspaceSVDLayer
        if isinstance(module, (MultiSubspaceSVDLayer, SharedParamMultiSubspaceSVDLayer)):
            # Make all gammas trainable
            for gamma in module.gammas:
                gamma.requires_grad = True

            # Make routing parameters trainable if using learned routing
            if routing_strategy == 'learned':
                for param in module.router.importance_net.parameters():
                    param.requires_grad = True

            # Make thresholds trainable if enabled
            if learnable_thresholds:
                module.router.thresholds.requires_grad = True

            # Make advanced routing parameters trainable if using gating
            if advanced_routing == 'gating' and hasattr(module.router, 'advanced_router'):
                if hasattr(module.router.advanced_router, 'gating_network'):
                    for param in module.router.advanced_router.gating_network.parameters():
                        param.requires_grad = True
                        print(f"Making gating network parameters trainable for {module.name}")




   # Helper function to get the actual model (unwrap if wrapped by DDP/accelerate)
    def get_actual_model(model):
        """Get the actual model, unwrapping if necessary."""
        if hasattr(model, 'module'):
            return model.module
        return model

   # SVDTrainer for MultiSubspaceSVDLayer
    def calculate_compression_loss(model, target_compression_ratio, lambda_reg):
        actual_model = get_actual_model(model)
        size_new = torch.tensor(0., device=actual_model.device)

        for name, module in model.named_modules():
            if isinstance(module, MultiSubspaceSVDLayer):
                RANK_RATIO = min(module.ori.in_features, module.ori.out_features) / SEQ_LEN

                # For multi-subspace, we use the average gamma weighted by routing distribution
                routing_dist = module.router.get_routing_distribution()
                avg_gamma = sum(gamma * routing_dist[i].item() for i, gamma in enumerate(module.gammas))

                if remapping:
                    size_now = max(module.ori.in_features, module.ori.out_features) * avg_gamma * RANK_RATIO
                else:
                    size_now = module.ori.in_features * avg_gamma * RANK_RATIO + module.ori.out_features * avg_gamma * RANK_RATIO

                size_ori = module.ori_weight_size
                size_new = torch.where(size_now < size_ori, size_now, size_ori) + size_new

        compression_ratio = size_new / actual_model.ori_weight_size

        compression_loss = abs(compression_ratio - torch.tensor(target_compression_ratio, device=compression_ratio.device))
        return lambda_reg * compression_loss, compression_ratio

    def Wrong_value_loss(model):
        actual_model = get_actual_model(model)
        penalty = torch.tensor(0., device=actual_model.device)

        for name, module in model.named_modules():
            if isinstance(module, MultiSubspaceSVDLayer):
                for gamma in module.gammas:
                    lower_penalty = torch.relu(-gamma) ** 2
                    upper_penalty = torch.relu(gamma - torch.tensor(SEQ_LEN, device=gamma.device)) ** 2
                    penalty += lower_penalty + upper_penalty

        return penalty

    class DynamicSVDTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            # Extract custom arguments before passing to parent
            self.use_gamma_regularization = kwargs.pop('use_gamma_regularization', False)
            self.use_temperature_scheduling = kwargs.pop('use_temperature_scheduling', False)
            self.temp_scheduler = kwargs.pop('temp_scheduler', None)
            self.gamma_regularizer = kwargs.pop('gamma_regularizer', None)

            super().__init__(*args, **kwargs)

            # Initialize step counter for temperature scheduling
            self.training_step_counter = 0

        def create_optimizer(self):
            """Override to use differentiated learning rates for gamma parameters."""
            if self.optimizer is None:
                # Check if we should use differentiated learning rates
                use_diff_lr = getattr(self.args, 'use_differentiated_lr', False)

                if use_diff_lr:
                    from utils.training_optimizers import setup_differentiated_optimizer
                    gamma_lr = getattr(self.args, 'gamma_lr', 1e-3)
                    other_lr = getattr(self.args, 'other_lr', 1e-4)
                    weight_decay = getattr(self.args, 'weight_decay', 1e-5)

                    self.optimizer = setup_differentiated_optimizer(
                        self.model,
                        gamma_lr=gamma_lr,
                        other_lr=other_lr,
                        weight_decay=weight_decay
                    )
                    print(f"[Optimizer] Using differentiated learning rates: "
                          f"gamma_lr={gamma_lr}, other_lr={other_lr}")
                else:
                    # Use default optimizer
                    super().create_optimizer()

        def compute_loss(self, model, inputs, return_outputs=False):
            actual_model = get_actual_model(model)
            outputs = model(**inputs)

            # Get cross-entropy loss (negative log likelihood)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            neg_log_likelihood = loss

            # Compute PPL for logging only (with numerical stability)
            # Clamp NLL to prevent overflow: exp(20) = 485M is already huge
            clamped_nll = torch.clamp(neg_log_likelihood, max=20.0)
            ppl = torch.exp(clamped_nll)

            # IMPORTANT: Use NLL as loss, NOT PPL!
            # PPL is only for monitoring/logging
            loss = neg_log_likelihood

            # Compression regularization
            reg_loss, compression_ratio = calculate_compression_loss(model, target_compression_ratio, lambda_reg)

            # Value penalty
            value_loss = Wrong_value_loss(model)

            # Load balance loss (encourage balanced routing)
            balance_loss = compute_load_balance_loss(model) * lambda_balance

            # OPTIMIZATION: Add gamma regularization if enabled
            gamma_reg_loss = torch.tensor(0.0, device=loss.device)
            gamma_reg_stats = {}
            if self.use_gamma_regularization and self.gamma_regularizer is not None:
                gamma_reg_loss, gamma_reg_stats = self.gamma_regularizer.compute_loss(model)

            # OPTIMIZATION: Update temperature if enabled
            if self.use_temperature_scheduling and self.temp_scheduler is not None:
                current_temp = self.temp_scheduler.get_temperature(self.training_step_counter)
                # Update temperature in all routing modules
                for module in model.modules():
                    if isinstance(module, (MultiSubspaceSVDLayer, SharedParamMultiSubspaceSVDLayer)):
                        module.routing_temperature = current_temp

                # Log temperature every 100 steps
                if self.training_step_counter % 100 == 0:
                    print(f"[TempScheduler] Step {self.training_step_counter}: temperature={current_temp:.3f}")

                self.training_step_counter += 1

            total_loss = loss + reg_loss + value_loss + balance_loss + gamma_reg_loss

            # Check for NaN/Inf and provide detailed error message
            # Use .any() to handle multi-element tensors
            is_nan_or_inf = torch.isnan(total_loss).any() or torch.isinf(total_loss).any()

            if is_nan_or_inf:
                print(f"[ERROR] NaN/Inf detected in loss computation:")

                # Safe item extraction
                def safe_item(tensor):
                    try:
                        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                            return 'NaN/Inf'
                        return tensor.item() if tensor.numel() == 1 else tensor.mean().item()
                    except:
                        return 'Error'

                print(f"  - task_loss (NLL): {safe_item(loss)}")
                print(f"  - reg_loss: {safe_item(reg_loss)}")
                print(f"  - value_loss: {safe_item(value_loss)}")
                print(f"  - balance_loss: {safe_item(balance_loss)}")
                print(f"  - gamma_reg_loss: {safe_item(gamma_reg_loss) if isinstance(gamma_reg_loss, torch.Tensor) else gamma_reg_loss}")
                print(f"  - total_loss: {safe_item(total_loss)}")

                # Replace NaN/Inf with a large but finite value to continue training
                total_loss = torch.where(
                    torch.isnan(total_loss) | torch.isinf(total_loss),
                    torch.tensor(1e6, device=total_loss.device, dtype=total_loss.dtype),
                    total_loss
                )
                print(f"  - Replaced with: {safe_item(total_loss)}")

            cur_lr = self.optimizer.param_groups[0]['lr']

            actual_model.epoch_cnt += 1
            if actual_model.epoch_cnt % save_epoch_num == 0:
                k_dict = {}

                # Save all gammas for each layer
                for name, module in self.model.named_modules():
                    if isinstance(module, (MultiSubspaceSVDLayer, SharedParamMultiSubspaceSVDLayer)):
                        k_dict[name] = {
                            'gammas': [g.detach().item() for g in module.gammas],
                            'routing_distribution': module.router.get_routing_distribution().cpu().tolist()
                        }

                k_dict['ppl'] = ppl.detach().tolist()
                k_dict['compression_ratio'] = compression_ratio.detach().tolist()
                k_dict['balance_loss'] = balance_loss.detach().tolist()
                k_dict['lr'] = cur_lr

                # Add gamma regularization stats if available
                if gamma_reg_stats:
                    k_dict['gamma_reg_stats'] = gamma_reg_stats

                output_json_path = str(TA_tarined_model_output_dir/'k_dict_{:05d}.json'.format(actual_model.epoch_cnt))
                with open(output_json_path, 'w') as json_file:
                    json.dump(k_dict, json_file, indent=4)

                # Save best model
                BEST_loss = actual_model.BEST_loss
                CURR_loss = total_loss.mean().item()
                if CURR_loss < BEST_loss:
                    actual_model.BEST_loss = torch.tensor(CURR_loss, device = actual_model.BEST_loss.device)
                    k_dict["PPL_ORIG"] = orig_PPL

                    # Also save routing statistics
                    routing_stats = get_model_routing_statistics(model)
                    k_dict["routing_stats"] = {k: {
                        'routing_distribution': v['routing_distribution'].tolist() if isinstance(v['routing_distribution'], np.ndarray) else v['routing_distribution'],
                        'gammas': v['gammas']
                    } for k, v in routing_stats.items()}

                    output_json_path = str(TA_tarined_model_output_dir/'best_gamma.json')
                    with open(output_json_path, 'w') as json_file:
                        json.dump(k_dict, json_file, indent=4)

            return (total_loss, outputs) if return_outputs else total_loss

        def training_step(self, model, inputs):
            """
            Override training_step to add gradient clipping for gamma parameters.

            This prevents gamma parameters from becoming NaN due to gradient explosion.
            """
            import torch

            model.train()
            inputs = self._prepare_inputs(inputs)

            # Forward pass
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)

            # Ensure loss is scalar (required for backward())
            if loss.ndim > 0:
                loss = loss.mean()

            # Backward pass
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps

            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)

            # NUMERICAL STABILITY: Clip gradients for gamma parameters specifically
            # This happens AFTER backward but BEFORE optimizer step
            actual_model = get_actual_model(model)
            gamma_params = []

            for module in actual_model.modules():
                if isinstance(module, (MultiSubspaceSVDLayer, SharedParamMultiSubspaceSVDLayer)):
                    if hasattr(module, 'gammas'):
                        for gamma in module.gammas:
                            if gamma.grad is not None:
                                gamma_params.append(gamma)

            if gamma_params:
                # First check for NaN gradients and zero them out
                for gamma in gamma_params:
                    if gamma.grad is not None and (torch.isnan(gamma.grad).any() or torch.isinf(gamma.grad).any()):
                        print(f"[WARNING] NaN/Inf gradient detected in gamma parameter. Zeroing gradient.")
                        gamma.grad.zero_()

                # Then clip gamma gradients to prevent explosion
                # Use a smaller clip value (0.5) for gamma since they're sensitive
                total_norm = torch.nn.utils.clip_grad_norm_(gamma_params, max_norm=0.5)
                if total_norm > 0.5:
                    print(f"[CLIP] Gamma gradient norm {total_norm:.3f} clipped to 0.5")

            return loss.detach()

        def create_scheduler(self, num_training_steps, optimizer):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_step_size, eta_min=scheduler_min_lr)
            self.lr_scheduler=scheduler
            return self.lr_scheduler

        def create_optimizer(self):
            optimizer = torch.optim.Adam(self.model.parameters(), lr = scheduler_lr)
            cur_lr = optimizer.param_groups[0]['lr']
            self.optimizer=optimizer
            return self.optimizer





   # training
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    from transformers import TrainingArguments
    training_args = TrainingArguments(
        output_dir= TA_tarined_model_output_dir,
        num_train_epochs = TA_num_train_epochs,
        evaluation_strategy = "epoch",
        per_device_train_batch_size = 1,
        per_device_eval_batch_size = 1,
        warmup_steps = TA_warmup_steps,
        lr_scheduler_type = "cosine",
        seed = args.seed,
        gradient_accumulation_steps = TA_gradient_accumulation_steps,
        save_strategy = "no",
        save_steps = 1000,
        save_total_limit = 2,
        remove_unused_columns=False,
    )

    # OPTIMIZATION: Initialize advanced training optimizers if enabled
    temp_scheduler = None
    gamma_regularizer = None
    use_temp_scheduling = args.use_temperature_scheduling
    use_gamma_reg = args.use_gamma_regularization

    if use_temp_scheduling:
        from utils.training_optimizers import TemperatureScheduler
        temp_scheduler = TemperatureScheduler(
            initial_temp=args.temp_initial,
            final_temp=args.temp_final,
            decay_steps=args.temp_decay_steps,
            decay_type=args.temp_decay_type
        )
        print(f"[Optimization] Temperature scheduling enabled: {args.temp_initial}→{args.temp_final} over {args.temp_decay_steps} steps ({args.temp_decay_type})")

    if use_gamma_reg:
        from utils.training_optimizers import GammaRegularizer
        gamma_regularizer = GammaRegularizer(
            l1_weight=args.gamma_l1_weight,
            diversity_weight=args.gamma_diversity_weight
        )
        print(f"[Optimization] Gamma regularization enabled: L1={args.gamma_l1_weight}, Diversity={args.gamma_diversity_weight}")

    # Add differentiated LR flag to training_args for trainer access
    if args.use_differentiated_lr:
        training_args.use_differentiated_lr = True
        training_args.gamma_lr = args.gamma_lr
        training_args.other_lr = args.other_lr
        print(f"[Optimization] Differentiated learning rates: gamma_lr={args.gamma_lr}, other_lr={args.other_lr}")

    trainer = DynamicSVDTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_traindata,
        eval_dataset=tokenized_valdata,
        data_collator=data_collator,
        use_temperature_scheduling=use_temp_scheduling,
        use_gamma_regularization=use_gamma_reg,
        temp_scheduler=temp_scheduler,
        gamma_regularizer=gamma_regularizer,
    )


    model,  train_dataloader, eval_dataloader = accelerator.prepare(
        model, trainer.get_train_dataloader(), trainer.get_eval_dataloader()
    )
    trainer.train = accelerator.prepare(trainer.train)


    # train
    print("Starting training...")
    trainer.train()
    print("Training completed!")

    # Save final gammas and routing statistics
    final_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, MultiSubspaceSVDLayer):
            final_dict[name] = {
                'gammas': [g.detach().item() for g in module.gammas],
                'routing_distribution': module.router.get_routing_distribution().cpu().tolist()
            }

    with open(TA_tarined_model_output_dir / 'final_gamma.json', 'w') as f:
        json.dump(final_dict, f, indent=4)

    print(f"Training outputs saved to: {TA_tarined_model_output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Original arguments
    parser.add_argument('--model_id', type=str, default='facebook/opt-125m')
    parser.add_argument('--target_ratio', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seq_len', type=int, default=2048)
    parser.add_argument('--n_train_samples', type=int, default=128)
    parser.add_argument('--n_eval_samples', type=int, default=64)
    parser.add_argument('--BETA', type=float, default=10.0)  # Reduced from 100.0 for numerical stability
    parser.add_argument('--n_train_epochs', type=int, default=5)
    parser.add_argument('--warmup_steps', type=int, default=10)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--save_epoch_num', type=int, default=50)
    parser.add_argument('--lambda_reg', type=float, default=10.0)
    parser.add_argument('--scheduler_lr', type=float, default=1e-3)
    parser.add_argument('--scheduler_gamma', type=float, default=0.9)
    parser.add_argument('--scheduler_min_lr', type=float, default=1e-5)
    parser.add_argument('--SAVE', action='store_true')
    parser.add_argument('--RECREATE', action='store_true')
    parser.add_argument('--DO_SAMPLE', action='store_true')
    parser.add_argument('--remapping', action='store_true')
    parser.add_argument('--training_dataset', type=str, default='wikitext')
    parser.add_argument('--path_head_folder', type=str, default='./results')
    parser.add_argument('--path_head_folder_output', type=str, default='./results')

    # Dynamic subspace specific arguments
    parser.add_argument('--n_subspaces', type=int, default=3, help='Number of subspaces')
    parser.add_argument('--routing_strategy', type=str, default='norm',
                       choices=['norm', 'l1_norm', 'value_aware', 'learned', 'attention', 'hybrid'],
                       help='Token importance routing strategy (value_aware=SOTA from EMNLP24)')
    parser.add_argument('--learnable_thresholds', action='store_true', help='Make routing thresholds learnable')
    parser.add_argument('--use_soft_routing', action='store_true', help='Use soft routing instead of hard')
    parser.add_argument('--routing_temperature', type=float, default=1.0)
    parser.add_argument('--lambda_balance', type=float, default=0.01, help='Load balance loss weight')
    parser.add_argument('--gamma_multipliers', nargs='+', type=float, default=[0.5, 1.0, 1.5])

    # Advanced routing strategies (optional, provides better load balancing)
    parser.add_argument('--advanced_routing', type=str, default=None,
                       choices=['topk', 'expert_choice', 'sinkhorn', 'gating', 'adaptive'],
                       help='Advanced routing method: topk (Switch Transformer), expert_choice (Google 2022), '
                            'sinkhorn (Optimal Transport), gating (learnable MLP), adaptive (dynamic quantiles)')
    parser.add_argument('--advanced_routing_topk', type=int, default=1,
                       help='Top-k value for topk routing (default: 1)')
    parser.add_argument('--advanced_routing_capacity', type=float, default=1.25,
                       help='Capacity factor for expert_choice/topk routing (default: 1.25)')
    parser.add_argument('--advanced_routing_sinkhorn_iters', type=int, default=3,
                       help='Number of Sinkhorn iterations (default: 3)')

    # Parameter sharing for training (VRAM optimization)
    parser.add_argument('--use_shared_params', action='store_true',
                       help='Use SharedParamMultiSubspaceSVDLayer (shares U,V across subspaces, saves ~66% VRAM)')
    parser.add_argument('--shared_svd_rank', type=int, default=None,
                       help='Rank for shared SVD (default: max_gamma + 10)')
    parser.add_argument('--use_gradient_checkpointing', action='store_true',
                       help='Enable gradient checkpointing for VRAM savings (trades 20%% compute for 20-30%% memory)')

    # Advanced training optimizations
    parser.add_argument('--use_temperature_scheduling', action='store_true',
                       help='Enable adaptive temperature scheduling for routing (exploration→exploitation)')
    parser.add_argument('--temp_initial', type=float, default=5.0,
                       help='Initial temperature for routing (high=soft routing, default: 5.0)')
    parser.add_argument('--temp_final', type=float, default=0.5,
                       help='Final temperature for routing (low=hard routing, default: 0.5)')
    parser.add_argument('--temp_decay_steps', type=int, default=5000,
                       help='Number of steps to decay from initial to final temperature (default: 5000)')
    parser.add_argument('--temp_decay_type', type=str, default='exponential',
                       choices=['exponential', 'linear', 'cosine'],
                       help='Temperature decay schedule type (default: exponential)')

    parser.add_argument('--use_gamma_regularization', action='store_true',
                       help='Enable gamma regularization (L1 + diversity loss) for better compression')
    parser.add_argument('--gamma_l1_weight', type=float, default=0.01,
                       help='Weight for L1 regularization on gammas (encourages compression, default: 0.01)')
    parser.add_argument('--gamma_diversity_weight', type=float, default=0.001,
                       help='Weight for diversity regularization on gammas (encourages differentiation, default: 0.001)')

    parser.add_argument('--use_differentiated_lr', action='store_true',
                       help='Use higher learning rate for gamma parameters (faster convergence)')
    parser.add_argument('--gamma_lr', type=float, default=1e-3,
                       help='Learning rate for gamma parameters (default: 1e-3, 10x higher than base)')
    parser.add_argument('--other_lr', type=float, default=1e-4,
                       help='Learning rate for non-gamma trainable parameters (default: 1e-4)')

    parser.add_argument('--use_layer_aware_gamma', action='store_true',
                       help='Use layer-specific gamma initialization based on importance (embedding > attention > MLP)')

    args = parser.parse_args()
    main(args)
