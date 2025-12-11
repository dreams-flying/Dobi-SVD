"""
Training script for Matryoshka SVD compression.

This script trains language models compressed with Matryoshka SVD,
featuring:
1. Multi-scale training loss (trains all rank levels)
2. Rank regularization (encourages compression)
3. Adaptive rank selection (no routing overhead)

Usage:
    # Memory-optimized (80GB GPU):
    python train_matryoshka_svd.py \
        --model meta-llama/Llama-2-7b-hf \
        --dataset wikitext2 \
        --r_max 64 \
        --r_min 16 \
        --target_compression 0.15 \
        --importance_strategy learned \
        --enable_multiscale_loss \
        --n_train_samples 128 \
        --n_eval_samples 128 \
        --batch_size 1 \
        --gradient_accumulation_steps 8 \
        --seq_len 512 \
        --use_fp16

Author: Claude (Anthropic)
Date: 2025-12-10
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
from tqdm import tqdm

# Add modules to path
sys.path.append(str(Path(__file__).parent))

from modules.matryoshka_svd import (
    MatryoshkaSVDConfig,
    MatryoshkaSVDLayer,
    replace_linear_with_matryoshka_svd
)
from utils.datautils import prepare_train_loaders


class MatryoshkaSVDTrainer:
    """Trainer for Matryoshka SVD compressed models."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"[Matryoshka SVD Trainer]")
        print(f"  Device: {self.device}")
        print(f"  Model: {args.model}")
        print(f"  Rank range: [{args.r_min}, {args.r_max}]")
        print(f"  Importance strategy: {args.importance_strategy}")
        print(f"  Multi-scale loss: {args.enable_multiscale_loss}")
        print(f"  Target compression: {args.target_compression:.1%}")
        print(f"  Gradient accumulation steps: {args.gradient_accumulation_steps}")

        # Memory optimization
        if torch.cuda.is_available():
            print(f"\n[Memory Info]")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            torch.cuda.empty_cache()

    def prepare_model(self):
        """Load model and apply Matryoshka SVD compression."""
        print("\n" + "="*80)
        print("STEP 1: Loading base model")
        print("="*80)

        # Load base model
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # CRITICAL MEMORY FIX: Load model on CPU FIRST
        # Then replace layers on CPU, finally move to GPU
        # This prevents OOM when old and new layers compete for GPU memory during replacement
        print(f"  CRITICAL: Loading model on CPU (not GPU)")
        print(f"  Will perform SVD replacement on CPU, then move to GPU")
        print(f"  This prevents OOM during layer replacement phase")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.model,
            torch_dtype=torch.float32,  # Always FP32 for proper mixed precision training
            device_map=None,  # CRITICAL: Load on CPU, not GPU
            low_cpu_mem_usage=True
        )

        # Clear cache after loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            torch.cuda.empty_cache()

        print(f"Model loaded: {self.model.config.model_type}")
        original_params = sum(p.numel() for p in self.model.parameters())
        print(f"Original parameters: {original_params:,}")

        print("\n" + "="*80)
        print("STEP 2: Applying Matryoshka SVD compression")
        print("="*80)

        # Replace target layers with Matryoshka SVD
        target_layers = self.args.target_layers.split(',') if self.args.target_layers else None

        # MEMORY CRITICAL: Use aggressive memory saving mode
        print(f"\n⚠️  MEMORY OPTIMIZATION MODE: Aggressive")
        print(f"  - Processing layers one by one")
        print(f"  - Clearing GPU cache after each layer")
        print(f"  - Using CPU for all SVD computations")

        # Create configuration for Matryoshka SVD
        svd_config = MatryoshkaSVDConfig(
            r_max=self.args.r_max,
            r_min=self.args.r_min,
            importance_strategy=self.args.importance_strategy,
            temperature=self.args.temperature,
            soft_truncation_margin=self.args.soft_truncation_margin,
            enable_multiscale_loss=self.args.enable_multiscale_loss,
            rank_regularization_weight=self.args.rank_reg_weight,
            learnable_singular_values=self.args.learnable_singular_values,
            activation_aware_init=self.args.activation_aware_init,
            spectral_regularization_weight=self.args.spectral_reg_weight,
            auto_rank_range=self.args.auto_rank_range,
            energy_threshold_low=self.args.energy_threshold_low,
            energy_threshold_high=self.args.energy_threshold_high
        )

        print(f"\n📋 Matryoshka SVD Configuration:")
        print(f"  Rank range: [{svd_config.r_min}, {svd_config.r_max}]")
        print(f"  Importance strategy: {svd_config.importance_strategy}")
        print(f"  Temperature: {svd_config.temperature}")
        print(f"  Soft truncation margin: {svd_config.soft_truncation_margin}")
        print(f"  Learnable singular values: {svd_config.learnable_singular_values}")
        print(f"  Activation-aware init: {svd_config.activation_aware_init}")
        print(f"  Auto rank range: {svd_config.auto_rank_range}")
        print(f"  Bucketed inference: {svd_config.use_bucketed_inference}")

        # Collect calibration data if activation-aware initialization is enabled
        calibration_data = None
        if svd_config.activation_aware_init:
            print(f"\n📊 Collecting calibration data for activation-aware SVD...")
            calibration_data = self.collect_calibration_data(
                num_samples=self.args.calibration_samples
            )
            print(f"  Collected calibration data for {len(calibration_data)} layers")

        self.model = replace_linear_with_matryoshka_svd(
            self.model,
            target_layers=target_layers,
            config=svd_config,
            calibration_data=calibration_data,
            verbose=True
        )

        compressed_params = sum(p.numel() for p in self.model.parameters())
        print(f"\nCompressed parameters: {compressed_params:,}")
        print(f"Compression ratio: {compressed_params/original_params:.1%}")
        print(f"Reduction: {(1 - compressed_params/original_params):.1%}")

        # CRITICAL: Enable gradient checkpointing to save memory
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            print("\n⚠️  Enabling gradient checkpointing to save memory")
            self.model.gradient_checkpointing_enable()

        # CRITICAL: Aggressive memory cleanup after SVD replacement
        if torch.cuda.is_available():
            import gc
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()

        # CRITICAL: Now move model from CPU to GPU
        # Since we loaded with device_map=None, model is on CPU
        # We must explicitly move it to GPU for training
        if torch.cuda.is_available():
            print(f"\n{'='*80}")
            print(f"Moving model from CPU to GPU...")
            print(f"{'='*80}")

            # Use device_map='auto' for optimal multi-GPU distribution
            from accelerate import dispatch_model, infer_auto_device_map
            from accelerate.utils import get_balanced_memory

            # Get available GPU memory
            max_memory = get_balanced_memory(
                self.model,
                max_memory=None,
                no_split_module_classes=["LlamaDecoderLayer"],
                dtype=torch.float32
            )

            # Infer device map
            device_map = infer_auto_device_map(
                self.model,
                max_memory=max_memory,
                no_split_module_classes=["LlamaDecoderLayer"],
                dtype=torch.float32
            )

            # Dispatch model to devices
            self.model = dispatch_model(self.model, device_map=device_map)

            print(f"✓ Model moved to GPU")
            print(f"  Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            print(f"  Memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

        self.model.train()

    def collect_calibration_data(self, num_samples: int = 128) -> Dict[str, torch.Tensor]:
        """
        Collect calibration data for activation-aware SVD initialization.

        Args:
            num_samples: Number of samples to use for calibration

        Returns:
            Dict mapping layer names to activation tensors [num_samples, hidden_size]
        """
        print(f"\n📊 Collecting calibration data ({num_samples} samples)...")

        # Temporarily prepare data if not already done
        if not hasattr(self, 'train_loader'):
            self.prepare_data()

        calibration_data = {}
        hooks = []

        # Register forward hooks to capture activations
        def make_hook(name):
            def hook(module, input, output):
                if name not in calibration_data:
                    calibration_data[name] = []
                # Store input activations (before the linear layer)
                if isinstance(input, tuple):
                    input_tensor = input[0]
                else:
                    input_tensor = input

                # Flatten batch and sequence dimensions
                # Shape: [batch, seq, hidden] -> [batch*seq, hidden]
                act = input_tensor.detach().cpu().reshape(-1, input_tensor.shape[-1])
                calibration_data[name].append(act)
            return hook

        # Register hooks for target layers
        target_patterns = self.args.target_layers.split(',') if self.args.target_layers else []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                should_hook = any(pattern in name for pattern in target_patterns)
                if should_hook:
                    hook = module.register_forward_hook(make_hook(name))
                    hooks.append(hook)

        # Run forward passes to collect activations
        self.model.eval()
        collected = 0

        with torch.no_grad():
            for batch in self.train_loader:
                if collected >= num_samples:
                    break

                input_ids = batch['input_ids'].to(next(self.model.parameters()).device)

                # Forward pass (will trigger hooks)
                try:
                    _ = self.model(input_ids=input_ids, use_cache=False)
                except:
                    pass  # Ignore errors, we just need the activations

                collected += input_ids.shape[0]

                # Clear memory
                del input_ids
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Concatenate and subsample to exactly num_samples
        final_calibration_data = {}
        for layer_name, acts_list in calibration_data.items():
            acts = torch.cat(acts_list, dim=0)  # [total_tokens, hidden]

            # Subsample to num_samples
            if acts.shape[0] > num_samples:
                indices = torch.randperm(acts.shape[0])[:num_samples]
                acts = acts[indices]

            final_calibration_data[layer_name] = acts
            print(f"  {layer_name}: {acts.shape}")

        self.model.train()

        return final_calibration_data

    def prepare_data(self):
        """Prepare training and evaluation datasets."""
        print("\n" + "="*80)
        print("STEP 3: Preparing datasets")
        print("="*80)

        # Setup paths for caching
        path_head_folder = Path(self.args.path_head_folder)
        data_cache_dir = path_head_folder / "data_cache"
        dataset_cache_dir = path_head_folder / "dataset_cache"
        data_cache_dir.mkdir(parents=True, exist_ok=True)
        dataset_cache_dir.mkdir(parents=True, exist_ok=True)

        # Use Dobi-SVD's dataset loading function
        tokenized_traindata, tokenized_valdata = prepare_train_loaders(
            tokenizer=self.tokenizer,
            DATASET_NAME=self.args.dataset,
            data_cache_dir=data_cache_dir,
            dataset_cache_dir=dataset_cache_dir,
            args=self.args
        )

        # Create dataloaders from tokenized data
        # CRITICAL: Use num_workers=0 to avoid forking issues and memory duplication
        self.train_loader = DataLoader(
            tokenized_traindata,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=0,  # Avoid forking issues
            pin_memory=True if torch.cuda.is_available() else False
        )

        self.eval_loader = DataLoader(
            tokenized_valdata,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=0,  # Avoid forking issues
            pin_memory=True if torch.cuda.is_available() else False
        )

        print(f"Train samples: {len(tokenized_traindata)}")
        print(f"Eval samples: {len(tokenized_valdata)}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Eval batches: {len(self.eval_loader)}")

    def setup_training(self):
        """Setup optimizer and scheduler."""
        print("\n" + "="*80)
        print("STEP 4: Setting up training")
        print("="*80)

        # Collect parameters
        svd_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # Matryoshka SVD layers should be frozen (U, S, V are buffers)
            # Only importance predictor parameters are trainable
            if 'importance_predictor' in name or 'predictor' in name:
                svd_params.append(param)
            else:
                other_params.append(param)

        print(f"Trainable parameters:")
        print(f"  - Importance predictors: {sum(p.numel() for p in svd_params):,}")
        print(f"  - Other (embeddings, LM head): {sum(p.numel() for p in other_params):,}")

        # Optimizer with differentiated learning rates
        if self.args.use_differentiated_lr:
            param_groups = [
                {'params': svd_params, 'lr': self.args.importance_lr},
                {'params': other_params, 'lr': self.args.other_lr}
            ]
            print(f"  - Importance predictor LR: {self.args.importance_lr}")
            print(f"  - Other LR: {self.args.other_lr}")
        else:
            param_groups = [
                {'params': svd_params + other_params, 'lr': self.args.learning_rate}
            ]
            print(f"  - Unified LR: {self.args.learning_rate}")

        self.optimizer = torch.optim.AdamW(
            param_groups,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self.args.weight_decay
        )

        # Learning rate scheduler
        total_steps = len(self.train_loader) * self.args.num_epochs
        warmup_steps = int(total_steps * 0.05)  # 5% warmup

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        print(f"Total training steps: {total_steps}")
        print(f"Warmup steps: {warmup_steps}")

        # CRITICAL: Setup mixed precision training with GradScaler
        if self.args.use_fp16 and torch.cuda.is_available():
            print(f"\n⚠️  Enabling mixed precision training (FP16)")
            self.scaler = torch.cuda.amp.GradScaler()
            self.use_amp = True
        else:
            self.scaler = None
            self.use_amp = False

    def compute_multiscale_loss(self, batch):
        """
        Compute multi-scale training loss across different rank levels.

        This is a KEY INNOVATION of Matryoshka SVD (vs Dobi-SVD):
        - Dobi-SVD optimizes for a single compression point
        - Matryoshka SVD optimizes across the entire rank range [r_min, r_max]

        Implementation:
        1. Main loss with adaptive per-token ranks
        2. Auxiliary losses at fixed ranks (low, mid, high)
        3. Rank regularization to control average compression

        This ensures good performance across all compression levels,
        enabling flexible deployment without retraining.
        """
        input_ids = batch['input_ids'].to(self.device)
        labels = input_ids.clone()

        # CRITICAL: Use autocast for mixed precision
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            # Forward pass (main loss with adaptive ranks)
            outputs = self.model(input_ids=input_ids, labels=labels)
            main_loss = outputs.loss

            # CRITICAL: Extract loss and delete outputs immediately to save memory
            main_loss_value = main_loss.clone()
            del outputs
            torch.cuda.empty_cache()

            loss_breakdown = {'main': main_loss_value.item()}

            if not self.args.enable_multiscale_loss:
                return main_loss_value, loss_breakdown

            # INNOVATION: Multi-scale loss across different rank levels
            # This is the key theoretical advantage over Dobi-SVD
            # CRITICAL: Use main_loss_value's device for multi-GPU compatibility
            multiscale_loss = torch.tensor(0.0, device=main_loss_value.device, dtype=main_loss_value.dtype)
            multiscale_count = 0

            # IMPROVED: Use stratified sampling to ensure coverage of all rank levels
            # Train on multiple fixed ranks per batch: r_min, r_max, and 1-2 random intermediate
            # Frequency increased to 50% for better multi-scale learning
            if torch.rand(1).item() < 0.5:  # 50% chance (increased from 20%)
                # CRITICAL: Always include r_min and r_max to ensure boundary performance
                # Add 1 random intermediate rank for diversity
                sampled_ranks = [
                    self.args.r_min,  # Always train at minimum rank
                    self.args.r_max,  # Always train at maximum rank
                    torch.randint(self.args.r_min + 1, self.args.r_max, (1,)).item()  # Random mid rank
                ]

                # Process each sampled rank
                for sampled_rank in sampled_ranks:
                    # Temporarily set all layers to use this fixed rank
                    # by modifying importance to achieve desired rank
                    target_importance = (sampled_rank - self.args.r_min) / (self.args.r_max - self.args.r_min)

                    # Forward pass with fixed rank
                    # We do this by monkey-patching the importance predictor
                    original_forwards = {}
                    for name, module in self.model.named_modules():
                        if isinstance(module, MatryoshkaSVDLayer):
                            predictor = module.importance_predictor
                            original_forwards[name] = predictor.forward

                            # Replace forward with fixed importance
                            def fixed_importance_forward(x, attention_scores=None,
                                                       target_imp=target_importance):
                                batch, seq_len = x.shape[0], x.shape[1]
                                return torch.full((batch, seq_len), target_imp,
                                                device=x.device, dtype=x.dtype)

                            predictor.forward = fixed_importance_forward

                    # CRITICAL: Clear cache before forward pass
                    torch.cuda.empty_cache()

                    # Forward pass with fixed rank
                    outputs_fixed = self.model(input_ids=input_ids, labels=labels)
                    fixed_rank_loss = outputs_fixed.loss.clone()

                    # CRITICAL: Delete outputs immediately
                    del outputs_fixed
                    torch.cuda.empty_cache()

                    # Restore original forwards
                    for name, module in self.model.named_modules():
                        if isinstance(module, MatryoshkaSVDLayer):
                            if name in original_forwards:
                                module.importance_predictor.forward = original_forwards[name]

                    # CRITICAL: Move to same device as multiscale_loss (for multi-GPU)
                    multiscale_loss += fixed_rank_loss.to(multiscale_loss.device)
                    multiscale_count += 1
                    loss_breakdown[f'rank_{sampled_rank}'] = fixed_rank_loss.item()

                    # CRITICAL: Delete intermediate tensors
                    del fixed_rank_loss

            # Regularization losses
            rank_reg_loss = self.compute_rank_regularization()
            spectral_reg_loss = self.compute_spectral_regularization()
            loss_breakdown['rank_reg'] = rank_reg_loss.item()
            loss_breakdown['spectral_reg'] = spectral_reg_loss.item()

            # Total loss
            # Weight: main (1.0) + multiscale (avg 0.3 per rank) + rank_reg + spectral_reg
            # CRITICAL: Move all losses to main_loss_value's device (for multi-GPU)
            # IMPROVED: Normalize multiscale_loss by count to maintain consistent weighting
            if multiscale_count > 0:
                # Average the multiscale loss across all sampled ranks
                avg_multiscale_loss = multiscale_loss.to(main_loss_value.device) / multiscale_count
                total_loss = (main_loss_value +
                             0.3 * avg_multiscale_loss +
                             rank_reg_loss.to(main_loss_value.device) +
                             spectral_reg_loss.to(main_loss_value.device))
            else:
                total_loss = (main_loss_value +
                             rank_reg_loss.to(main_loss_value.device) +
                             spectral_reg_loss.to(main_loss_value.device))

        return total_loss, loss_breakdown

    def compute_rank_regularization(self):
        """Compute rank regularization loss to encourage compression."""
        rank_reg_loss = 0.0
        count = 0

        # CRITICAL: Get target device for loss accumulation (first parameter's device)
        # This handles multi-GPU scenarios where layers are on different devices
        target_device = next(self.model.parameters()).device

        for module in self.model.modules():
            if isinstance(module, MatryoshkaSVDLayer):
                # CRITICAL: Move each layer's loss to target device before summing
                # When model is distributed across GPUs, layers may be on different devices
                layer_loss = module.get_rank_regularization_loss()
                rank_reg_loss += layer_loss.to(target_device)
                count += 1

        if count > 0:
            rank_reg_loss /= count

        return rank_reg_loss

    def compute_spectral_regularization(self):
        """Compute spectral regularization loss for learnable singular values."""
        spectral_reg_loss = 0.0
        count = 0

        # Get target device
        target_device = next(self.model.parameters()).device

        for module in self.model.modules():
            if isinstance(module, MatryoshkaSVDLayer):
                layer_loss = module.get_spectral_regularization_loss()
                spectral_reg_loss += layer_loss.to(target_device)
                count += 1

        if count > 0:
            spectral_reg_loss /= count

        return spectral_reg_loss

    def train_epoch(self, epoch):
        """Train for one epoch with gradient accumulation."""
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.num_epochs}")

        # MEMORY OPTIMIZATION: Gradient accumulation
        accumulation_steps = self.args.gradient_accumulation_steps
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            # Compute loss
            loss, loss_breakdown = self.compute_multiscale_loss(batch)

            # Scale loss by accumulation steps
            loss = loss / accumulation_steps

            # CRITICAL: Backward with mixed precision
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Optimizer step every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                if self.scaler is not None:
                    # Unscale gradients for clipping
                    self.scaler.unscale_(self.optimizer)

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                # Optimizer step with scaler
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()

                # CRITICAL: Clear cache more aggressively
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Logging
            total_loss += loss.item() * batch['input_ids'].size(0)
            total_samples += batch['input_ids'].size(0)

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })

            # Periodic detailed logging
            if (batch_idx + 1) % self.args.log_interval == 0:
                print(f"\nStep {batch_idx+1}/{len(self.train_loader)}:")
                print(f"  Loss breakdown: {loss_breakdown}")
                self.print_rank_statistics()

        # CRITICAL: Clear cache at end of epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        avg_loss = total_loss / total_samples
        return avg_loss

    def evaluate(self):
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        # CRITICAL: Clear cache before evaluation to free memory from training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.eval_loader, desc="Evaluating")):
                input_ids = batch['input_ids'].to(self.device)
                labels = input_ids.clone()

                # Use autocast for evaluation too
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs = self.model(input_ids=input_ids, labels=labels)
                    loss = outputs.loss

                total_loss += loss.item() * input_ids.size(0)
                total_samples += input_ids.size(0)

                # CRITICAL: Clear cache after each batch to prevent fragmentation
                if torch.cuda.is_available():
                    del input_ids, labels, outputs, loss
                    torch.cuda.empty_cache()

        # Clear cache after evaluation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()

        avg_loss = total_loss / total_samples
        perplexity = torch.exp(torch.tensor(avg_loss))

        return avg_loss, perplexity.item()

    def print_rank_statistics(self):
        """Print rank usage statistics across all layers."""
        print("\n" + "="*60)
        print("RANK STATISTICS")
        print("="*60)

        total_avg_rank = 0.0
        count = 0

        for name, module in self.model.named_modules():
            if isinstance(module, MatryoshkaSVDLayer):
                avg_rank = module.avg_rank_tracker.item()
                compression = module.get_compression_ratio()
                total_avg_rank += avg_rank
                count += 1

                print(f"{name}:")
                print(f"  Avg rank: {avg_rank:.1f} / {module.r_max}")
                print(f"  Compression: {compression:.1%}")

        if count > 0:
            print(f"\nOverall average rank: {total_avg_rank/count:.1f}")

    def train(self):
        """Main training loop."""
        print("\n" + "="*80)
        print("STEP 5: Training")
        print("="*80)

        best_eval_loss = float('inf')

        for epoch in range(self.args.num_epochs):
            print(f"\n{'='*80}")
            print(f"Epoch {epoch+1}/{self.args.num_epochs}")
            print(f"{'='*80}")

            # Train
            train_loss = self.train_epoch(epoch)
            print(f"\nTrain loss: {train_loss:.4f}")

            # CRITICAL: Clear cache and run garbage collection before evaluation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()

            # Evaluate
            eval_loss, eval_ppl = self.evaluate()
            print(f"Eval loss: {eval_loss:.4f}, Perplexity: {eval_ppl:.2f}")

            # Print rank statistics
            self.print_rank_statistics()

            # Save checkpoint if best
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                self.save_checkpoint(epoch, eval_loss, eval_ppl, is_best=True)

            # Save periodic checkpoint
            if (epoch + 1) % self.args.save_interval == 0:
                self.save_checkpoint(epoch, eval_loss, eval_ppl, is_best=False)

        print("\n" + "="*80)
        print("Training complete!")
        print(f"Best eval loss: {best_eval_loss:.4f}")
        print("="*80)

    def save_checkpoint(self, epoch, eval_loss, eval_ppl, is_best=False):
        """Save model checkpoint."""
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'eval_loss': eval_loss,
            'eval_ppl': eval_ppl,
            'args': vars(self.args)
        }

        if is_best:
            checkpoint_path = output_dir / 'best_checkpoint.pt'
            print(f"\n💾 Saving best checkpoint to {checkpoint_path}")
        else:
            checkpoint_path = output_dir / f'checkpoint_epoch{epoch+1}.pt'
            print(f"\n💾 Saving checkpoint to {checkpoint_path}")

        torch.save(checkpoint, checkpoint_path)

        # Save rank statistics
        rank_stats = {}
        for name, module in self.model.named_modules():
            if isinstance(module, MatryoshkaSVDLayer):
                rank_stats[name] = {
                    'avg_rank': module.avg_rank_tracker.item(),
                    'r_min': module.r_min,
                    'r_max': module.r_max,
                    'compression': module.get_compression_ratio()
                }

        stats_path = output_dir / ('best_rank_stats.json' if is_best else f'rank_stats_epoch{epoch+1}.json')
        with open(stats_path, 'w') as f:
            json.dump(rank_stats, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description='Train Matryoshka SVD compressed models')

    # Model arguments
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-hf',
                       help='Model name or path')
    parser.add_argument('--dataset', type=str, default='wikitext2',
                       choices=['wikitext2', 'c4', 'ptb'], help='Dataset name')

    # Matryoshka SVD arguments
    parser.add_argument('--r_max', type=int, default=64,
                       help='Maximum rank (default: 64 for memory efficiency, use 128 for 80GB GPU, 256 for A100)')
    parser.add_argument('--r_min', type=int, default=16,
                       help='Minimum rank (default: 16, typically r_max/8 to r_max/4)')
    parser.add_argument('--importance_strategy', type=str, default='norm',
                       choices=['norm', 'learned', 'attention'],
                       help='Importance computation strategy')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Soft truncation temperature')
    parser.add_argument('--soft_truncation_margin', type=float, default=2.0,
                       help='Soft truncation margin (higher = sharper transition)')

    # Enhanced features
    parser.add_argument('--learnable_singular_values', action='store_true',
                       help='Enable learnable singular values for fine-tuning')
    parser.add_argument('--activation_aware_init', action='store_true',
                       help='Use activation-aware SVD initialization (requires calibration data)')
    parser.add_argument('--calibration_samples', type=int, default=128,
                       help='Number of samples for calibration data collection (default: 128)')
    parser.add_argument('--auto_rank_range', action='store_true',
                       help='Automatically determine r_min and r_max based on spectral energy')
    parser.add_argument('--energy_threshold_low', type=float, default=0.90,
                       help='Energy threshold for r_min (default: 90%%)')
    parser.add_argument('--energy_threshold_high', type=float, default=0.99,
                       help='Energy threshold for r_max (default: 99%%)')
    parser.add_argument('--spectral_reg_weight', type=float, default=0.0001,
                       help='Spectral regularization weight (for learnable S)')

    parser.add_argument('--target_layers', type=str, default='q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj',
                       help='Comma-separated layer patterns to compress')
    parser.add_argument('--target_compression', type=float, default=0.15,
                       help='Target compression ratio')

    # Training arguments
    parser.add_argument('--enable_multiscale_loss', action='store_true',
                       help='Enable multi-scale training loss')
    parser.add_argument('--rank_reg_weight', type=float, default=0.001,
                       help='Rank regularization weight')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size per GPU (keep at 1 for memory efficiency)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                       help='Gradient accumulation steps (effective batch size = batch_size * accumulation). Higher value = less memory')
    parser.add_argument('--seq_len', type=int, default=512,
                       help='Sequence length (lower = less memory)')
    parser.add_argument('--num_epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--use_differentiated_lr', action='store_true',
                       help='Use different LRs for importance predictors vs other params')
    parser.add_argument('--importance_lr', type=float, default=1e-3,
                       help='Learning rate for importance predictors')
    parser.add_argument('--other_lr', type=float, default=1e-4,
                       help='Learning rate for other parameters')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='Max gradient norm for clipping')

    # Dataset arguments (from Dobi-SVD)
    parser.add_argument('--n_train_samples', type=int, default=256,
                       help='Number of samples used for training')
    parser.add_argument('--n_eval_samples', type=int, default=256,
                       help='Number of samples used for evaluation')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed')
    parser.add_argument('--SAVE', action='store_true', default=False,
                       help='Whether to save the generated dataset')
    parser.add_argument('--RECREATE', action='store_true', default=False,
                       help='Whether to regenerate the dataset')
    parser.add_argument('--DO_SAMPLE', action='store_true', default=False,
                       help='Whether to obtain the dataset by sampling')
    parser.add_argument('--path_head_folder', type=str, default='./',
                       help='Path of the model and dataset cache')

    # System arguments
    parser.add_argument('--output_dir', type=str, default='./output_matryoshka',
                       help='Output directory')
    parser.add_argument('--log_interval', type=int, default=100,
                       help='Logging interval (steps)')
    parser.add_argument('--save_interval', type=int, default=1,
                       help='Save interval (epochs)')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of data loading workers (0 to avoid forking issues)')
    parser.add_argument('--use_fp16', action='store_true', default=True,
                       help='Use FP16 mixed precision training (enabled by default for memory efficiency)')
    parser.add_argument('--no_device_map', action='store_true',
                       help='Disable automatic device mapping')

    return parser.parse_args()


def main():
    args = parse_args()

    # Create trainer
    trainer = MatryoshkaSVDTrainer(args)

    # Prepare model and data
    trainer.prepare_model()
    trainer.prepare_data()
    trainer.setup_training()

    # Train
    trainer.train()


if __name__ == '__main__':
    main()
