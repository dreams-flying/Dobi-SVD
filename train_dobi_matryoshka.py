"""
Training script for Dobi-Matryoshka SVD

Based on Dobi-SVD's proven training approach, with Matryoshka multi-scale training added.

Key features:
- Dynamic SVD on activations (from Dobi-SVD)
- Compression regularization (from Dobi-SVD)
- Multi-scale loss (from Matryoshka)
- Uses HuggingFace Trainer (from Dobi-SVD)

Author: Claude
Date: 2025-12-11
"""

import argparse
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from pathlib import Path
import random
import json

from modules.dobi_matryoshka_svd import (
    replace_linear_with_dobi_matryoshka,
    get_compression_regularization,
    DobiMatryoshkaSVDLayer
)
from utils.datautils import prepare_train_loaders


class DobiMatryoshkaTrainer(Trainer):
    """
    Custom Trainer that combines Dobi-SVD compression loss with Matryoshka multi-scale training.
    Includes gamma parameter protection to prevent NaN/Inf.
    """

    def __init__(self, *args, r_max=256, r_min=64, lambda_reg=0.001, multiscale_frequency=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.r_max = r_max
        self.r_min = r_min
        self.lambda_reg = lambda_reg
        self.multiscale_frequency = multiscale_frequency

    def training_step(self, model, inputs):
        """
        Override training_step to add post-optimizer gamma validation.

        Note: Gradient clipping is handled by gradient hooks in the layer itself,
        which run during backward pass before optimizer step.
        This method just validates gamma after optimizer update.
        """
        # Perform normal training step (includes forward, backward, optimizer step)
        loss = super().training_step(model, inputs)

        # Post-optimizer validation: ensure gamma stays in valid range
        # This is a safety net - gradient hooks should prevent issues
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, DobiMatryoshkaSVDLayer):
                    # Clamp gamma to valid range [r_min, r_max]
                    module.gamma.data.clamp_(module.r_min, module.r_max)

                    # Emergency check for NaN/Inf (should be very rare with hooks)
                    if torch.isnan(module.gamma).any() or torch.isinf(module.gamma).any():
                        print(f"EMERGENCY: Detected NaN/Inf in gamma after optimizer step")
                        print(f"  Layer: {module.name if hasattr(module, 'name') and module.name else 'unknown'}")
                        print(f"  Resetting to midpoint: {(module.r_min + module.r_max) / 2.0}")
                        module.gamma.data.fill_((module.r_min + module.r_max) / 2.0)

        return loss

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute loss with compression regularization and multi-scale training.
        """
        # Multi-scale training (Matryoshka)
        use_multiscale = random.random() < self.multiscale_frequency

        if use_multiscale and self.model.training:
            # Sample ranks for multi-scale training
            sampled_ranks = [
                self.r_min,
                self.r_max,
                random.randint(self.r_min + 1, self.r_max - 1)
            ]

            # Compute loss at each rank
            total_loss = 0
            for rank in sampled_ranks:
                # Set all layers to use this rank
                for module in model.modules():
                    if isinstance(module, DobiMatryoshkaSVDLayer):
                        module.fixed_rank_override = float(rank)

                # Forward pass
                outputs = model(**inputs)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                total_loss += loss

                # Clear override
                for module in model.modules():
                    if isinstance(module, DobiMatryoshkaSVDLayer):
                        module.fixed_rank_override = None

            main_loss = total_loss / len(sampled_ranks)
        else:
            # Regular forward
            outputs = model(**inputs)
            main_loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]

        # Compression regularization (from Dobi-SVD)
        reg_loss = get_compression_regularization(model)

        # Total loss
        total_loss = main_loss + self.lambda_reg * reg_loss

        return (total_loss, outputs) if return_outputs else total_loss


def main():
    parser = argparse.ArgumentParser(description='Dobi-Matryoshka SVD Training')

    # Model arguments
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--dataset', type=str, default='wikitext2')

    # Compression arguments
    parser.add_argument('--r_max', type=int, default=256)
    parser.add_argument('--r_min', type=int, default=64)
    parser.add_argument('--beta', type=float, default=10.0, help='Truncation sharpness')
    parser.add_argument('--target_layers', type=str,
                       default='q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj')

    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--warmup_steps', type=int, default=100)

    # Matryoshka arguments
    parser.add_argument('--multiscale_frequency', type=float, default=0.5,
                       help='Frequency of multi-scale training (0.5 = 50%%)')

    # Dobi-SVD regularization
    parser.add_argument('--lambda_reg', type=float, default=0.001,
                       help='Weight for compression regularization')

    # Data arguments
    parser.add_argument('--seq_len', type=int, default=256)
    parser.add_argument('--n_train_samples', type=int, default=256)
    parser.add_argument('--n_eval_samples', type=int, default=128)

    # Other arguments
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--use_fp16', action='store_true', default=True)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='./output_dobi_matryoshka')

    # Dobi-SVD data arguments
    parser.add_argument('--path_head_folder', type=str, default='./')
    parser.add_argument('--SAVE', action='store_true', default=False)
    parser.add_argument('--RECREATE', action='store_true', default=False)

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print("\n" + "="*80)
    print("Dobi-Matryoshka SVD Training")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Rank range: [{args.r_min}, {args.r_max}]")

    # Load model and tokenizer
    print("\nLoading model...")
    # Load model with memory optimization
    # Strategy: Load in FP16 to save memory, use gradient checkpointing
    # Trainer will handle FP32 master weights internally
    print("\nLoading model with memory optimization...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,  # FP16 to save memory (~14GB vs ~28GB)
        device_map=None,
        low_cpu_mem_usage=True
    )

    # Enable gradient checkpointing to save activation memory
    # This trades computation for memory (recomputes activations during backward)
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("✓ Gradient checkpointing enabled (saves activation memory)")

    # Move to GPU
    model = model.cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    original_params = sum(p.numel() for p in model.parameters())
    print(f"Original parameters: {original_params:,}")

    # Replace with Dobi-Matryoshka layers
    print("\n" + "="*80)
    print("Applying Dobi-Matryoshka SVD")
    print("="*80)

    target_layers = args.target_layers.split(',') if args.target_layers else None
    model = replace_linear_with_dobi_matryoshka(
        model,
        target_layers=target_layers,
        r_max=args.r_max,
        r_min=args.r_min,
        beta=args.beta,
        verbose=True
    )

    # Prepare data
    print("\n" + "="*80)
    print("Preparing datasets")
    print("="*80)

    path_head_folder = Path(args.path_head_folder)
    data_cache_dir = path_head_folder / "data_cache"
    dataset_cache_dir = path_head_folder / "dataset_cache"
    data_cache_dir.mkdir(parents=True, exist_ok=True)
    dataset_cache_dir.mkdir(parents=True, exist_ok=True)

    # Returns lists of dicts (not DataLoaders)
    tokenized_traindata, tokenized_valdata = prepare_train_loaders(
        tokenizer=tokenizer,
        DATASET_NAME=args.dataset,
        data_cache_dir=data_cache_dir,
        dataset_cache_dir=dataset_cache_dir,
        args=args
    )

    print(f"Train samples: {len(tokenized_traindata)}")
    print(f"Val samples: {len(tokenized_valdata)}")

    # Data collator (from Dobi-SVD)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Training arguments
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure mixed precision training
    # BF16 is preferred over FP16 as it doesn't require gradient scaling
    # and works better with custom layers
    use_bf16 = torch.cuda.is_bf16_supported() if args.use_fp16 else False
    use_fp16 = args.use_fp16 and not use_bf16

    if use_bf16:
        print("✓ Using BF16 mixed precision (no gradient scaling needed)")
    elif use_fp16:
        print("✓ Using FP16 mixed precision (with gradient scaling)")
    else:
        print("✓ Using FP32 training")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        warmup_steps=args.warmup_steps,
        logging_steps=args.log_interval,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=use_bf16,  # Prefer BF16 over FP16
        fp16=use_fp16,  # Use FP16 only if BF16 not available
        gradient_checkpointing=True,  # Enable to save memory
        remove_unused_columns=False,
        dataloader_drop_last=False,
        report_to="none"
    )

    # Create trainer
    trainer = DobiMatryoshkaTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_traindata,
        eval_dataset=tokenized_valdata,
        data_collator=data_collator,
        r_max=args.r_max,
        r_min=args.r_min,
        lambda_reg=args.lambda_reg,
        multiscale_frequency=args.multiscale_frequency
    )

    # Save config
    config = {
        "model": args.model,
        "dataset": args.dataset,
        "r_max": args.r_max,
        "r_min": args.r_min,
        "beta": args.beta,
        "target_layers": args.target_layers,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "lambda_reg": args.lambda_reg,
        "multiscale_frequency": args.multiscale_frequency,
    }

    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Train
    print("\n" + "="*80)
    print("Starting training")
    print("="*80)

    trainer.train()

    # Evaluate at different ranks
    print("\n" + "="*80)
    print("Multi-rank evaluation")
    print("="*80)

    eval_ranks = [
        ('r_min', args.r_min),
        ('r_mid', (args.r_min + args.r_max) // 2),
        ('r_max', args.r_max),
        ('adaptive', None)
    ]

    results = {}
    for rank_name, rank_value in eval_ranks:
        print(f"\nEvaluating at {rank_name}" + (f" (rank={rank_value})" if rank_value else " (learned gamma)"))

        # Set fixed rank or use learned gamma
        if rank_value is not None:
            for module in model.modules():
                if isinstance(module, DobiMatryoshkaSVDLayer):
                    module.fixed_rank_override = float(rank_value)

        # Evaluate
        metrics = trainer.evaluate()

        # Clear override
        if rank_value is not None:
            for module in model.modules():
                if isinstance(module, DobiMatryoshkaSVDLayer):
                    module.fixed_rank_override = None

        ppl = torch.exp(torch.tensor(metrics['eval_loss'])).item()
        results[rank_name] = {'loss': metrics['eval_loss'], 'ppl': ppl}

        print(f"  Loss: {metrics['eval_loss']:.4f}, PPL: {ppl:.2f}")

    # Save results
    with open(output_dir / "final_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*80)
    print("Training complete!")
    print(f"Results saved to {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
