#!/usr/bin/env python3
"""
Training script with improved strategy to break loss plateau.

This script uses:
1. Parameter group learning rates (rank predictor: 1e-4, U/V: 1e-6, other: 5e-5)
2. Progressive rank training (curriculum learning)

Expected: Loss should decrease from 7.0 to ~4.2-4.5
"""

import argparse
import torch
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    default_data_collator
)
from datasets import load_dataset
from improved_training_strategy import (
    create_optimizer_with_param_groups,
    ProgressiveRankTrainer
)
from matryoshka_model_utils import load_matryoshka_model


class ImprovedMatryoshkaTrainer(Trainer):
    """Custom Trainer with parameter groups and progressive rank."""

    def __init__(
        self,
        *args,
        optimizer=None,
        progressive_trainer=None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.custom_optimizer = optimizer
        self.progressive_trainer = progressive_trainer
        self._epoch_started = False

    def create_optimizer(self):
        """Use custom optimizer with parameter groups."""
        if self.custom_optimizer is not None:
            self.optimizer = self.custom_optimizer
            return self.optimizer
        return super().create_optimizer()

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        """Hook to update progressive rank at epoch boundaries."""
        current_epoch = int(epoch) if epoch is not None else 0

        # Update rank range at the start of each new epoch
        if self.progressive_trainer is not None and not self._epoch_started:
            r_min, r_max = self.progressive_trainer.update_model_rank_range(current_epoch)
            print(f"\n{'='*80}")
            print(f"Epoch {current_epoch}: Updated rank range to [{r_min}, {r_max}]")
            print(f"{'='*80}\n")
            self._epoch_started = True

        # Reset flag when epoch changes
        if self.state.epoch != current_epoch:
            self._epoch_started = False

        return super()._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)


def main():
    parser = argparse.ArgumentParser()

    # Model arguments
    parser.add_argument("--matryoshka_model_path", type=str, required=True,
                        help="Path to matryoshka checkpoint")
    parser.add_argument("--base_model_name", type=str, default="meta-llama/Llama-2-7b-hf",
                        help="Base model name")

    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./matryoshka_improved",
                        help="Output directory")
    parser.add_argument("--num_train_epochs", type=int, default=5,
                        help="Number of epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation")

    # Learning rate arguments
    parser.add_argument("--rank_predictor_lr", type=float, default=1e-4,
                        help="Learning rate for rank predictor (large)")
    parser.add_argument("--uv_lr", type=float, default=1e-6,
                        help="Learning rate for U/V projections (small)")
    parser.add_argument("--other_lr", type=float, default=5e-5,
                        help="Learning rate for other parameters (medium)")

    # Progressive training
    parser.add_argument("--use_progressive_rank", action="store_true",
                        help="Enable progressive rank training")
    parser.add_argument("--r_max", type=int, default=512,
                        help="Maximum rank")
    parser.add_argument("--r_min", type=int, default=256,
                        help="Minimum rank (final target)")

    # Dataset
    parser.add_argument("--dataset_name", type=str, default="wikitext",
                        help="Dataset name")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1",
                        help="Dataset config")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="Max sequence length")

    args = parser.parse_args()

    print("="*80)
    print("Training with Improved Strategy to Break Loss Plateau")
    print("="*80)
    print(f"\nStrategy:")
    print(f"  1. Parameter Group Learning Rates:")
    print(f"     - Rank Predictor: {args.rank_predictor_lr:.0e} (learn fast)")
    print(f"     - U/V Projections: {args.uv_lr:.0e} (fine-tune slowly)")
    print(f"     - Other Params: {args.other_lr:.0e} (medium)")
    print(f"  2. Progressive Rank Training: {args.use_progressive_rank}")
    if args.use_progressive_rank:
        print(f"     - Start from high ranks, gradually compress")
        print(f"     - Final target: [{args.r_min}, {args.r_max}]")
    print(f"\nExpected: Loss should decrease from ~7.0 to ~4.2-4.5")
    print("="*80 + "\n")

    # Load model
    print("Loading matryoshka model...")
    model = load_matryoshka_model(
        matryoshka_model_path=args.matryoshka_model_path,
        base_model_name_or_path=args.base_model_name,
        device_map="auto"
    )
    print(f"✅ Model loaded\n")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"✅ Tokenizer loaded\n")

    # Load dataset
    print(f"Loading dataset: {args.dataset_name}/{args.dataset_config}...")
    dataset = load_dataset(args.dataset_name, args.dataset_config)

    def tokenize_function(examples):
        outputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_seq_length,
            padding="max_length",
            return_tensors=None
        )
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs

    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing"
    )

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    print(f"✅ Dataset loaded")
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Eval samples: {len(eval_dataset)}\n")

    # Create optimizer with parameter groups
    print("Creating optimizer with parameter groups...")
    optimizer = create_optimizer_with_param_groups(
        model,
        rank_predictor_lr=args.rank_predictor_lr,
        uv_lr=args.uv_lr,
        other_lr=args.other_lr
    )
    print(f"✅ Optimizer created with 3 parameter groups\n")

    # Create progressive rank trainer if enabled
    progressive_trainer = None
    if args.use_progressive_rank:
        print("Setting up progressive rank training...")
        progressive_trainer = ProgressiveRankTrainer(
            model=model,
            r_max=args.r_max,
            r_min=args.r_min,
            num_epochs=args.num_train_epochs
        )
        print(f"✅ Progressive rank trainer created")
        print(f"   Will gradually compress from r_min={int(0.9*args.r_max)} to {args.r_min}\n")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        logging_steps=10,
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        dataloader_drop_last=True,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    # Create trainer
    print("Creating trainer...")
    trainer = ImprovedMatryoshkaTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        optimizer=optimizer,
        progressive_trainer=progressive_trainer,
    )
    print(f"✅ Trainer created\n")

    # Train
    print("="*80)
    print("Starting training...")
    print("="*80)
    print("\nMonitor the following:")
    print("  - Loss should drop below 7.0 within first epoch")
    print("  - Target final loss: 4.2-4.5")
    print("  - If progressive rank enabled, watch rank range updates")
    print("\n" + "="*80 + "\n")

    trainer.train()

    # Save final model
    print("\n" + "="*80)
    print("Training completed!")
    print("="*80)

    final_output_dir = f"{args.output_dir}/final_model"
    trainer.save_model(final_output_dir)
    print(f"\n✅ Final model saved to: {final_output_dir}")

    # Print final stats
    if trainer.state.log_history:
        final_train_loss = [x for x in trainer.state.log_history if 'loss' in x][-1]['loss']
        print(f"\nFinal training loss: {final_train_loss:.4f}")

        if final_train_loss < 5.0:
            print("✅ Success! Loss decreased significantly from ~7.0")
        elif final_train_loss < 6.0:
            print("⚠️  Loss improved but still high. Consider:")
            print("   - Increasing num_train_epochs")
            print("   - Using knowledge distillation (see TRAINING_STRATEGIES_TO_BREAK_PLATEAU.md)")
        else:
            print("❌ Loss still at plateau. Try:")
            print("   - Knowledge distillation (best solution)")
            print("   - Check if rank predictor is learning (should see rank changes)")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
