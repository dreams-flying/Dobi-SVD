#!/usr/bin/env python3
"""
Training script using advanced compression strategies (NO knowledge distillation).

Recommended combination:
1. Self-distillation (full rank as teacher)
2. Two-stage training
3. Layer-wise adaptive rank
4. Learned temperature scheduling

Expected: Loss 7.0 → 3.5-4.0 (comparable to distillation, no extra memory!)
"""

import argparse
import torch
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    default_data_collator
)
from datasets import load_dataset

from advanced_compression_strategies import (
    SelfDistillationLoss,
    TwoStageTrainer,
    LayerWiseRankConfig,
    LearnedTemperatureScheduler,
    ActivationReconstructionLoss,
    create_advanced_trainer,
)
from matryoshka_model_utils import load_matryoshka_model
from improved_training_strategy import create_optimizer_with_param_groups


def main():
    parser = argparse.ArgumentParser()

    # Model arguments
    parser.add_argument("--matryoshka_model_path", type=str, required=True)
    parser.add_argument("--base_model_name", type=str, default="meta-llama/Llama-2-7b-hf")

    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./matryoshka_advanced")
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)

    # Strategy selection
    parser.add_argument("--strategy", type=str, default="self_distill",
                        choices=["self_distill", "recon", "both"],
                        help="Compression strategy: self_distill, recon, or both")
    parser.add_argument("--use_two_stage", action="store_true",
                        help="Use two-stage training (recommended)")
    parser.add_argument("--use_layerwise_rank", action="store_true",
                        help="Use layer-wise adaptive rank")
    parser.add_argument("--use_learned_temp", action="store_true",
                        help="Use learned temperature scheduling")

    # Strategy hyperparameters
    parser.add_argument("--self_distill_alpha", type=float, default=1.0)
    parser.add_argument("--self_distill_beta", type=float, default=0.3)
    parser.add_argument("--recon_weight", type=float, default=0.1)
    parser.add_argument("--initial_tau", type=float, default=5.0)
    parser.add_argument("--final_tau", type=float, default=0.5)

    # Rank configuration
    parser.add_argument("--r_max", type=int, default=512)
    parser.add_argument("--r_min", type=int, default=256)

    # Dataset
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--max_seq_length", type=int, default=512)

    args = parser.parse_args()

    print("=" * 80)
    print("Advanced Compression Training (NO Knowledge Distillation)")
    print("=" * 80)
    print(f"\nStrategies enabled:")
    print(f"  Compression: {args.strategy}")
    print(f"  Two-stage training: {args.use_two_stage}")
    print(f"  Layer-wise rank: {args.use_layerwise_rank}")
    print(f"  Learned temperature: {args.use_learned_temp}")
    print(f"\nExpected: Loss 7.0 → 3.5-4.0")
    print("=" * 80 + "\n")

    # Load model
    print("Loading matryoshka model...")
    model = load_matryoshka_model(
        matryoshka_model_path=args.matryoshka_model_path,
        base_model_name_or_path=args.base_model_name,
        device_map="auto"
    )
    print(f"✅ Model loaded\n")

    # Apply layer-wise rank configuration
    if args.use_layerwise_rank:
        print("Applying layer-wise adaptive rank configuration...")
        num_layers = len([m for m in model.modules() if hasattr(m, 'rank_predictor')])
        layerwise_config = LayerWiseRankConfig(
            num_layers=num_layers,
            global_r_min=args.r_min,
            global_r_max=args.r_max,
        )
        layerwise_config.apply_to_model(model)
        print()

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

    print(f"✅ Dataset loaded ({len(train_dataset)} train, {len(eval_dataset)} eval)\n")

    # Create compression strategy
    print(f"Creating compression strategy: {args.strategy}...")
    if args.strategy == "self_distill":
        compression_strategy = SelfDistillationLoss(
            temperature=2.0,
            alpha=args.self_distill_alpha,
            beta=args.self_distill_beta,
        )
    elif args.strategy == "recon":
        compression_strategy = ActivationReconstructionLoss(
            reconstruction_weight=args.recon_weight,
            match_layers=[8, 16, 24] if num_layers >= 32 else None,
        )
    elif args.strategy == "both":
        # Combine both strategies
        self_distill = SelfDistillationLoss(
            temperature=2.0,
            alpha=args.self_distill_alpha,
            beta=args.self_distill_beta,
        )
        recon_loss = ActivationReconstructionLoss(
            reconstruction_weight=args.recon_weight,
        )

        # Wrapper to use both
        class CombinedStrategy:
            def __init__(self, self_distill, recon_loss):
                self.self_distill = self_distill
                self.recon_loss = recon_loss

            def __call__(self, model, input_ids, attention_mask, labels):
                loss1, dict1 = self.self_distill(model, input_ids, attention_mask, labels)
                loss2, dict2 = self.recon_loss(model, input_ids, attention_mask, labels)

                total_loss = loss1 + loss2
                loss_dict = {**dict1, **dict2, 'loss_combined': total_loss.item()}
                return total_loss, loss_dict

        compression_strategy = CombinedStrategy(self_distill, recon_loss)

    print(f"✅ Compression strategy created\n")

    # Create learned temperature scheduler
    temp_scheduler = None
    if args.use_learned_temp:
        print("Creating learned temperature scheduler...")
        temp_scheduler = LearnedTemperatureScheduler(
            initial_tau=args.initial_tau,
            final_tau=args.final_tau,
            num_epochs=args.num_train_epochs,
            schedule_type='cosine',
        )
        print(f"✅ Temperature: {args.initial_tau} → {args.final_tau}\n")

    # Two-stage training setup
    if args.use_two_stage:
        print("Using two-stage training...")
        two_stage = TwoStageTrainer(
            model=model,
            stage1_epochs=2,
            stage2_epochs=args.num_train_epochs - 2,
        )

        # Stage 1: Train rank predictor only
        print("\n" + "=" * 80)
        print("STAGE 1: Training Rank Predictor Only")
        print("=" * 80 + "\n")

        stage1_optimizer = two_stage.get_stage1_optimizer()

        training_args_stage1 = TrainingArguments(
            output_dir=f"{args.output_dir}/stage1",
            num_train_epochs=2,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            logging_steps=10,
            save_strategy="no",
            fp16=torch.cuda.is_available(),
        )

        from transformers import Trainer

        class Stage1Trainer(Trainer):
            def create_optimizer(self):
                self.optimizer = stage1_optimizer
                return self.optimizer

        trainer_stage1 = Stage1Trainer(
            model=model,
            args=training_args_stage1,
            train_dataset=train_dataset,
            data_collator=default_data_collator,
        )

        trainer_stage1.train()
        print("\n✅ Stage 1 completed\n")

        # Stage 2: Joint training
        print("=" * 80)
        print("STAGE 2: Joint Training (All Parameters)")
        print("=" * 80 + "\n")

        stage2_optimizer = two_stage.get_stage2_optimizer()
        actual_epochs = args.num_train_epochs - 2

    else:
        # Single-stage training with parameter groups
        print("Creating optimizer with parameter groups...")
        stage2_optimizer = create_optimizer_with_param_groups(
            model,
            rank_predictor_lr=1e-4,
            uv_lr=1e-6,
            other_lr=5e-5,
        )
        print(f"✅ Optimizer created\n")
        actual_epochs = args.num_train_epochs

    # Main training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir if not args.use_two_stage else f"{args.output_dir}/stage2",
        num_train_epochs=actual_epochs,
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
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    # Create advanced trainer
    from transformers import Trainer
    AdvancedTrainer = create_advanced_trainer(Trainer, strategy=args.strategy)

    class CustomAdvancedTrainer(AdvancedTrainer):
        def __init__(self, *args, temp_scheduler=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.temp_scheduler = temp_scheduler

        def create_optimizer(self):
            self.optimizer = stage2_optimizer
            return self.optimizer

        def on_epoch_begin(self, args, state, control, **kwargs):
            # Update temperature if using learned scheduling
            if self.temp_scheduler is not None:
                tau, num_updated = self.temp_scheduler.update_model_temperature(
                    self.model, state.epoch
                )
                print(f"\nEpoch {state.epoch:.1f}: Updated temperature τ={tau:.3f} for {num_updated} layers")

            return super().on_epoch_begin(args, state, control, **kwargs)

    trainer = CustomAdvancedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        compression_strategy=compression_strategy,
        temp_scheduler=temp_scheduler,
    )

    # Train
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80 + "\n")

    trainer.train()

    # Save final model
    print("\n" + "=" * 80)
    print("Training Completed!")
    print("=" * 80 + "\n")

    final_output_dir = f"{args.output_dir}/final_model"
    trainer.save_model(final_output_dir)
    print(f"✅ Final model saved to: {final_output_dir}\n")

    # Print final stats
    if trainer.state.log_history:
        final_train_loss = [x for x in trainer.state.log_history if 'loss' in x][-1]['loss']
        print(f"Final training loss: {final_train_loss:.4f}")

        if final_train_loss < 4.0:
            print("✅ Excellent! Loss decreased to < 4.0")
        elif final_train_loss < 5.0:
            print("✅ Good! Loss improved significantly from ~7.0")
        else:
            print("⚠️  Loss still high. Consider combining more strategies.")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
