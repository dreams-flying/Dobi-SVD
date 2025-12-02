#!/usr/bin/env python3
"""
Fine-tuning Tool for Optimized Models

Performs lightweight fine-tuning after compression to recover quality.

Usage:
    python tools/finetune_optimized.py \
        --model_path results/pruned_model \
        --n_epochs 3 \
        --lr 1e-5 \
        --output results/finetuned_model
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import argparse
import os
import sys
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset


def prepare_dataset(tokenizer, dataset_name='wikitext', split='train', max_samples=5000, seq_len=512):
    """Prepare training dataset."""
    print(f"Loading dataset: {dataset_name}")

    if dataset_name == 'wikitext':
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
    else:
        dataset = load_dataset('json', data_files=dataset_name, split='train')

    # Limit samples
    if max_samples is not None and len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))

    print(f"Dataset size: {len(dataset)} examples")

    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=seq_len,
            padding='max_length',
            return_tensors='pt'
        )

    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )

    return tokenized


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    """Train for one epoch."""
    model.train()

    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')

    for batch in progress_bar:
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # Forward
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids
        )

        loss = outputs.loss

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        # Update progress bar
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def evaluate(model, dataloader, device):
    """Evaluate model."""
    model.eval()

    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )

            total_loss += outputs.loss.item() * input_ids.numel()
            total_tokens += input_ids.numel()

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return avg_loss, perplexity


def main():
    parser = argparse.ArgumentParser(description='Fine-tune optimized model')

    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model to fine-tune')
    parser.add_argument('--n_epochs', type=int, default=3,
                       help='Number of epochs (default 3)')
    parser.add_argument('--lr', type=float, default=1e-5,
                       help='Learning rate (default 1e-5)')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size (default 4)')
    parser.add_argument('--max_samples', type=int, default=5000,
                       help='Max training samples (default 5000)')
    parser.add_argument('--seq_len', type=int, default=512,
                       help='Sequence length (default 512)')
    parser.add_argument('--train_data', type=str, default='wikitext',
                       help='Training dataset (default wikitext)')
    parser.add_argument('--val_data', type=str, default=None,
                       help='Validation dataset (optional)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for fine-tuned model')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (default cuda)')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                       help='Warmup ratio (default 0.1)')

    args = parser.parse_args()

    print("="*70)
    print("Fine-tuning Tool")
    print("="*70)
    print(f"Model: {args.model_path}")
    print(f"Epochs: {args.n_epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max samples: {args.max_samples}")
    print(f"Output: {args.output}")
    print("="*70)

    # Load model and tokenizer
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = model.to(args.device)

    # Prepare datasets
    print("\nPreparing datasets...")
    train_dataset = prepare_dataset(
        tokenizer,
        dataset_name=args.train_data,
        split='train',
        max_samples=args.max_samples,
        seq_len=args.seq_len
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    # Prepare validation dataset if provided
    val_dataloader = None
    if args.val_data:
        val_dataset = prepare_dataset(
            tokenizer,
            dataset_name=args.val_data,
            split='validation',
            max_samples=500,
            seq_len=args.seq_len
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False
        )

    # Setup optimizer and scheduler
    print("\nSetting up optimizer...")
    optimizer = AdamW(model.parameters(), lr=args.lr)

    total_steps = len(train_dataloader) * args.n_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    print(f"Total training steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")

    # Training loop
    print("\n" + "="*70)
    print("Starting fine-tuning")
    print("="*70)

    best_loss = float('inf')
    training_history = []

    for epoch in range(1, args.n_epochs + 1):
        print(f"\nEpoch {epoch}/{args.n_epochs}")

        # Train
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, args.device, epoch)
        print(f"  Training loss: {train_loss:.4f}")

        # Evaluate
        if val_dataloader:
            val_loss, val_ppl = evaluate(model, val_dataloader, args.device)
            print(f"  Validation loss: {val_loss:.4f}, PPL: {val_ppl:.2f}")

            training_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_ppl': val_ppl
            })

            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                print(f"  ✓ New best model (loss: {val_loss:.4f})")
        else:
            training_history.append({
                'epoch': epoch,
                'train_loss': train_loss
            })

    # Save fine-tuned model
    print(f"\nSaving fine-tuned model to: {args.output}")
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    # Save training history
    import json
    history_path = output_path / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)

    print(f"✓ Training history saved to: {history_path}")

    print("\n" + "="*70)
    print("✓ Fine-tuning complete!")
    print("="*70)

    if val_dataloader:
        print(f"Best validation loss: {best_loss:.4f}")

    print(f"\nFine-tuned model: {args.output}")


if __name__ == '__main__':
    main()
