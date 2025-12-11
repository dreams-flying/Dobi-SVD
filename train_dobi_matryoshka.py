"""
Training script for Dobi-Matryoshka SVD

Based on Dobi-SVD's proven training approach, with Matryoshka multi-scale training added.

Key features:
- Dynamic SVD on activations (from Dobi-SVD)
- Compression regularization (from Dobi-SVD)
- Multi-scale loss (from Matryoshka)
- Stable training (from Dobi-SVD)

Author: Claude
Date: 2025-12-11
"""

import argparse
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import random
from pathlib import Path

from modules.dobi_matryoshka_svd import (
    replace_linear_with_dobi_matryoshka,
    compute_multiscale_loss,
    get_compression_regularization,
    DobiMatryoshkaSVDLayer
)
from utils.datautils import prepare_train_loaders
from evaluate import evaluate_perplexity


class DobiMatryoshkaTrainer:
    """Trainer combining Dobi-SVD and Matryoshka approaches."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print("\n" + "="*80)
        print("Dobi-Matryoshka SVD Training")
        print("="*80)
        print(f"Model: {args.model}")
        print(f"Dataset: {args.dataset}")
        print(f"Rank range: [{args.r_min}, {args.r_max}]")
        print(f"Device: {self.device}")

        # Load model and tokenizer
        print("\nLoading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map=None,  # Load on CPU first
            low_cpu_mem_usage=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        original_params = sum(p.numel() for p in self.model.parameters())
        print(f"Original parameters: {original_params:,}")

        # Replace with Dobi-Matryoshka layers
        print("\n" + "="*80)
        print("Applying Dobi-Matryoshka SVD")
        print("="*80)

        target_layers = args.target_layers.split(',') if args.target_layers else None
        self.model = replace_linear_with_dobi_matryoshka(
            self.model,
            target_layers=target_layers,
            r_max=args.r_max,
            r_min=args.r_min,
            beta=args.beta,
            verbose=True
        )

        # Move to GPU
        print(f"\nMoving model to {self.device}...")
        self.model = self.model.to(self.device)

        # Prepare data
        print("\n" + "="*80)
        print("Preparing datasets")
        print("="*80)

        self.train_loader, self.val_loader = prepare_train_loaders(
            dataset_name=args.dataset,
            tokenizer=self.tokenizer,
            seqlen=args.seq_len,
            nsamples_train=args.n_train_samples,
            nsamples_val=args.n_eval_samples,
            seed=args.seed,
            batch_size=args.batch_size
        )

        print(f"Train samples: {args.n_train_samples}")
        print(f"Val samples: {args.n_eval_samples}")

        # Setup training
        self.setup_training()

    def setup_training(self):
        """Setup optimizer, scheduler, etc."""
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )

        # Scheduler
        total_steps = len(self.train_loader) * self.args.num_epochs // self.args.gradient_accumulation_steps
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=total_steps
        )

        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if self.args.use_fp16 else None

        print("\nTraining setup:")
        print(f"  Total steps: {total_steps}")
        print(f"  Warmup steps: {self.args.warmup_steps}")
        print(f"  Learning rate: {self.args.learning_rate}")
        print(f"  Weight decay: {self.args.weight_decay}")
        print(f"  Gradient accumulation: {self.args.gradient_accumulation_steps}")
        print(f"  Mixed precision: {self.args.use_fp16}")

    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_main_loss = 0
        total_reg_loss = 0
        total_samples = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.num_epochs}")

        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch.get('labels', input_ids).to(self.device)

            # === Multi-scale training (Matryoshka) ===
            use_multiscale = random.random() < self.args.multiscale_frequency

            if use_multiscale:
                # Sample ranks for multi-scale training
                sampled_ranks = [
                    self.args.r_min,
                    self.args.r_max,
                    random.randint(self.args.r_min + 1, self.args.r_max - 1)
                ]

                # Multi-scale forward
                def forward_fn(model, input_ids, labels):
                    with torch.cuda.amp.autocast(enabled=self.args.use_fp16):
                        outputs = model(input_ids=input_ids, labels=labels)
                        return outputs.loss

                with torch.cuda.amp.autocast(enabled=self.args.use_fp16):
                    main_loss = compute_multiscale_loss(
                        self.model,
                        forward_fn,
                        sampled_ranks,
                        input_ids,
                        labels
                    )
            else:
                # Regular forward
                with torch.cuda.amp.autocast(enabled=self.args.use_fp16):
                    outputs = self.model(input_ids=input_ids, labels=labels)
                    main_loss = outputs.loss

            # === Compression regularization (Dobi-SVD) ===
            reg_loss = get_compression_regularization(self.model)

            # Total loss
            loss = main_loss + self.args.lambda_reg * reg_loss

            # Backward
            if self.scaler is not None:
                self.scaler.scale(loss / self.args.gradient_accumulation_steps).backward()
            else:
                (loss / self.args.gradient_accumulation_steps).backward()

            # Optimizer step
            if (batch_idx + 1) % self.args.gradient_accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()

            # Logging
            total_loss += loss.item()
            total_main_loss += main_loss.item()
            total_reg_loss += reg_loss.item()
            total_samples += 1

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'main': f"{main_loss.item():.4f}",
                'reg': f"{reg_loss.item():.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })

            # Periodic logging
            if (batch_idx + 1) % self.args.log_interval == 0:
                avg_loss = total_loss / total_samples
                avg_main = total_main_loss / total_samples
                avg_reg = total_reg_loss / total_samples

                # Get average gamma
                avg_gamma = self.get_average_gamma()

                print(f"\n  Step {batch_idx+1}/{len(self.train_loader)}")
                print(f"    Avg loss: {avg_loss:.4f}")
                print(f"    Main loss: {avg_main:.4f}")
                print(f"    Reg loss: {avg_reg:.4f}")
                print(f"    Avg gamma: {avg_gamma:.1f}")

        return total_loss / total_samples

    def get_average_gamma(self) -> float:
        """Get average gamma across all layers."""
        gammas = []
        for module in self.model.modules():
            if isinstance(module, DobiMatryoshkaSVDLayer):
                gammas.append(module.gamma.item())

        return sum(gammas) / len(gammas) if gammas else 0

    def evaluate(self) -> dict:
        """Evaluate at multiple ranks."""
        self.model.eval()
        results = {}

        # Evaluate at r_min, r_mid, r_max, and adaptive
        eval_ranks = [
            ('r_min', self.args.r_min),
            ('r_mid', (self.args.r_min + self.args.r_max) // 2),
            ('r_max', self.args.r_max),
            ('adaptive', None)  # Use learned gamma
        ]

        for rank_name, rank_value in eval_ranks:
            print(f"\nEvaluating at {rank_name}" + (f" (rank={rank_value})" if rank_value else " (learned gamma)"))

            # Set fixed rank or use learned gamma
            if rank_value is not None:
                for module in self.model.modules():
                    if isinstance(module, DobiMatryoshkaSVDLayer):
                        module.fixed_rank_override = float(rank_value)

            # Evaluate
            total_loss = 0
            total_samples = 0

            with torch.no_grad():
                for batch in tqdm(self.val_loader, desc=f"Eval {rank_name}"):
                    input_ids = batch['input_ids'].to(self.device)
                    labels = batch.get('labels', input_ids).to(self.device)

                    outputs = self.model(input_ids=input_ids, labels=labels)
                    total_loss += outputs.loss.item()
                    total_samples += 1

            # Clear override
            if rank_value is not None:
                for module in self.model.modules():
                    if isinstance(module, DobiMatryoshkaSVDLayer):
                        module.fixed_rank_override = None

            avg_loss = total_loss / total_samples
            ppl = torch.exp(torch.tensor(avg_loss)).item()

            results[rank_name] = {'loss': avg_loss, 'ppl': ppl}
            print(f"  Loss: {avg_loss:.4f}, PPL: {ppl:.2f}")

        return results

    def train(self):
        """Main training loop."""
        print("\n" + "="*80)
        print("Starting training")
        print("="*80)

        best_ppl = float('inf')

        for epoch in range(self.args.num_epochs):
            print(f"\n{'='*80}")
            print(f"Epoch {epoch+1}/{self.args.num_epochs}")
            print(f"{'='*80}")

            # Train
            train_loss = self.train_epoch(epoch)
            print(f"\nEpoch {epoch+1} train loss: {train_loss:.4f}")

            # Evaluate
            results = self.evaluate()

            # Save if best
            adaptive_ppl = results['adaptive']['ppl']
            if adaptive_ppl < best_ppl:
                best_ppl = adaptive_ppl
                if self.args.save_model:
                    self.save_model(epoch, adaptive_ppl)

        print("\n" + "="*80)
        print("Training complete!")
        print(f"Best PPL (adaptive): {best_ppl:.2f}")
        print("="*80)

    def save_model(self, epoch: int, ppl: float):
        """Save model checkpoint."""
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        save_path = output_dir / f"epoch_{epoch+1}_ppl_{ppl:.2f}.pt"

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'ppl': ppl,
            'args': vars(self.args)
        }, save_path)

        print(f"\n✅ Model saved to {save_path}")


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
                       help='Frequency of multi-scale training (0.5 = 50%)')

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
    parser.add_argument('--save_model', action='store_true')

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Train
    trainer = DobiMatryoshkaTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
