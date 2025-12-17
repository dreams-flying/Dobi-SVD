#!/usr/bin/env python3
"""
Check if the model was actually trained or just initialized.
"""

import torch
from matryoshka_model_utils import load_matryoshka_model
import sys

print("="*80)
print("Checking if Model was Actually Trained")
print("="*80)

# Load model
print("\nLoading model...")
model, tokenizer = load_matryoshka_model(
    checkpoint_path="matryoshka_output0/final",
    device='cuda',
    torch_dtype=torch.float32
)

# Test on real text
print("\n" + "="*80)
print("Testing on REAL text (not random tokens)")
print("="*80)

test_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a subset of artificial intelligence.",
    "The weather is nice today.",
]

model.eval()
total_loss = 0
total_tokens = 0

with torch.no_grad():
    for text in test_texts:
        # Tokenize
        inputs = tokenizer(text, return_tensors='pt').to('cuda')
        input_ids = inputs['input_ids']

        # Forward pass
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

        ppl = torch.exp(loss).item()

        print(f"\nText: {text}")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  PPL: {ppl:.4f}")

        total_loss += loss.item() * input_ids.size(1)
        total_tokens += input_ids.size(1)

avg_loss = total_loss / total_tokens
avg_ppl = torch.exp(torch.tensor(avg_loss)).item()

print(f"\n{'='*80}")
print(f"Average PPL on real text: {avg_ppl:.4f}")
print(f"{'='*80}")

if avg_ppl > 10000:
    print("\n❌ CONCLUSION: Model was NEVER trained!")
    print("   PPL > 10000 on real text = random predictions")
    print("\n   Possible causes:")
    print("   1. Checkpoint is from epoch 0 (before training)")
    print("   2. Training script has a bug and didn't actually train")
    print("   3. Training diverged/failed but checkpoint was still saved")
elif avg_ppl > 1000:
    print("\n❌ CONCLUSION: Model training FAILED badly")
    print("   PPL > 1000 = very poor performance")
elif avg_ppl > 100:
    print("\n⚠️  WARNING: Model training was incomplete or poor")
    print("   PPL > 100 = needs more training")
else:
    print("\n✅ Model appears to be trained")
    print("   PPL < 100 = reasonable performance")

# Check training metadata if exists
import json
from pathlib import Path

checkpoint_path = Path("matryoshka_output0/final")
trainer_state_file = checkpoint_path / "trainer_state.json"

if trainer_state_file.exists():
    print(f"\n{'='*80}")
    print("Training Metadata Found")
    print("="*80)

    with open(trainer_state_file, 'r') as f:
        trainer_state = json.load(f)

    print(f"\nEpoch: {trainer_state.get('epoch', 'unknown')}")
    print(f"Global step: {trainer_state.get('global_step', 'unknown')}")

    if 'log_history' in trainer_state:
        log_history = trainer_state['log_history']
        if log_history:
            print(f"\nTraining log entries: {len(log_history)}")

            # Show first few and last few logs
            print(f"\nFirst log entry:")
            print(f"  {log_history[0]}")

            if len(log_history) > 1:
                print(f"\nLast log entry:")
                print(f"  {log_history[-1]}")

                # Check if loss decreased
                if 'loss' in log_history[0] and 'loss' in log_history[-1]:
                    initial_loss = log_history[0]['loss']
                    final_loss = log_history[-1]['loss']

                    print(f"\nLoss comparison:")
                    print(f"  Initial loss: {initial_loss:.4f}")
                    print(f"  Final loss: {final_loss:.4f}")
                    print(f"  Change: {final_loss - initial_loss:.4f}")

                    if final_loss >= initial_loss:
                        print(f"  ❌ Loss did NOT decrease - training failed!")
                    else:
                        print(f"  ✅ Loss decreased - training happened")
else:
    print(f"\n⚠️  No trainer_state.json found - cannot verify training history")

print(f"\n{'='*80}")
print("Diagnostic Complete")
print("="*80)
