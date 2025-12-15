"""
Evaluate Matryoshka SVD model at different compression levels.

This script:
1. Loads trained Matryoshka model
2. Evaluates perplexity at different target ranks
3. Measures inference speed
4. Compares with baseline

Author: Claude
Date: 2025-12-15
"""

import argparse
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import time
from typing import Dict, List
import numpy as np
from tqdm import tqdm

from modules.matryoshka_svd_layer import MatryoshkaSVDLayer


def set_model_rank(model: nn.Module, rank: float):
    """Set fixed rank for all MatryoshkaSVDLayers."""
    for module in model.modules():
        if isinstance(module, MatryoshkaSVDLayer):
            module.set_fixed_rank(rank)
        elif hasattr(module, 'use_matryoshka'):
            # For monkey-patched modules
            module.fixed_rank = rank


def compute_perplexity(
    model: nn.Module,
    tokenizer,
    dataset,
    max_samples: int = 100,
    device: str = 'cuda'
) -> float:
    """Compute perplexity on dataset."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for i, example in enumerate(tqdm(dataset, desc="Computing PPL")):
            if i >= max_samples:
                break

            inputs = tokenizer(
                example['text'],
                return_tensors='pt',
                truncation=True,
                max_length=512
            ).to(device)

            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss

            seq_len = inputs['input_ids'].shape[1]
            total_loss += loss.item() * seq_len
            total_tokens += seq_len

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return perplexity


def main():
    parser = argparse.ArgumentParser(description='Evaluate Matryoshka SVD model')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--target_rank', type=int, default=None)
    parser.add_argument('--dataset', type=str, default='wikitext2')
    parser.add_argument('--max_samples', type=int, default=100)
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}")
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint).cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
    
    if args.target_rank:
        print(f"Evaluating at rank = {args.target_rank}")
        set_model_rank(model, args.target_rank)
        ppl = compute_perplexity(model, tokenizer, dataset['test'], args.max_samples)
        print(f"Perplexity: {ppl:.2f}")
    else:
        # Evaluate at multiple ranks
        for rank in [64, 128, 192, 256]:
            print(f"\nEvaluating at rank = {rank}")
            set_model_rank(model, rank)
            ppl = compute_perplexity(model, tokenizer, dataset['test'], args.max_samples)
            print(f"Perplexity: {ppl:.2f}")


if __name__ == '__main__':
    main()
