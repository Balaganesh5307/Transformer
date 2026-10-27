"""
Evaluation Script for Transformer Next-Token Prediction.

Evaluates the trained model on the held-out test partition and exports:
- Test loss
- Test perplexity (exp(test_loss))
- Token-level next-token accuracy
- Parameter counts and dataset split metrics
- JSON metrics export to results/metrics/test_metrics.json
"""

import os
import sys

# Ensure transformers root directory is in sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import math
import json
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional
try:
    from models.transformer_model import TransformerLM
    from src.tokenizer import SimpleTokenizer, PAD_ID
except (ImportError, ModuleNotFoundError):
    from transformers.models.transformer_model import TransformerLM
    from transformers.src.tokenizer import SimpleTokenizer, PAD_ID


def evaluate_test_set(
    model: TransformerLM,
    test_loader: torch.utils.data.DataLoader,
    tokenizer: SimpleTokenizer,
    device: torch.device
) -> Dict[str, Any]:
    """
    Evaluate trained model on the held-out test partition.

    Args:
        model: Trained TransformerLM instance.
        test_loader: DataLoader for held-out test data.
        tokenizer: Tokenizer with matching vocabulary.
        device: Evaluation torch device.

    Returns:
        Dict of quantitative test metrics.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction="sum")

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits, _ = model(inputs)

            # Mask out padding tokens
            valid_mask = (targets != tokenizer.pad_token_id)
            valid_targets = targets[valid_mask]
            valid_logits = logits[valid_mask]

            if valid_targets.numel() > 0:
                loss = criterion(valid_logits, valid_targets)
                total_loss += loss.item()

                preds = torch.argmax(valid_logits, dim=-1)
                total_correct += (preds == valid_targets).sum().item()
                total_tokens += valid_targets.numel()

    avg_test_loss = (total_loss / total_tokens) if total_tokens > 0 else 0.0
    perplexity = math.exp(avg_test_loss) if avg_test_loss < 20 else float("inf")
    test_accuracy = (total_correct / total_tokens) if total_tokens > 0 else 0.0

    metrics = {
        "test_loss": round(avg_test_loss, 4),
        "test_perplexity": round(perplexity, 4),
        "test_token_accuracy": round(test_accuracy, 4),
        "test_token_accuracy_pct": round(test_accuracy * 100, 2),
        "total_evaluated_tokens": total_tokens,
        "total_correct_tokens": total_correct,
        "vocabulary_size": tokenizer.vocab_size,
        "model_parameters": model.count_parameters()["total"],
        "trainable_parameters": model.count_parameters()["trainable"]
    }

    return metrics


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device
) -> Tuple[TransformerLM, SimpleTokenizer, Dict[str, Any]]:
    """
    Load Transformer model and tokenizer directly from a checkpoint file.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Reconstruct tokenizer
    tokenizer = SimpleTokenizer()
    tokenizer.token2id = checkpoint["token2id"]
    tokenizer.id2token = {int(v): k for k, v in checkpoint["token2id"].items()}

    # Reconstruct model from saved configuration
    cfg = checkpoint["model_config"]
    model = TransformerLM(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        max_seq_len=cfg["max_seq_len"],
        d_ff=cfg["d_ff"],
        dropout=cfg["dropout"],
        tie_weights=cfg["tie_weights"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, tokenizer, checkpoint
