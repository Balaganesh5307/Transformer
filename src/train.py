"""
Training Pipeline for Transformer Next-Token Prediction.

Features:
- AdamW optimizer with optional weight decay
- CrossEntropyLoss with ignore_index for <PAD> tokens
- Gradient clipping to prevent exploding gradients
- Learning rate scheduler (CosineAnnealing or ReduceLROnPlateau)
- Early stopping based on validation loss
- Comprehensive epoch logging of train loss, val loss, and token-level accuracy
- Checkpoint persistence (weights + config + vocabulary)
"""

import os
import sys

# Ensure transformers root directory is in sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import json
import time
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, List, Optional

from models.transformer_model import TransformerLM  # type: ignore
from src.tokenizer import SimpleTokenizer, PAD_ID    # type: ignore


def calculate_accuracy(logits: torch.Tensor, targets: torch.Tensor, pad_id: int = PAD_ID) -> Tuple[int, int]:
    """
    Calculate token-level next-token prediction accuracy, excluding padding.

    Args:
        logits: Tensor of shape (B, T, V)
        targets: Tensor of shape (B, T)
        pad_id: Padding token ID to ignore

    Returns:
        (num_correct_tokens, num_valid_tokens)
    """
    predictions = torch.argmax(logits, dim=-1)  # (B, T)
    valid_mask = (targets != pad_id)            # (B, T) boolean mask

    valid_predictions = predictions[valid_mask]
    valid_targets = targets[valid_mask]

    correct = (valid_predictions == valid_targets).sum().item()
    total = valid_targets.numel()

    return correct, total


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    clip_grad_norm: float = 1.0,
    pad_id: int = PAD_ID
) -> Tuple[float, float]:
    """Train model for one epoch."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass: logits shape (B, T, V)
        logits, _ = model(inputs)

        # Reshape for CrossEntropyLoss:
        # logits: (B * T, V), targets: (B * T)
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

        # Backward pass
        loss.backward()

        # Gradient clipping
        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

        # Optimizer step
        optimizer.step()

        # Metrics accumulation: exact token-weighted cross-entropy
        correct, count = calculate_accuracy(logits, targets, pad_id=pad_id)
        if count > 0:
            total_loss += loss.item() * count
            total_correct += correct
            total_tokens += count

    avg_loss = (total_loss / total_tokens) if total_tokens > 0 else 0.0
    accuracy = (total_correct / total_tokens) if total_tokens > 0 else 0.0
    return avg_loss, accuracy


def evaluate_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    pad_id: int = PAD_ID
) -> Tuple[float, float]:
    """Evaluate model on validation or test dataset without gradients."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits, _ = model(inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

            # Metrics accumulation: exact token-weighted cross-entropy
            correct, count = calculate_accuracy(logits, targets, pad_id=pad_id)
            if count > 0:
                total_loss += loss.item() * count
                total_correct += correct
                total_tokens += count

    avg_loss = (total_loss / total_tokens) if total_tokens > 0 else 0.0
    accuracy = (total_correct / total_tokens) if total_tokens > 0 else 0.0
    return avg_loss, accuracy


def train_model(
    model: TransformerLM,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    tokenizer: SimpleTokenizer,
    epochs: int = 50,
    lr: float = 0.002,
    weight_decay: float = 0.01,
    device: Optional[torch.device] = None,
    checkpoint_dir: str = "models/checkpoints",
    history_path: str = "results/metrics/history.json",
    patience: int = 15
) -> Dict[str, Any]:
    """
    Execute full training routine with checkpointing and early stopping.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    model = model.to(device)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(history_path)), exist_ok=True)

    # CrossEntropyLoss ignores <PAD> token in target positions
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # Separate parameters for weight decay: exclude LayerNorm and biases
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "ln" in name or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0}
    ], lr=lr, betas=(0.9, 0.98), eps=1e-8)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epochs": [],
        "best_epoch": 0,
        "best_val_loss": float("inf"),
        "training_time_sec": 0.0
    }

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_checkpoint_path = os.path.join(checkpoint_dir, "best_transformer.pt")

    start_time = time.time()
    print("-" * 75)
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Train Acc':^10} | {'Val Loss':^12} | {'Val Acc':^10} | {'LR':^9}")
    print("-" * 75)

    for epoch in range(1, epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            pad_id=tokenizer.pad_token_id
        )

        val_loss, val_acc = evaluate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            pad_id=tokenizer.pad_token_id
        )

        scheduler.step()

        history["epochs"].append(epoch)
        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))

        # Check for best validation loss
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            # Save full checkpoint with model config and state dict
            checkpoint_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "model_config": model.get_config(),
                "vocab_size": tokenizer.vocab_size,
                "token2id": tokenizer.token2id
            }
            torch.save(checkpoint_data, best_checkpoint_path)
            best_marker = " * BEST"
        else:
            patience_counter += 1
            best_marker = ""

        print(
            f"{epoch:^7d} | {train_loss:^12.4f} | {train_acc * 100:^9.2f}% | "
            f"{val_loss:^12.4f} | {val_acc * 100:^9.2f}% | {current_lr:^9.6f}{best_marker}"
        )

        if patience_counter >= patience:
            print(f"\n[Early Stopping Triggered] No improvement in validation loss for {patience} epochs.")
            break

    total_time = time.time() - start_time
    history["best_epoch"] = best_epoch
    history["best_val_loss"] = round(best_val_loss, 4)
    history["training_time_sec"] = round(total_time, 2)

    # Save history to JSON
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("-" * 75)
    print(f"Training completed in {total_time:.2f} seconds.")
    print(f"Best Model Checkpoint saved to: {best_checkpoint_path}")
    print(f"Best Validation Loss: {best_val_loss:.4f} at Epoch {best_epoch}")
    print("-" * 75)

    return history


if __name__ == "__main__":
    import argparse
    from src.dataset import load_and_split_corpus, get_dataloaders
    from src.utils import set_seed, plot_loss_curves

    parser = argparse.ArgumentParser(
        description="Train Transformer Next-Token Prediction Model (Standalone Entry Point)"
    )
    parser.add_argument("--corpus", type=str, default="data/raw/corpus.txt", help="Path to raw corpus file")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs (default: 50)")
    parser.add_argument("--lr", type=float, default=0.002, help="Learning rate (default: 0.002)")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience (default: 15)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--seq_len", type=int, default=32, help="Sequence length (default: 32)")
    parser.add_argument("--d_model", type=int, default=64, help="Embedding dimension (default: 64)")
    parser.add_argument("--num_heads", type=int, default=4, help="Attention heads (default: 4)")
    parser.add_argument("--num_layers", type=int, default=2, help="Transformer blocks (default: 2)")
    parser.add_argument("--d_ff", type=int, default=256, help="Feed-forward dimension (default: 256)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate (default: 0.1)")
    parser.add_argument("--checkpoint_dir", type=str, default="models/checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--history_path", type=str, default="results/metrics/history.json", help="Path to save history JSON")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    corpus_path = os.path.join(ROOT_DIR, args.corpus) if not os.path.isabs(args.corpus) else args.corpus
    train_lines, val_lines, test_lines = load_and_split_corpus(corpus_path, seed=args.seed)

    tokenizer = SimpleTokenizer()
    tokenizer.fit(train_lines)

    loaders = get_dataloaders(
        train_lines=train_lines,
        val_lines=val_lines,
        test_lines=test_lines,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        seq_len=args.seq_len
    )

    model = TransformerLM(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_seq_len=args.seq_len,
        d_ff=args.d_ff,
        dropout=args.dropout,
        tie_weights=True
    )

    ckpt_dir = os.path.join(ROOT_DIR, args.checkpoint_dir) if not os.path.isabs(args.checkpoint_dir) else args.checkpoint_dir
    hist_path = os.path.join(ROOT_DIR, args.history_path) if not os.path.isabs(args.history_path) else args.history_path

    history = train_model(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        tokenizer=tokenizer,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        checkpoint_dir=ckpt_dir,
        history_path=hist_path,
        patience=args.patience
    )

    plot_save_path = os.path.join(ROOT_DIR, "results", "plots", "loss_curves.png")
    plot_loss_curves(history, save_path=plot_save_path)
