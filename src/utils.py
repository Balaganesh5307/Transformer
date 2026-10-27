"""
Visualization, Reproducibility, and Logging Utilities.

Includes:
- Combined Training and Validation loss curves
- Attention weight heatmap visualization
- Random seed initialization for full reproducibility
"""

import os
import random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # Headless backend for reliable server execution
import matplotlib.pyplot as plt
from typing import List, Optional


def set_seed(seed: int = 42):
    """Set random seed across Python, NumPy, and PyTorch for deterministic results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot_loss_curves(
    history: dict,
    save_path: str = "results/plots/loss_curves.png"
):
    """
    Generate professional training and validation loss curves.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    epochs = history.get("epochs", list(range(1, len(history["train_loss"]) + 1)))
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    best_epoch = history.get("best_epoch", int(np.argmin(val_loss) + 1))
    best_val_loss = min(val_loss)

    plt.figure(figsize=(9, 5.5), dpi=300)
    plt.plot(epochs, train_loss, label="Training Loss", color="#1f77b4", linewidth=2.2)
    plt.plot(epochs, val_loss, label="Validation Loss", color="#ff7f0e", linewidth=2.2, linestyle="--")

    # Mark best checkpoint
    plt.scatter([best_epoch], [best_val_loss], color="#d62728", s=100, zorder=5,
                label=f"Best Checkpoint (Ep {best_epoch}: {best_val_loss:.4f})")

    plt.title("Transformer Next-Token Prediction: Training vs. Validation Loss", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Cross-Entropy Loss", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11, loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Loss plot saved to: {save_path}")


def plot_attention_map(
    attention_matrix: np.ndarray,
    tokens: List[str],
    layer_idx: int = 0,
    head_idx: int = 0,
    save_path: str = "results/plots/attention_map.png"
):
    """
    Generate and save a heatmap of the causal self-attention weights.

    Args:
        attention_matrix: 2D numpy array of shape (seq_len, seq_len)
        tokens: List of token strings corresponding to rows and columns
        layer_idx: Index of Transformer block layer
        head_idx: Index of attention head
        save_path: Path to output PNG
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    seq_len = len(tokens)
    sub_matrix = attention_matrix[:seq_len, :seq_len]

    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    cax = ax.matshow(sub_matrix, cmap="Blues", vmin=0.0, vmax=1.0)

    # Set axis ticks and labels
    ax.set_xticks(range(seq_len))
    ax.set_yticks(range(seq_len))
    ax.set_xticklabels(tokens, rotation=45, ha="left", fontsize=10)
    ax.set_yticklabels(tokens, fontsize=10)

    # Title & labels
    ax.set_xlabel("Keys (Attended Tokens)", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_ylabel("Queries (Current Tokens)", fontsize=11, fontweight="bold")
    plt.title(f"Causal Self-Attention Weights (Layer {layer_idx + 1}, Head {head_idx + 1})\nLower-Triangular Masking Enforced",
              fontsize=12, fontweight="bold", pad=20)

    # Add numeric values inside cells for clarity
    for i in range(seq_len):
        for j in range(seq_len):
            val = sub_matrix[i, j]
            if val > 0.01:
                color = "white" if val > 0.55 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)

    fig.colorbar(cax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Attention map saved to: {save_path}")
