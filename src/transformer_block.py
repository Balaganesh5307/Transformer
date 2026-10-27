"""
Transformer Decoder Block using Custom Causal Self-Attention.

Architecture (Pre-LayerNorm Transformer Block):
    Input x
      ↓
    LayerNorm_1
      ↓
    Causal Self-Attention
      ↓
    + Residual Connection (x + attn_output)
      ↓
    LayerNorm_2
      ↓
    Feed-Forward Network (FFN)
      ↓
    + Residual Connection (x + ffn_output)
      ↓
    Output
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
try:
    from src.self_attention import CausalSelfAttention
except (ImportError, ModuleNotFoundError):
    from transformers.src.self_attention import CausalSelfAttention


class FeedForwardNetwork(nn.Module):
    """
    Two-layer position-wise Feed-Forward Network (FFN).

    FFN(x) = GELU(x * W1 + b1) * W2 + b2
    Expansion factor is typically 4 (d_ff = 4 * d_model).
    """

    def __init__(self, d_model: int = 128, d_ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, d_model) -> (B, T, d_ff) -> (B, T, d_model)
        return self.dropout(self.fc2(self.activation(self.fc1(x))))


class TransformerBlock(nn.Module):
    """
    Standard Pre-LN Causal Transformer Decoder Block.

    Pre-LN stabilizes gradients across multiple layers, allowing robust training
    without requiring aggressive learning rate warmup.
    """

    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        max_seq_len: int = 64,
        d_ff: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        # Pre-attention Layer Normalization
        self.ln1 = nn.LayerNorm(d_model)
        # Custom Causal Self-Attention
        self.attn = CausalSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout=dropout
        )
        # Pre-FFN Layer Normalization
        self.ln2 = nn.LayerNorm(d_model)
        # Feed-Forward Network
        self.ffn = FeedForwardNetwork(d_model=d_model, d_ff=d_ff, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with Pre-LayerNorm and residual connections.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            padding_mask: Optional key padding mask
            return_attention: Whether to return self-attention weights

        Returns:
            x: Processed tensor of shape (batch_size, seq_len, d_model)
            attn_weights: Optional attention weights
        """
        # Sublayer 1: Masked Causal Self-Attention + Residual
        normed_x1 = self.ln1(x)
        attn_out, attn_weights = self.attn(
            normed_x1,
            padding_mask=padding_mask,
            return_attention=return_attention
        )
        x = x + attn_out

        # Sublayer 2: Feed-Forward Network + Residual
        normed_x2 = self.ln2(x)
        ffn_out = self.ffn(normed_x2)
        x = x + ffn_out

        return x, attn_weights
