"""
Positional Encodings and Positional Embeddings for Transformers.

Why Positional Information is Essential:
Unlike Recurrent Neural Networks (RNNs) which process tokens sequentially step-by-step,
Self-Attention processes all tokens in parallel simultaneously via matrix operations.
Matrix multiplication is permutation-equivariant: without explicit position signals,
a sequence like ["dog", "bites", "man"] and ["man", "bites", "dog"] would produce
identical un-ordered representations. Positional encodings inject spatial coordinates
so the model distinguishes grammatical roles and word order.
"""

import math
import torch
import torch.nn as nn


class LearnablePositionalEmbedding(nn.Module):
    """
    Learnable Positional Embeddings (similar to GPT-2 / BERT).
    
    Each position index [0, 1, 2, ..., max_seq_len - 1] is mapped to a dedicated
    trainable embedding vector of size d_model that updates during backpropagation.
    """

    def __init__(self, max_seq_len: int = 64, d_model: int = 128):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.embedding = nn.Embedding(max_seq_len, d_model)

        # Register position index buffer [0, 1, ..., max_seq_len - 1]
        self.register_buffer("position_ids", torch.arange(max_seq_len))

    def forward(self, seq_len: int, device: torch.device = None) -> torch.Tensor:
        """
        Produce positional embeddings for a sequence of length seq_len.

        Args:
            seq_len: Current length of input token sequence (T <= max_seq_len).
            device: Optional torch device.

        Returns:
            Positional embedding tensor of shape (1, seq_len, d_model).
        """
        assert seq_len <= self.max_seq_len, (
            f"Requested length {seq_len} exceeds max_seq_len {self.max_seq_len}"
        )
        pos_ids = self.position_ids[:seq_len]
        if device is not None:
            pos_ids = pos_ids.to(device)
        # Shape: (T, d_model) -> (1, T, d_model) for broadcasting across batch dimension
        return self.embedding(pos_ids).unsqueeze(0)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed Sinusoidal Positional Encoding from 'Attention Is All You Need' (Vaswani et al., 2017).

    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
    """

    def __init__(self, max_seq_len: int = 64, d_model: int = 128):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Shape: (1, max_seq_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, seq_len: int, device: torch.device = None) -> torch.Tensor:
        """Return sinusoidal positional encoding of shape (1, seq_len, d_model)."""
        tensor = self.pe[:, :seq_len]
        if device is not None:
            tensor = tensor.to(device)
        return tensor
