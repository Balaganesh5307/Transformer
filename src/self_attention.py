"""
Causal Self-Attention Mechanism Implemented from Scratch with PyTorch.

Mathematical Formulation:
    Attention(Q, K, V) = softmax( (Q * K^T / sqrt(d_k)) + M ) * V

Where:
    - Q (Queries): Linear projection of input tokens searching for relevant context.
    - K (Keys):    Linear projection of input tokens describing their own content.
    - V (Values):  Linear projection of input tokens holding semantic representation.
    - d_k:         Dimensionality of each attention head (scaling factor prevents vanishing gradients).
    - M:           Causal mask: Upper-triangular matrix set to -infinity, ensuring token at
                   position t can only attend to tokens at positions <= t.
"""

import math
import torch
import torch.nn as nn
from typing import Tuple, Optional


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Dropout] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pure tensor computation of Scaled Dot-Product Attention with causal masking.

    Args:
        q: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
        k: Key tensor of shape (batch_size, num_heads, seq_len, head_dim)
        v: Value tensor of shape (batch_size, num_heads, seq_len, head_dim)
        mask: Optional boolean or float mask of shape (1, 1, seq_len, seq_len)
              where False (or 0) indicates positions to block (-inf).
        dropout: Optional nn.Dropout module applied to attention weights.

    Returns:
        output: Aggregated context tensor of shape (batch_size, num_heads, seq_len, head_dim)
        attention_weights: Normalized attention probability matrix (batch_size, num_heads, seq_len, seq_len)
    """
    # Key dimension d_k used for scaling
    d_k = q.size(-1)

    # Step 1: Compute raw similarity scores Q * K^T
    # q: (B, H, T, d_k), k^T: (B, H, d_k, T) -> scores: (B, H, T, T)
    scores = torch.matmul(q, k.transpose(-2, -1))

    # Step 2: Scale scores by 1 / sqrt(d_k)
    # Scaling maintains variance near 1.0, preventing softmax saturation & vanishing gradients
    scaled_scores = scores / math.sqrt(d_k)

    # Step 3: Apply Causal Mask (prevent future token visibility)
    if mask is not None:
        # Fill masked positions (where mask == 0) with a very large negative number (-1e9)
        # In softmax, exp(-inf) = 0.0, ensuring zero attention to future tokens
        scaled_scores = scaled_scores.masked_fill(mask == 0, float("-inf"))

    # Step 4: Softmax normalization over the key dimension (last dimension)
    # Ensures attention weights across valid preceding positions sum to 1.0
    attention_weights = torch.softmax(scaled_scores, dim=-1)

    # Handle edge case where entire row is masked (e.g. all padding) by replacing NaNs with 0
    if torch.isnan(attention_weights).any():
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)

    # Step 5: Optional dropout on attention probabilities
    weights_dropped = dropout(attention_weights) if dropout is not None else attention_weights

    # Step 6: Weighted sum of Value vectors
    # weights: (B, H, T, T), v: (B, H, T, d_k) -> output: (B, H, T, d_k)
    output = torch.matmul(weights_dropped, v)

    return output, attention_weights


class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention Layer built from first principles.

    Supports:
    - Independent linear projections for Q, K, V
    - Multi-head partitioning for diverse representation subspaces
    - Lower-triangular causal autoregressive mask
    - Final linear projection of concatenated attention heads
    - Extraction of attention weight matrices for explainability & visualization
    """

    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        max_seq_len: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_seq_len = max_seq_len

        # Explicit linear projections: Q, K, V
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        # Output projection back to d_model
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Attention dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Register lower-triangular causal mask buffer (not a trainable parameter)
        # Shape: (1, 1, max_seq_len, max_seq_len)
        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).view(
            1, 1, max_seq_len, max_seq_len
        )
        self.register_buffer("causal_mask", causal_mask)

        # Cache last computed attention weights for visualization
        self.last_attention_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for Multi-Head Causal Self-Attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            padding_mask: Optional boolean mask of shape (batch_size, 1, 1, seq_len)
                          where True = valid token, False = padding token.
            return_attention: If True, returns the attention weight matrix.

        Returns:
            output: Attention output of shape (batch_size, seq_len, d_model)
            attention_weights: Optional attention weights (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.size()
        assert seq_len <= self.max_seq_len, (
            f"Input sequence length ({seq_len}) exceeds max_seq_len ({self.max_seq_len})"
        )

        # Step 1: Project inputs to Queries, Keys, and Values
        # Shape: (B, T, d_model) -> (B, T, d_model)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Step 2: Split into multi-head format and transpose
        # (B, T, d_model) -> (B, T, num_heads, head_dim) -> (B, num_heads, T, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Step 3: Slice causal mask for the current sequence length T
        # Shape: (1, 1, T, T)
        combined_mask = self.causal_mask[:, :, :seq_len, :seq_len]

        # Combine with optional key padding mask if provided
        if padding_mask is not None:
            # padding_mask: (B, 1, 1, T) broadcasts across (B, 1, T, T)
            combined_mask = combined_mask * padding_mask

        # Step 4: Scaled dot-product attention
        # context: (B, H, T, head_dim), attn_weights: (B, H, T, T)
        context, attn_weights = scaled_dot_product_attention(
            q=q,
            k=k,
            v=v,
            mask=combined_mask,
            dropout=self.attn_dropout if self.training else None
        )

        # Cache attention weights
        self.last_attention_weights = attn_weights.detach()

        # Step 5: Concatenate heads back together
        # (B, H, T, head_dim) -> (B, T, H, head_dim) -> (B, T, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Step 6: Final linear output projection and residual dropout
        output = self.resid_dropout(self.out_proj(context))

        if return_attention:
            return output, attn_weights
        return output, None
