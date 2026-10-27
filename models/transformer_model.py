"""
Complete Transformer Causal Language Model for Next-Token Prediction.

Architecture Overview:
    Input Token IDs: (Batch, Seq_Len)
            ↓
    [Token Embedding] + [Positional Embedding]
            ↓
       [Dropout]
            ↓
    [Transformer Block 1] (Custom Causal Self-Attention + FFN)
            ↓
    [Transformer Block 2] (Custom Causal Self-Attention + FFN)
            ↓
    [Final LayerNorm]
            ↓
    [Linear LM Head: d_model -> vocab_size]
            ↓
    Output Logits: (Batch, Seq_Len, Vocab_Size)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any
try:
    from src.positional_encoding import LearnablePositionalEmbedding
    from src.transformer_block import TransformerBlock
except (ImportError, ModuleNotFoundError):
    from transformers.src.positional_encoding import LearnablePositionalEmbedding
    from transformers.src.transformer_block import TransformerBlock


class TransformerLM(nn.Module):
    """
    A compact, pedagogical Transformer language model trained for next-token prediction.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        max_seq_len: int = 64,
        d_ff: int = 512,
        dropout: float = 0.1,
        tie_weights: bool = True
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.d_ff = d_ff
        self.dropout_rate = dropout
        self.tie_weights = tie_weights

        # 1. Token Embeddings
        self.tok_embed = nn.Embedding(vocab_size, d_model)

        # 2. Learnable Positional Embeddings
        self.pos_embed = LearnablePositionalEmbedding(max_seq_len=max_seq_len, d_model=d_model)

        # 3. Input Dropout
        self.drop = nn.Dropout(dropout)

        # 4. Stack of Transformer Decoder Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                max_seq_len=max_seq_len,
                d_ff=d_ff,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # 5. Final Layer Normalization
        self.ln_final = nn.LayerNorm(d_model)

        # 6. Language Modeling Head (Linear projection to vocabulary logits)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: share parameters between token embedding and LM head
        # Reduces parameter count and improves generalization in small language models
        if tie_weights:
            self.lm_head.weight = self.tok_embed.weight

        # Parameter initialization (standard normal with small std)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Forward pass for next-token prediction.

        Args:
            input_ids: LongTensor of token IDs, shape (batch_size, seq_len)
            padding_mask: Optional mask for padding tokens
            return_attention: If True, returns attention weights for all layers

        Returns:
            logits: Output logits of shape (batch_size, seq_len, vocab_size)
            attentions: List of attention weight tensors from each block (if return_attention=True)
        """
        batch_size, seq_len = input_ids.size()
        assert seq_len <= self.max_seq_len, (
            f"Sequence length {seq_len} exceeds maximum supported {self.max_seq_len}"
        )

        # Step 1: Compute token embeddings: (B, T) -> (B, T, d_model)
        tok_embeddings = self.tok_embed(input_ids)

        # Step 2: Compute positional embeddings: (1, T, d_model)
        pos_embeddings = self.pos_embed(seq_len=seq_len, device=input_ids.device)

        # Step 3: Combine token and position information and apply dropout
        x = self.drop(tok_embeddings + pos_embeddings)

        # Step 4: Pass through sequential Transformer blocks
        all_attentions = [] if return_attention else None
        for block in self.blocks:
            x, attn_weights = block(
                x,
                padding_mask=padding_mask,
                return_attention=return_attention
            )
            if return_attention and attn_weights is not None:
                all_attentions.append(attn_weights)

        # Step 5: Final layer normalization
        x = self.ln_final(x)

        # Step 6: Project to vocabulary logits: (B, T, d_model) -> (B, T, vocab_size)
        logits = self.lm_head(x)

        return logits, all_attentions

    def count_parameters(self) -> Dict[str, int]:
        """Count trainable and total parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    def get_config(self) -> Dict[str, Any]:
        """Return architecture configuration dictionary."""
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "max_seq_len": self.max_seq_len,
            "d_ff": self.d_ff,
            "dropout": self.dropout_rate,
            "tie_weights": self.tie_weights
        }
