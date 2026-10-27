"""
Dataset Preparation and DataLoaders for Causal Next-Token Prediction.

Key Concepts:
- Deterministic sentence-level splitting prevents data leakage.
- Input sequences:  [token_0, token_1, ..., token_{T-1}]
- Target sequences: [token_1, token_2, ..., token_T]
- Padding tokens (<PAD>) are aligned and masked out in loss computation via ignore_index.
"""

import os
import random
from typing import List, Tuple, Dict
import torch
from torch.utils.data import Dataset, DataLoader
try:
    from src.tokenizer import SimpleTokenizer, PAD_ID
except (ImportError, ModuleNotFoundError):
    from transformers.src.tokenizer import SimpleTokenizer, PAD_ID


def load_and_split_corpus(
    corpus_path: str,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    Read corpus and partition into train, val, test splits deterministically.

    Args:
        corpus_path: Path to raw corpus text file.
        train_ratio: Fraction for training (default 0.80).
        val_ratio: Fraction for validation (default 0.10).
        seed: Random seed for reproducibility.

    Returns:
        (train_lines, val_lines, test_lines)
    """
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    with open(corpus_path, "r", encoding="utf-8") as f:
        lines = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]

    # Deterministic shuffle by sentence lines BEFORE sequence extraction
    rng = random.Random(seed)
    shuffled_lines = list(lines)
    rng.shuffle(shuffled_lines)

    total = len(shuffled_lines)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_lines = shuffled_lines[:train_end]
    val_lines = shuffled_lines[train_end:val_end]
    test_lines = shuffled_lines[val_end:]

    return train_lines, val_lines, test_lines


class CausalTextDataset(Dataset):
    """
    PyTorch Dataset generating input-target token pairs for causal language modeling.
    """

    def __init__(
        self,
        texts: List[str],
        tokenizer: SimpleTokenizer,
        seq_len: int = 32,
        stride: int = 16
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []

        self._build_samples(texts, stride)

    def _build_samples(self, texts: List[str], stride: int):
        """Convert sentences into aligned (input_ids, target_ids) tensors."""
        for text in texts:
            # Encode sentence with <BOS> and <EOS>
            token_ids = self.tokenizer.encode(text, add_bos=True, add_eos=True)

            if len(token_ids) < 2:
                continue

            # If shorter than or equal to seq_len + 1: single padded window
            if len(token_ids) <= self.seq_len + 1:
                input_seq = token_ids[:-1]
                target_seq = token_ids[1:]

                # Pad to fixed sequence length
                pad_len = self.seq_len - len(input_seq)
                if pad_len > 0:
                    input_seq = input_seq + [self.tokenizer.pad_token_id] * pad_len
                    target_seq = target_seq + [self.tokenizer.pad_token_id] * pad_len

                self.samples.append((
                    torch.tensor(input_seq, dtype=torch.long),
                    torch.tensor(target_seq, dtype=torch.long)
                ))
            else:
                # Sliding window for longer sequences
                for start_idx in range(0, len(token_ids) - 1, stride):
                    chunk = token_ids[start_idx : start_idx + self.seq_len + 1]
                    if len(chunk) < 2:
                        continue

                    input_seq = chunk[:-1]
                    target_seq = chunk[1:]

                    pad_len = self.seq_len - len(input_seq)
                    if pad_len > 0:
                        input_seq = input_seq + [self.tokenizer.pad_token_id] * pad_len
                        target_seq = target_seq + [self.tokenizer.pad_token_id] * pad_len

                    self.samples.append((
                        torch.tensor(input_seq, dtype=torch.long),
                        torch.tensor(target_seq, dtype=torch.long)
                    ))

                    if start_idx + self.seq_len + 1 >= len(token_ids):
                        break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


def get_dataloaders(
    train_lines: List[str],
    val_lines: List[str],
    test_lines: List[str],
    tokenizer: SimpleTokenizer,
    batch_size: int = 16,
    seq_len: int = 32
) -> Dict[str, DataLoader]:
    """Create Train, Val, and Test DataLoaders."""
    train_ds = CausalTextDataset(train_lines, tokenizer, seq_len=seq_len)
    val_ds = CausalTextDataset(val_lines, tokenizer, seq_len=seq_len)
    test_ds = CausalTextDataset(test_lines, tokenizer, seq_len=seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds
    }
