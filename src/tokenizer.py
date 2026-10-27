"""
Word-Level Tokenizer for Transformer Next-Token Prediction.

Features:
- Normalized regex-based tokenization (words and punctuation separated)
- Special tokens: <PAD>, <UNK>, <BOS>, <EOS>
- Vocabulary construction strictly from training data
- Safe handling of out-of-vocabulary (OOV) tokens via <UNK>
- Serialization / deserialization to JSON
"""

import json
import os
import re
from typing import List, Dict, Optional, Union

# Special token definitions and fixed reserved IDs
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3


class SimpleTokenizer:
    """
    A transparent, educational word-level tokenizer with punctuation preservation.
    """

    def __init__(self):
        self.pad_token = PAD_TOKEN
        self.unk_token = UNK_TOKEN
        self.bos_token = BOS_TOKEN
        self.eos_token = EOS_TOKEN

        self.pad_token_id = PAD_ID
        self.unk_token_id = UNK_ID
        self.bos_token_id = BOS_ID
        self.eos_token_id = EOS_ID

        # Initialize with special tokens
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
        self._init_special_tokens()

        # Tokenization regex pattern: matches alphanumeric sequences or individual punctuation
        self.pattern = re.compile(r"\w+|[^\w\s]")

    def _init_special_tokens(self):
        """Register reserved special tokens at fixed indices."""
        self.token2id = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
        self.id2token = {idx: token for idx, token in enumerate(SPECIAL_TOKENS)}

    @property
    def vocab_size(self) -> int:
        """Return total number of unique tokens in vocabulary."""
        return len(self.token2id)

    def tokenize(self, text: str) -> List[str]:
        """
        Normalize text and extract tokens (words and punctuation).
        
        Args:
            text (str): Raw input sentence.
            
        Returns:
            List[str]: List of lowercased token strings.
        """
        if not text or not isinstance(text, str):
            return []
        normalized = text.lower().strip()
        tokens = self.pattern.findall(normalized)
        return tokens

    def fit(self, texts: List[str], min_freq: int = 1):
        """
        Build vocabulary from a corpus of texts (training split only).
        
        Args:
            texts (List[str]): Sentences or documents from training split.
            min_freq (int): Minimum frequency required to include token.
        """
        self._init_special_tokens()
        freq: Dict[str, int] = {}

        for text in texts:
            for token in self.tokenize(text):
                freq[token] = freq.get(token, 0) + 1

        # Sort by frequency descending, then alphabetically for determinism
        sorted_tokens = sorted(freq.items(), key=lambda x: (-x[1], x[0]))

        current_id = len(SPECIAL_TOKENS)
        for token, count in sorted_tokens:
            if count >= min_freq and token not in self.token2id:
                self.token2id[token] = current_id
                self.id2token[current_id] = token
                current_id += 1

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False
    ) -> List[int]:
        """
        Convert text into a list of integer token IDs.
        
        Args:
            text (str): Input text.
            add_bos (bool): If True, prepend <BOS> token ID.
            add_eos (bool): If True, append <EOS> token ID.
            
        Returns:
            List[int]: Sequence of token IDs.
        """
        tokens = self.tokenize(text)
        ids = [self.token2id.get(token, self.unk_token_id) for token in tokens]

        if add_bos:
            ids = [self.bos_token_id] + ids
        if add_eos:
            ids = ids + [self.eos_token_id]

        return ids

    def decode(
        self,
        ids: List[int],
        skip_special_tokens: bool = True
    ) -> str:
        """
        Convert list of token IDs back into human-readable text.
        
        Args:
            ids (List[int]): Sequence of token IDs.
            skip_special_tokens (bool): If True, exclude <PAD>, <BOS>, <EOS>, <UNK>.
            
        Returns:
            str: Reconstructed text with clean punctuation spacing.
        """
        tokens = []
        for token_id in ids:
            token = self.id2token.get(token_id, self.unk_token)
            if skip_special_tokens and token in SPECIAL_TOKENS:
                continue
            tokens.append(token)

        # Reconstruct sentence with natural punctuation spacing
        text = " ".join(tokens)
        # Detokenize basic punctuation (e.g., "word ." -> "word.")
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)
        text = re.sub(r"\(\s+", "(", text)
        text = re.sub(r"\s+\)", ")", text)
        return text

    def save(self, filepath: str):
        """Serialize vocabulary to JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = {
            "token2id": self.token2id,
            "vocab_size": self.vocab_size,
            "special_tokens": SPECIAL_TOKENS
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str) -> "SimpleTokenizer":
        """Deserialize tokenizer and vocabulary from JSON."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Vocabulary file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokenizer = cls()
        tokenizer.token2id = {k: int(v) for k, v in data["token2id"].items()}
        tokenizer.id2token = {int(v): k for k, v in data["token2id"].items()}
        return tokenizer
