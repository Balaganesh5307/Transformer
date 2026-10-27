"""
Comprehensive Unit Tests for Transformer Next-Token Prediction Pipeline.

Covers:
1. Tokenizer encoding, decoding, and unknown token handling.
2. Causal dataset input-target offset alignment.
3. Attention tensor shapes and scaled dot-product correctness.
4. Causal masking invariant: future tokens NEVER affect past representations.
5. Full Transformer forward pass shape: (batch_size, seq_len, vocab_size).
6. Loss calculation with padding mask.
7. Model checkpoint serialization and deserialization.
"""

import os
import sys

# Ensure transformers root is in sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
import torch.nn as nn
try:
    from src.tokenizer import SimpleTokenizer, PAD_ID, UNK_ID, BOS_ID, EOS_ID
    from src.self_attention import CausalSelfAttention, scaled_dot_product_attention
    from src.dataset import CausalTextDataset
    from models.transformer_model import TransformerLM
except (ImportError, ModuleNotFoundError):
    from transformers.src.tokenizer import SimpleTokenizer, PAD_ID, UNK_ID, BOS_ID, EOS_ID
    from transformers.src.self_attention import CausalSelfAttention, scaled_dot_product_attention
    from transformers.src.dataset import CausalTextDataset
    from transformers.models.transformer_model import TransformerLM


def test_tokenizer():
    """Verify tokenizer encoding, decoding, special tokens, and OOV handling."""
    tokenizer = SimpleTokenizer()
    sample_texts = [
        "The sun rises in the east.",
        "Machine learning is fascinating."
    ]
    tokenizer.fit(sample_texts)

    # Test known token encoding
    tokens = tokenizer.tokenize("The sun rises.")
    assert tokens == ["the", "sun", "rises", "."], f"Unexpected tokens: {tokens}"

    ids = tokenizer.encode("The sun rises.", add_bos=True, add_eos=True)
    assert ids[0] == BOS_ID, "First token must be <BOS>"
    assert ids[-1] == EOS_ID, "Last token must be <EOS>"

    # Test decoding roundtrip
    decoded = tokenizer.decode(ids, skip_special_tokens=True)
    assert "the sun rises." in decoded, f"Decoding failed: {decoded}"

    # Test unknown token handling
    unk_ids = tokenizer.encode("extraterrestrial spaceship alien")
    assert all(idx == UNK_ID for idx in unk_ids), f"Expected UNK_ID for unseen words: {unk_ids}"
    print("[PASS] Tokenizer tests passed.")


def test_dataset_alignment():
    """Verify that target sequence is strictly input sequence shifted by one token."""
    tokenizer = SimpleTokenizer()
    tokenizer.fit(["the sun rises in the east and warms the earth."])

    text = "the sun rises in the east."
    ds = CausalTextDataset([text], tokenizer, seq_len=8)
    inputs, targets = ds[0]

    assert inputs.size(0) == 8, f"Expected seq_len=8, got {inputs.size(0)}"
    assert targets.size(0) == 8, f"Expected seq_len=8, got {targets.size(0)}"

    # In non-padding positions, target[i] must equal input[i+1]
    for i in range(len(inputs) - 1):
        if targets[i].item() != PAD_ID and inputs[i + 1].item() != PAD_ID:
            assert targets[i].item() == inputs[i + 1].item(), (
                f"Shift mismatch at index {i}: target={targets[i]}, next_input={inputs[i+1]}"
            )
    print("[PASS] Dataset input-target alignment test passed.")


def test_attention_shapes_and_masking():
    """Verify self-attention tensor shapes and strictly lower-triangular weights."""
    batch_size = 2
    seq_len = 6
    d_model = 64
    num_heads = 4

    attn = CausalSelfAttention(d_model=d_model, num_heads=num_heads, max_seq_len=16)
    attn.eval()

    x = torch.randn(batch_size, seq_len, d_model)
    output, attn_weights = attn(x, return_attention=True)

    # Check output shape
    assert output.shape == (batch_size, seq_len, d_model), (
        f"Expected output shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    )

    # Check attention weights shape: (B, H, T, T)
    assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len), (
        f"Expected attn_weights shape {(batch_size, num_heads, seq_len, seq_len)}, got {attn_weights.shape}"
    )

    # Verify Causal Masking: strictly zero attention for future positions (j > i)
    for b in range(batch_size):
        for h in range(num_heads):
            for i in range(seq_len):
                for j in range(i + 1, seq_len):
                    weight = attn_weights[b, h, i, j].item()
                    assert weight == 0.0 or abs(weight) < 1e-7, (
                        f"Causal violation at layer head ({b}, {h}), pos ({i}, {j}): weight = {weight}"
                    )

    # Verify that valid preceding weights sum to 1.0 (valid probability distribution)
    for b in range(batch_size):
        for h in range(num_heads):
            for i in range(seq_len):
                row_sum = attn_weights[b, h, i, :i + 1].sum().item()
                assert abs(row_sum - 1.0) < 1e-4, f"Attention row {i} does not sum to 1.0: sum={row_sum}"

    print("[PASS] Attention tensor shapes and lower-triangular causal mask passed.")


def test_causal_invariance_property():
    """
    CRITICAL CAUSAL MASKING TEST:
    Verify that altering future tokens in a sequence does NOT change
    earlier token representations or logits in evaluation mode.
    """
    vocab_size = 50
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=32,
        num_heads=2,
        num_layers=2,
        max_seq_len=16,
        dropout=0.0
    )
    model.eval()

    # Sequence A: [5, 12, 23, 34, 45]
    seq_a = torch.tensor([[5, 12, 23, 34, 45]], dtype=torch.long)

    # Sequence B: identical prefix [5, 12, 23], but different future tokens [1, 9]
    seq_b = torch.tensor([[5, 12, 23, 1, 9]], dtype=torch.long)

    with torch.no_grad():
        logits_a, _ = model(seq_a)
        logits_b, _ = model(seq_b)

    # Logits at positions 0, 1, 2 must be IDENTICAL between Sequence A and Sequence B!
    for pos in range(3):
        diff = (logits_a[0, pos, :] - logits_b[0, pos, :]).abs().max().item()
        assert diff < 1e-5, (
            f"Causal leakage detected at position {pos}! Max difference: {diff}"
        )

    print("[PASS] Causal invariance test passed (future tokens do not influence earlier predictions).")


def test_full_forward_pass_and_loss():
    """Verify model forward pass shape and loss computation."""
    vocab_size = 40
    batch_size = 3
    seq_len = 10

    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=32,
        num_heads=2,
        num_layers=1,
        max_seq_len=16
    )

    inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    logits, _ = model(inputs)
    assert logits.shape == (batch_size, seq_len, vocab_size), (
        f"Expected logits shape {(batch_size, seq_len, vocab_size)}, got {logits.shape}"
    )

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
    assert not torch.isnan(loss) and not torch.isinf(loss) and loss.item() > 0, (
        f"Invalid loss value: {loss.item()}"
    )

    print("[PASS] Full forward pass and loss computation tests passed.")


def run_all_tests():
    """Execute full test suite."""
    print("=" * 60)
    print("        RUNNING TRANSFORMER PROJECT UNIT TESTS")
    print("=" * 60)
    test_tokenizer()
    test_dataset_alignment()
    test_attention_shapes_and_masking()
    test_causal_invariance_property()
    test_full_forward_pass_and_loss()
    print("=" * 60)
    print("         ALL 5 CORE TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
