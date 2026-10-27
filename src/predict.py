"""
Next-Token Prediction and Autoregressive Text Generation.

Supports:
- Top-5 candidate token predictions with probability breakdown
- Greedy decoding (argmax)
- Temperature sampling with configurable temperature T
- Multi-step autoregressive generation
- Graceful handling of unknown words and empty input
"""

import os
import sys

# Ensure transformers root directory is in sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from typing import List, Dict, Any, Tuple, Optional
import torch
import torch.nn.functional as F
try:
    from models.transformer_model import TransformerLM
    from src.tokenizer import SimpleTokenizer, BOS_ID, EOS_ID, PAD_ID
except (ImportError, ModuleNotFoundError):
    from transformers.models.transformer_model import TransformerLM
    from transformers.src.tokenizer import SimpleTokenizer, BOS_ID, EOS_ID, PAD_ID


def predict_next_token_distribution(
    model: TransformerLM,
    tokenizer: SimpleTokenizer,
    prompt: str,
    device: torch.device,
    add_bos: bool = True
) -> Tuple[List[Tuple[str, float]], torch.Tensor, List[str]]:
    """
    Given a prompt, predict the probability distribution over the next token.

    Args:
        model: Trained TransformerLM.
        tokenizer: Tokenizer.
        prompt: User input string.
        device: Torch device.
        add_bos: If True, prepends <BOS> token.

    Returns:
        top_candidates: List of (token_str, probability) for the top candidates.
        probs: Full probability tensor of shape (vocab_size,)
        input_tokens: List of token strings in the prompt.
    """
    model.eval()

    # Tokenize prompt
    token_ids = tokenizer.encode(prompt, add_bos=add_bos, add_eos=False)
    input_tokens = [tokenizer.id2token.get(idx, tokenizer.unk_token) for idx in token_ids]

    if len(token_ids) == 0:
        token_ids = [tokenizer.bos_token_id]
        input_tokens = [tokenizer.bos_token]

    # Truncate to max sequence length if input is too long
    if len(token_ids) > model.max_seq_len:
        token_ids = token_ids[-model.max_seq_len:]
        input_tokens = input_tokens[-model.max_seq_len:]

    input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        logits, _ = model(input_tensor)
        # Logits at the very last token position: (1, vocab_size)
        last_token_logits = logits[0, -1, :]
        probs = F.softmax(last_token_logits, dim=-1)

    # Extract Top-5 predictions
    top_k = min(5, tokenizer.vocab_size)
    top_probs, top_indices = torch.topk(probs, k=top_k)

    top_candidates = [
        (tokenizer.id2token.get(idx.item(), tokenizer.unk_token), prob.item())
        for idx, prob in zip(top_indices, top_probs)
    ]

    return top_candidates, probs, input_tokens


def generate_text(
    model: TransformerLM,
    tokenizer: SimpleTokenizer,
    prompt: str,
    max_new_tokens: int = 10,
    temperature: float = 1.0,
    sample: bool = False,
    repetition_penalty: float = 1.15,
    device: Optional[torch.device] = None,
    stop_at_eos: bool = True
) -> Tuple[str, List[str]]:
    """
    Autoregressively generate subsequent tokens given a prompt.

    Args:
        model: Trained TransformerLM.
        tokenizer: Tokenizer.
        prompt: Initial text prompt.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Temperature scaling factor (T > 0).
        sample: If True, uses multinomial sampling; otherwise greedy argmax.
        device: Torch device.
        stop_at_eos: If True, halts generation upon producing <EOS>.

    Returns:
        generated_text: Complete generated string.
        generated_tokens: List of newly generated token strings.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    token_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    if len(token_ids) == 0:
        token_ids = [tokenizer.bos_token_id]

    generated_tokens: List[str] = []

    for _ in range(max_new_tokens):
        # Keep window within max_seq_len
        curr_ids = token_ids[-model.max_seq_len:]
        input_tensor = torch.tensor([curr_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            logits, _ = model(input_tensor)
            next_logits = logits[0, -1, :]

            # Prevent predicting <PAD> or <BOS>
            next_logits[tokenizer.pad_token_id] = float("-inf")
            next_logits[tokenizer.bos_token_id] = float("-inf")

            # Repetition penalty on recently generated tokens (window of 6)
            if repetition_penalty != 1.0 and len(generated_tokens) > 0:
                recent_ids = set(token_ids[-min(6, len(token_ids)):])
                for prev_id in recent_ids:
                    if next_logits[prev_id] > 0:
                        next_logits[prev_id] /= repetition_penalty
                    else:
                        next_logits[prev_id] *= repetition_penalty

            if sample:
                scaled_logits = next_logits / max(temperature, 1e-4)
                probs = F.softmax(scaled_logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).item()
            else:
                next_id = torch.argmax(next_logits).item()

        if stop_at_eos and next_id == tokenizer.eos_token_id:
            break

        token_str = tokenizer.id2token.get(next_id, tokenizer.unk_token)
        generated_tokens.append(token_str)
        token_ids.append(next_id)

    full_text = tokenizer.decode(token_ids, skip_special_tokens=True)
    return full_text, generated_tokens


if __name__ == "__main__":
    import argparse
    from src.evaluate import load_model_from_checkpoint

    parser = argparse.ArgumentParser(description="Transformer Next-Token Prediction")
    parser.add_argument("prompt", nargs="?", default="The sun rises in the", help="Input prompt text")
    parser.add_argument("--tokens", type=int, default=8, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--sample", action="store_true", help="Use sampling instead of greedy argmax")
    args = parser.parse_args()

    checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "checkpoints", "best_transformer.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, _ = load_model_from_checkpoint(checkpoint_path, device)

    top_candidates, _, input_tokens = predict_next_token_distribution(model, tokenizer, args.prompt, device)

    print("\n" + "=" * 60)
    print("        TRANSFORMER NEXT-TOKEN PREDICTION REPORT")
    print("=" * 60)
    print(f"Prompt:        '{args.prompt}'")
    print(f"Input Tokens:  {input_tokens}")
    print("-" * 60)
    print("Top-5 Candidates:")
    for rank, (tok, prob) in enumerate(top_candidates, 1):
        print(f"  {rank}. {tok:<16} : {prob * 100:6.2f}%")
    print("-" * 60)
    print(f"Most Likely Next Token: >>> '{top_candidates[0][0]}' <<<")

    text, gen_tokens = generate_text(
        model, tokenizer, args.prompt,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        sample=args.sample,
        device=device
    )
    print(f"Full Completion:        '{text}'")
    print("=" * 60 + "\n")

