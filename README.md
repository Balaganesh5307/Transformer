# Development of a Transformer-Based Next-Token Prediction Model Using Basic Self-Attention

A complete, self-contained academic deep learning project implementing a causal Transformer language model from scratch in PyTorch. The central scaled dot-product self-attention mechanism, causal autoregressive masking, learnable positional embeddings, and Transformer blocks are built from first principles using tensor operations without relying on prebuilt Transformer libraries or pretrained models.

---

## 1. Abstract
Autoregressive language models predict probability distributions over the next token in a text sequence conditioned on preceding context tokens. Unlike Recurrent Neural Networks (RNNs) that process tokens sequentially, the Transformer architecture utilizes self-attention to dynamically capture syntactic and semantic dependencies across all tokens simultaneously. This project provides a transparent, mathematically rigorous implementation of a causal Transformer for next-token prediction, trained and evaluated on an educational multi-domain corpus.

---

## 2. Problem Statement
Given an input sequence of tokens:

$$X = (x_1, x_2, \dots, x_t)$$

the objective of next-token prediction is to estimate the conditional probability distribution over the vocabulary $V$ for the subsequent token $x_{t+1}$:

$$P(x_{t+1} \mid x_1, x_2, \dots, x_t)$$

Because language semantics depend on both local phrase structure and long-range dependencies, models must:
1. Retain order information without recurrence.
2. Relate any two tokens in $O(1)$ operations regardless of their distance.
3. Prevent information leakage from future tokens ($x_{j > t}$) during training.

---

## 3. Project Objectives
- **Scratch Attention Implementation:** Build Multi-Head Scaled Dot-Product Attention using pure PyTorch tensor operations.
- **Strict Causal Masking:** Enforce lower-triangular attention masking ($M_{i,j} = -\infty$ for $j > i$) to prevent future token lookahead.
- **End-to-End Pipeline:** Provide dataset preparation, custom word-level tokenization, sequence alignment, training with early stopping, quantitative test evaluation, and autoregressive text generation.
- **Explainability:** Extract and visualize real attention weight heatmaps from the trained model.
- **Interactive Execution:** Provide both command-line interactive prediction and an optional Streamlit web interface (`app.py`).

---

## 4. Technology Stack
- **Python 3.12** (Core programming language)
- **PyTorch 2.x** (Tensor computation, automatic differentiation, neural network modules)
- **NumPy** (Numerical operations and array handling)
- **Matplotlib** (Loss curve visualization and attention heatmaps)
- **Pandas** (Tabular metrics and UI data structures)
- **Streamlit** (Interactive web application for next-token inference)

---

## 5. Project Folder Structure

```text
transformers/
├── data/
│   ├── raw/
│   │   └── corpus.txt              # Multi-domain educational training corpus
│   └── processed/
│       └── vocab.json              # Serialized vocabulary (token-to-ID mapping)
├── models/
│   ├── transformer_model.py        # Complete TransformerLM neural network architecture
│   └── checkpoints/
│       └── best_transformer.pt     # Saved best model weights, config, and optimizer state
├── src/
│   ├── __init__.py
│   ├── tokenizer.py                # Custom word-level tokenizer with punctuation support
│   ├── dataset.py                  # Causal sequence alignment and PyTorch DataLoaders
│   ├── positional_encoding.py      # Learnable & sinusoidal positional embeddings
│   ├── self_attention.py           # Core scaled dot-product causal self-attention
│   ├── transformer_block.py        # Pre-LN Transformer decoder blocks (Attn + FFN)
│   ├── train.py                    # Training loop with validation and early stopping
│   ├── evaluate.py                 # Held-out test set evaluation and perplexity calculation
│   ├── predict.py                  # Next-token inference and autoregressive generation
│   └── utils.py                    # Loss plotting, attention visualization, and seed setup
├── results/
│   ├── plots/
│   │   ├── loss_curves.png         # Training vs. validation loss curve
│   │   └── attention_map.png       # Extracted causal attention weight heatmap
│   ├── metrics/
│   │   ├── history.json            # Epoch-by-epoch loss and accuracy history
│   │   └── test_metrics.json       # Quantitative evaluation results on test set
│   └── sample_predictions.txt      # Automated multi-domain next-token prediction report
├── tests/
│   └── test_model.py               # Unit tests (tokenization, shapes, causal invariance)
├── app.py                          # Streamlit interactive web application
├── main.py                         # Unified command-line pipeline runner with dataclass config
├── requirements.txt                # Python package dependencies
├── .gitignore                      # Git exclusion rules
└── README.md                       # Comprehensive project documentation
```

---

## 6. Mathematical Architecture

### Scaled Dot-Product Self-Attention
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$

Where $M$ is the causal mask ensuring autoregressive validity:
$$M_{i,j} = \begin{cases} 0 & \text{if } j \le i \\ -\infty & \text{if } j > i \end{cases}$$

### Architectural Hyperparameters (`TransformerConfig`):
- **Embedding Dimension ($d_{\text{model}}$):** 64
- **Attention Heads ($h$):** 4 ($d_k = 16$ per head)
- **Transformer Decoder Layers:** 2
- **Feed-Forward Dimension ($d_{ff}$):** 256
- **Context Length:** 32 tokens
- **Total Parameters:** 158,336

---

## 7. Performance & Accuracy Comparison

In next-token prediction over a vocabulary of **886 unique tokens**, random guessing yields an expected accuracy of only $1 / 886 \approx 0.11\%$:

| Metric / Parameter | Random Guess Baseline | Initial Model Run (Ep 1) | Present (Trained Model Checkpoint) | Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **Token-Level Test Accuracy** | **0.11%** | **10.45%** | **20.21%** (58 / 287 correct) | **~180× gain over baseline** 📈 |
| **Test Loss (Cross-Entropy)** | $\approx 6.7867$ | 6.5310 | **5.5576** | **-1.2291 loss reduction** 📉 |
| **Test Perplexity ($\exp(\text{Loss})$)**| **886.00** | 686.08 | **259.20** | **-70.7% perplexity reduction** 📉 |
| **Validation Accuracy** | 0.11% | 17.19% | **18.36% – 19.14%** | Generalizes cleanly without collapse |
| **Training Accuracy** | 0.11% | 10.45% | **70.40%** (at Ep 30) | High syntactic convergence |

---
streamlit run app.py
```
Provides a browser interface with interactive Top-5 probability bars, temperature sliders, and 2D causal attention heatmaps.
