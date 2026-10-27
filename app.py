"""
Streamlit Web Application for Transformer Next-Token Prediction & Self-Attention Visualization.

Launch with:
    streamlit run app.py
(from inside the transformers/ directory)
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from src.tokenizer import SimpleTokenizer
    from src.evaluate import load_model_from_checkpoint
    from src.predict import predict_next_token_distribution, generate_text
except (ImportError, ModuleNotFoundError):
    from transformers.src.tokenizer import SimpleTokenizer
    from transformers.src.evaluate import load_model_from_checkpoint
    from transformers.src.predict import predict_next_token_distribution, generate_text

# Paths
CHECKPOINT_PATH = os.path.join(BASE_DIR, "models", "checkpoints", "best_transformer.pt")
METRICS_PATH = os.path.join(BASE_DIR, "results", "metrics", "test_metrics.json")
LOSS_PLOT_PATH = os.path.join(BASE_DIR, "results", "plots", "loss_curves.png")

# Page configuration
st.set_page_config(
    page_title="Transformer Next-Token Prediction",
    page_icon="🧠",
    layout="wide"
)


@st.cache_resource
def get_model():
    """Load model and tokenizer once and cache in memory."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(CHECKPOINT_PATH):
        return None, None, None, f"Model checkpoint not found at: {CHECKPOINT_PATH}. Please run training first (python main.py --mode train)."
    try:
        model, tokenizer, checkpoint = load_model_from_checkpoint(CHECKPOINT_PATH, device)
        return model, tokenizer, device, None
    except Exception as e:
        return None, None, None, str(e)


def main():
    st.title("🧠 Transformer Next-Token Prediction Model")
    st.markdown(
        """
        **Academic Deep Learning Demonstration**  
        *Causal Self-Attention mechanism implemented completely from scratch using PyTorch tensor operations.*
        """
    )

    model, tokenizer, device, error_msg = get_model()

    if error_msg:
        st.error(error_msg)
        st.info("💡 To train the model, open PowerShell and run: `python main.py --mode all`")
        return

    # Sidebar: Model Architecture & Status
    with st.sidebar:
        st.header("⚙️ Model Architecture")
        config = model.get_config()
        params = model.count_parameters()

        st.metric("Total Parameters", f"{params['total']:,}")
        st.metric("Vocabulary Size", f"{config['vocab_size']:,} tokens")
        st.write(f"- **Embedding Dimension:** `{config['d_model']}`")
        st.write(f"- **Attention Heads:** `{config['num_heads']}` ($d_k = {config['d_model'] // config['num_heads']}$)")
        st.write(f"- **Transformer Blocks:** `{config['num_layers']}`")
        st.write(f"- **Feed-Forward Dim:** `{config['d_ff']}`")
        st.write(f"- **Max Sequence Length:** `{config['max_seq_len']}`")
        st.write(f"- **Weight Tying:** `Enabled`")

        st.divider()
        st.header("📐 Scaled Dot-Product Attention")
        st.latex(r"\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V")
        st.caption(
            "Causal upper-triangular mask $M$ enforces autoregressive ordering by preventing attention to future tokens ($j > i$)."
        )

    # Main Tabs
    tab_pred, tab_attn, tab_metrics = st.tabs(["🔮 Next-Token Prediction & Generation", "🔍 Self-Attention Heatmap", "📊 Training & Test Metrics"])

    with tab_pred:
        col1, col2 = st.columns([1.2, 1.0])

        with col1:
            st.subheader("1. Enter Input Prompt")
            prompt_input = st.text_input(
                "Prompt text:",
                value="The sun rises in the",
                help="Type any sentence prefix to predict the next token."
            )

            st.subheader("2. Generation Settings")
            c_mode, c_tokens, c_temp = st.columns([1, 1, 1])
            with c_mode:
                gen_mode = st.radio("Decoding Strategy:", ["Greedy (Argmax)", "Sampling"], index=0)
            with c_tokens:
                num_new_tokens = st.slider("Tokens to Generate:", min_value=1, max_value=15, value=6)
            with c_temp:
                temperature = st.slider("Sampling Temperature:", min_value=0.1, max_value=2.0, value=0.8, step=0.1)

            predict_btn = st.button("🚀 Predict Next Token & Generate", type="primary", use_container_width=True)

        with col2:
            st.subheader("3. Next-Token Probabilities (Top 5)")
            if prompt_input.strip():
                top_candidates, _, input_tokens = predict_next_token_distribution(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt_input,
                    device=device
                )

                df = pd.DataFrame({
                    "Token": [f"'{cand[0]}'" for cand in top_candidates],
                    "Probability (%)": [cand[1] * 100 for cand in top_candidates]
                })

                st.bar_chart(df.set_index("Token"), color="#1f77b4")

                best_token = top_candidates[0][0]
                best_prob = top_candidates[0][1] * 100
                st.success(f"**Most Likely Next Token:** `{best_token}` ({best_prob:.2f}% probability)")

        if predict_btn or prompt_input:
            st.divider()
            st.subheader("4. Autoregressive Text Generation")
            is_sampling = (gen_mode == "Sampling")

            completed_text, gen_tokens = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt_input,
                max_new_tokens=num_new_tokens,
                temperature=temperature,
                sample=is_sampling,
                device=device
            )

            st.markdown(f"**Full Generated Sentence:**")
            st.info(f"👉 **{completed_text}**")
            st.caption(f"Newly added tokens: `{[t for t in gen_tokens]}` | Mode: {gen_mode}")

    with tab_attn:
        st.subheader("Causal Self-Attention Weight Heatmap")
        st.markdown(
            """
            This heatmap visualizes the **exact real attention weights** computed by the model for your prompt.
            Notice the **strict lower-triangular structure**: tokens can attend to themselves and preceding words,
            while all cells above the diagonal are strictly zero (preventing future information leakage).
            """
        )

        attn_prompt = st.text_input("Attention Visualization Prompt:", value="the sun rises in the east")
        tokens = tokenizer.tokenize(attn_prompt)
        ids = tokenizer.encode(attn_prompt, add_bos=True, add_eos=False)
        display_tokens = [tokenizer.bos_token] + tokens

        input_tensor = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            _, attentions = model(input_tensor, return_attention=True)

        if attentions and len(attentions) > 0:
            c_layer, c_head = st.columns(2)
            with c_layer:
                layer_choice = st.selectbox("Select Transformer Layer:", [f"Layer {i+1}" for i in range(len(attentions))], index=0)
                layer_idx = int(layer_choice.split()[1]) - 1
            with c_head:
                head_choice = st.selectbox("Select Attention Head:", [f"Head {i+1}" for i in range(model.num_heads)], index=0)
                head_idx = int(head_choice.split()[1]) - 1

            attn_matrix = attentions[layer_idx][0, head_idx].cpu().numpy()
            seq_len = len(display_tokens)
            sub_matrix = attn_matrix[:seq_len, :seq_len]

            fig, ax = plt.subplots(figsize=(7, 6))
            cax = ax.matshow(sub_matrix, cmap="Blues", vmin=0.0, vmax=1.0)
            ax.set_xticks(range(seq_len))
            ax.set_yticks(range(seq_len))
            ax.set_xticklabels(display_tokens, rotation=45, ha="left", fontsize=9)
            ax.set_yticklabels(display_tokens, fontsize=9)
            ax.set_xlabel("Keys (Attended Preceding Tokens)", fontsize=10, fontweight="bold", labelpad=8)
            ax.set_ylabel("Queries (Current Sequence Tokens)", fontsize=10, fontweight="bold")

            for i in range(seq_len):
                for j in range(seq_len):
                    val = sub_matrix[i, j]
                    if val > 0.01:
                        color = "white" if val > 0.55 else "black"
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)

            fig.colorbar(cax, fraction=0.046, pad=0.04)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with tab_metrics:
        st.subheader("Training and Evaluation Performance")
        col_m1, col_m2 = st.columns(2)

        with col_m1:
            if os.path.exists(LOSS_PLOT_PATH):
                st.image(LOSS_PLOT_PATH, caption="Training vs. Validation Loss Curves", use_container_width=True)
            else:
                st.info("Loss plot will appear here once training completes.")

        with col_m2:
            st.markdown("### Quantitative Test Metrics")
            if os.path.exists(METRICS_PATH):
                import json
                with open(METRICS_PATH, "r", encoding="utf-8") as f:
                    metrics_data = json.load(f)
                st.json(metrics_data)
            else:
                st.info("Evaluation metrics will appear here after evaluation.")


if __name__ == "__main__":
    main()
