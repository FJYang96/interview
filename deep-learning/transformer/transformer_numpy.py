import numpy as np


def layer_norm(x, gamma, beta, eps=1e-5):
    """
    Applies Layer Normalization.

    Args:
        x: np.ndarray of shape (batch_size, seq_len, d_model)
        gamma: np.ndarray of shape (d_model,)
        beta: np.ndarray of shape (d_model,)
        eps: float, small constant for numerical stability

    Returns:
        np.ndarray of shape (batch_size, seq_len, d_model)
    """
    x_normed = (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + eps)
    return x_normed * gamma + beta


def get_positional_encoding(seq_len, d_model):
    """
    Generates sinusoidal positional embeddings.

    Args:
        seq_len: int, length of the sequence
        d_model: int, dimension of the model

    Returns:
        np.ndarray of shape (seq_len, d_model)
    """
    pe = np.zeros((seq_len, d_model))
    pos = np.arange(seq_len)[:, None]
    div_term = np.exp(np.log(1e-4) * (np.arange(d_model)[0::2] / d_model))
    pe[:, 0::2] = np.sin(pos * div_term)
    pe[:, 1::2] = np.cos(pos * div_term)
    return pe


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Computes scaled dot-product attention.

    Args:
        Q: np.ndarray of shape (batch_size, num_heads, seq_len, d_k)
        K: np.ndarray of shape (batch_size, num_heads, seq_len, d_k)
        V: np.ndarray of shape (batch_size, num_heads, seq_len, d_v)
        mask: np.ndarray of shape (batch_size, 1, seq_len, seq_len) (optional)

    Returns:
        tuple: (attention_output, attention_weights)
    """
    d_k = Q.shape[-1]
    score = Q @ K.swapaxes(2, 3) / np.sqrt(d_k)  # (b, h, l, l)
    if mask is not None:
        score = np.where(mask, score, -1e9)
    score -= score.max(-1, keepdims=True)
    score_exp = np.exp(score)
    weight = score_exp / score_exp.sum(-1, keepdims=True)
    return weight @ V, weight


def multi_head_attention(x, W_q, W_k, W_v, W_o, num_heads, mask=None):
    """
    Computes multi-head attention.

    Args:
        x: np.ndarray of shape (batch_size, seq_len, d_model)
        W_q, W_k, W_v: np.ndarray of shape (d_model, d_model)
        W_o: np.ndarray of shape (d_model, d_model)
        num_heads: int, number of attention heads
        mask: np.ndarray (optional)

    Returns:
        np.ndarray of shape (batch_size, seq_len, d_model)
    """
    Q = np.stack(np.split(x @ W_q, num_heads, -1), 1)  # (b, h, l, d)
    K = np.stack(np.split(x @ W_k, num_heads, -1), 1)
    V = np.stack(np.split(x @ W_v, num_heads, -1), 1)
    out, _ = scaled_dot_product_attention(Q, K, V, mask)  # (b, h, l, d)
    return out.transpose(0, 2, 1, 3).reshape(x.shape) @ W_o


def moe_layer(x, W_gate, experts_W1, experts_W2, top_k=2):
    """
    Computes a sparse Mixture of Experts (MoE) feed-forward layer.

    Args:
        x: np.ndarray of shape (batch_size, seq_len, d_model)
        W_gate: np.ndarray of shape (d_model, num_experts) for routing
        experts_W1: np.ndarray of shape (num_experts, d_model, d_ff)
        experts_W2: np.ndarray of shape (num_experts, d_ff, d_model)
        top_k: int, number of experts to route each token to

    Returns:
        np.ndarray of shape (batch_size, seq_len, d_model)
    """
    # Find expert weights
    logits = x @ W_gate  # (B, L, E)
    logits -= logits.max(-1, keepdims=True)

    # Forward pass
    B, L, d_model = x.shape
    out = np.zeros_like(x)
    for b in range(B):
        for l in range(L):
            exp_logits = logits[b, l]
            exp_inds = np.argpartition(-exp_logits, top_k)[:top_k]  # (k,)
            hidden = np.maximum(
                0, x[b, l, None, :] @ experts_W1[exp_inds, :, :]
            )  # (1, d_model) @ (k, d_model, d_ff) -> (k, 1, d_ff)
            y = np.squeeze(hidden @ experts_W2[exp_inds, :, :], axis=1)  # (k, d_model)
            exp_weights = np.exp(exp_logits[exp_inds]) / np.sum(
                np.exp(exp_logits[exp_inds])
            )  # (k,)
            out[b, l] = np.sum(y * exp_weights[:, None], axis=0)

    return out


def transformer_block(
    x, mha_params, moe_params, ln1_params, ln2_params, num_heads, mask=None
):
    """
    Assembles a single Transformer block (Pre-LN architecture).

    Args:
        x: np.ndarray of shape (batch_size, seq_len, d_model)
        mha_params: tuple (W_q, W_k, W_v, W_o)
        moe_params: tuple (W_gate, experts_W1, experts_W2)
        ln1_params: tuple (gamma1, beta1) for MHA norm
        ln2_params: tuple (gamma2, beta2) for MoE norm
        num_heads: int
        mask: np.ndarray (optional)

    Returns:
        np.ndarray of shape (batch_size, seq_len, d_model)
    """
    norm1 = layer_norm(x, *ln1_params)
    attn1 = multi_head_attention(norm1, *mha_params, num_heads, mask)
    out1 = x + attn1

    norm2 = layer_norm(out1, *ln2_params)
    ff2 = moe_layer(norm2, *moe_params)
    return out1 + ff2


# ==========================================
# REFERENCE IMPLEMENTATIONS (DO NOT PEEK!)
# ==========================================


def ref_layer_norm(x, gamma, beta, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return ((x - mean) / np.sqrt(var + eps)) * gamma + beta


def ref_positional_encoding(seq_len, d_model):
    pe = np.zeros((seq_len, d_model))
    pos = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(pos * div_term)
    pe[:, 1::2] = np.cos(pos * div_term)
    return pe


def ref_sdpa(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = np.matmul(Q, K.swapaxes(-2, -1)) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    scores -= np.max(scores, axis=-1, keepdims=True)  # numeric stability
    attn = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
    return np.matmul(attn, V), attn


def ref_mha(x, W_q, W_k, W_v, W_o, num_heads, mask=None):
    B, S, D = x.shape
    d_k = D // num_heads

    Q = np.matmul(x, W_q).reshape(B, S, num_heads, d_k).transpose(0, 2, 1, 3)
    K = np.matmul(x, W_k).reshape(B, S, num_heads, d_k).transpose(0, 2, 1, 3)
    V = np.matmul(x, W_v).reshape(B, S, num_heads, d_k).transpose(0, 2, 1, 3)

    out, _ = ref_sdpa(Q, K, V, mask)
    out = out.transpose(0, 2, 1, 3).reshape(B, S, D)
    return np.matmul(out, W_o)


def ref_moe(x, W_gate, experts_W1, experts_W2, top_k=2):
    B, S, D = x.shape
    logits = np.matmul(x, W_gate)
    out = np.zeros_like(x)

    for b in range(B):
        for s in range(S):
            token_logits = logits[b, s]
            top_indices = np.argsort(token_logits)[-top_k:]
            top_logits = token_logits[top_indices]

            top_logits -= np.max(top_logits)
            gates = np.exp(top_logits) / np.sum(np.exp(top_logits))

            token_out = np.zeros(D)
            for i, idx in enumerate(top_indices):
                h = np.maximum(0, np.matmul(x[b, s], experts_W1[idx]))  # ReLU
                token_out += gates[i] * np.matmul(h, experts_W2[idx])
            out[b, s] = token_out
    return out


def ref_block(x, mha_params, moe_params, ln1_params, ln2_params, num_heads, mask=None):
    norm1 = ref_layer_norm(x, *ln1_params)
    mha_out = ref_mha(norm1, *mha_params, num_heads, mask)
    x = x + mha_out

    norm2 = ref_layer_norm(x, *ln2_params)
    moe_out = ref_moe(norm2, *moe_params)
    return x + moe_out


# ==========================================
# TESTS
# ==========================================


def test_layer_norm():
    np.random.seed(42)
    x = np.random.randn(4, 10, 16)
    gamma, beta = np.random.randn(16), np.random.randn(16)

    expected = ref_layer_norm(x, gamma, beta)
    actual = layer_norm(x, gamma, beta)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
    print("Layer Norm passed.")


def test_positional_encoding():
    expected = ref_positional_encoding(20, 32)
    actual = get_positional_encoding(20, 32)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
    print("Positional Encoding passed.")


def test_scaled_dot_product_attention():
    np.random.seed(42)
    Q, K, V = np.random.randn(3, 2, 4, 5, 8)

    mask = np.triu(np.ones((5, 5)), k=1).astype(bool).reshape(1, 1, 5, 5)

    expected_out, expected_weights = ref_sdpa(Q, K, V, mask)
    actual_out, actual_weights = scaled_dot_product_attention(Q, K, V, mask)

    np.testing.assert_allclose(actual_out, expected_out, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(actual_weights, expected_weights, rtol=1e-4, atol=1e-5)
    print("Scaled Dot-Product Attention passed.")


def test_multi_head_attention():
    np.random.seed(42)
    B, S, D, H = 2, 5, 16, 4
    x = np.random.randn(B, S, D)
    params = [np.random.randn(D, D) for _ in range(4)]

    expected = ref_mha(x, *params, H)
    actual = multi_head_attention(x, *params, H)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
    print("Multi-Head Attention passed.")


def test_moe_layer():
    np.random.seed(42)
    B, S, D, E, D_ff = 2, 3, 8, 4, 16
    x = np.random.randn(B, S, D)
    W_gate = np.random.randn(D, E)
    experts_W1 = np.random.randn(E, D, D_ff)
    experts_W2 = np.random.randn(E, D_ff, D)

    expected = ref_moe(x, W_gate, experts_W1, experts_W2, top_k=2)
    actual = moe_layer(x, W_gate, experts_W1, experts_W2, top_k=2)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
    print("Mixture of Experts passed.")


def test_transformer_block():
    np.random.seed(42)
    B, S, D, H, E, D_ff = 2, 5, 16, 4, 4, 32
    x = np.random.randn(B, S, D)

    mha_params = [np.random.randn(D, D) for _ in range(4)]
    moe_params = [
        np.random.randn(D, E),
        np.random.randn(E, D, D_ff),
        np.random.randn(E, D_ff, D),
    ]
    ln1_params = [np.random.randn(D), np.random.randn(D)]
    ln2_params = [np.random.randn(D), np.random.randn(D)]

    expected = ref_block(x, mha_params, moe_params, ln1_params, ln2_params, H)
    actual = transformer_block(x, mha_params, moe_params, ln1_params, ln2_params, H)
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)
    print("Transformer Block passed.")


if __name__ == "__main__":
    test_layer_norm()
    test_positional_encoding()
    test_scaled_dot_product_attention()
    test_multi_head_attention()
    test_moe_layer()
    test_transformer_block()
    print("All mathematical correctness tests passed successfully.")
