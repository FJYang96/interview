import math

import torch
import torch.nn.functional as F


def scaled_dot_product_attention(query, key, value, mask=None):
    # query, key, value shape: (batch_size, num_heads, seq_len, d_k)
    d_k = query.size(-1)

    # Compute attention scores: Q * K^T / sqrt(d_k)
    # Transpose only the last two dimensions of key
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # Fill masked positions with negative infinity before softmax
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # Apply softmax to get attention weights
    p_attn = F.softmax(scores, dim=-1)

    # Multiply weights by values
    return torch.matmul(p_attn, value), p_attn
