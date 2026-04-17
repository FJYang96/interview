import torch.nn as nn
from attention import scaled_dot_product_attention


class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        # Linear layers for Q, K, V and the final output
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1. Apply linear projections and reshape to (batch, heads, seq_len, d_k)
        # view() splits d_model into num_heads x d_k. transpose() moves heads to dim 1.
        q = (
            self.w_q(query)
            .view(batch_size, -1, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        k = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = (
            self.w_v(value)
            .view(batch_size, -1, self.num_heads, self.d_k)
            .transpose(1, 2)
        )

        # 2. Apply scaled dot-product attention (using function from above)
        x, attn = scaled_dot_product_attention(q, k, v, mask=mask)

        # 3. Concatenate heads: reshape back to (batch, seq_len, d_model)
        # contiguous() is required after transpose before calling view()
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.num_heads * self.d_k)
        )

        # 4. Final linear projection
        return self.w_o(x)
