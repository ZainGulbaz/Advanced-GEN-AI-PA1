import math
import torch
import torch.nn as nn
from einops import rearrange
from src.layers import Linear
from src.rope import RotaryPositionalEmbedding


def custom_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if x.numel() == 0:
        return x
    max_val = torch.amax(x, dim=dim, keepdim=True)
    shifted = x - max_val
    exp_x = torch.exp(shifted)
    denom = exp_x.sum(dim=dim, keepdim=True)
    if torch.any(denom == 0) or not torch.isfinite(denom).all():
        raise ValueError("softmax is undefined for the supplied values")
    return exp_x / denom


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k must have the same head dimension")
    if k.shape[-2] != v.shape[-2]:
        raise ValueError("k and v must have the same sequence dimension")
    if q.ndim < 2 or k.ndim < 2 or v.ndim < 2:
        raise ValueError("q, k, and v must have at least two dimensions")

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if mask is not None:
        if mask.dtype != torch.bool:
            raise TypeError("attention mask must have boolean dtype")
        if mask.shape[-2:] != scores.shape[-2:]:
            raise ValueError("mask final dimensions must be (n_q, n_kv)")
        try:
            mask = torch.broadcast_to(mask, scores.shape)
        except RuntimeError as exc:
            raise ValueError("mask leading dimensions are not broadcastable to attention scores") from exc
        if not torch.all(mask.any(dim=-1)):
            raise ValueError("every query must have at least one permitted key position")
        scores = scores.masked_fill(~mask, float("-inf"))

    return torch.matmul(custom_softmax(scores, dim=-1), v)


class GroupedQuerySelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_q_heads: int,
        n_kv_heads: int,
        context_length: int,
        rope_theta: float = 10000.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if d_model <= 0 or n_q_heads <= 0 or n_kv_heads <= 0:
            raise ValueError("model and head counts must be positive")
        if d_model % n_q_heads != 0:
            raise ValueError("d_model must be divisible by n_q_heads")
        if n_q_heads % n_kv_heads != 0:
            raise ValueError("n_q_heads must be divisible by n_kv_heads")
        self.d_model = d_model
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.group_size = n_q_heads // n_kv_heads
        self.head_dim = d_model // n_q_heads
        self.context_length = context_length

        self.w_q = Linear(d_model, n_q_heads * self.head_dim, device=device, dtype=dtype)
        self.w_k = Linear(d_model, n_kv_heads * self.head_dim, device=device, dtype=dtype)
        self.w_v = Linear(d_model, n_kv_heads * self.head_dim, device=device, dtype=dtype)
        self.w_o = Linear(n_q_heads * self.head_dim, d_model, device=device, dtype=dtype)
        self.rope = RotaryPositionalEmbedding(rope_theta, self.head_dim, context_length, device=device)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("self-attention expects x with shape (batch, sequence, d_model)")
        batch, seq_len, d_model = x.shape
        if d_model != self.d_model:
            raise ValueError("input final dimension does not match d_model")
        if not (1 <= seq_len <= self.context_length):
            raise ValueError("sequence length must be between 1 and context_length")

        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
        elif token_positions.ndim == 0 or token_positions.shape[-1] != seq_len:
            raise ValueError("token_positions must have final dimension equal to sequence length")

        q = rearrange(self.w_q(x), "b n (h g d) -> b h g n d",
                      h=self.n_kv_heads, g=self.group_size, d=self.head_dim)
        k = rearrange(self.w_k(x), "b n (h d) -> b h n d",
                      h=self.n_kv_heads, d=self.head_dim)
        v = rearrange(self.w_v(x), "b n (h d) -> b h n d",
                      h=self.n_kv_heads, d=self.head_dim)

        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)

        scores = torch.einsum("bhgnd,bhmd->bhgnm", q, k) / math.sqrt(self.head_dim)
        causal_mask = torch.arange(seq_len, device=x.device).unsqueeze(1) >= torch.arange(seq_len, device=x.device).unsqueeze(0)
        scores = scores.masked_fill(~causal_mask, float("-inf"))
        probs = custom_softmax(scores, dim=-1)
        out = torch.einsum("bhgnm,bhmd->bhgnd", probs, v)
        out = rearrange(out, "b h g n d -> b n (h g d)")
        return self.w_o(out)
