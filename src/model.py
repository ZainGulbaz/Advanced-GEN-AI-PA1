import torch
import torch.nn as nn
from src.layers import Linear, Embedding, RMSNorm, SwiGLU
from src.attention import GroupedQuerySelfAttention


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_q_heads, n_kv_heads, d_ff,
                 context_length, rope_theta, norm_eps,
                 device=None, dtype=None):
        super().__init__()
        self.attention_norm = RMSNorm(d_model, norm_eps, device=device, dtype=dtype)
        self.attention = GroupedQuerySelfAttention(
            d_model, n_q_heads, n_kv_heads, context_length, rope_theta,
            device=device, dtype=dtype,
        )
        self.ffn_norm = RMSNorm(d_model, norm_eps, device=device, dtype=dtype)
        self.swiglu = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), token_positions=token_positions)
        x = x + self.swiglu(self.ffn_norm(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers,
                 n_q_heads, n_kv_heads, d_ff,
                 rope_theta=10000.0, norm_eps=1e-5,
                 device=None, dtype=None):
        super().__init__()
        if vocab_size <= 0 or context_length <= 0 or num_layers <= 0 or d_ff <= 0:
            raise ValueError("vocab_size, context_length, num_layers, and d_ff must be positive")
        self.context_length = context_length
        self.token_embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model, n_q_heads, n_kv_heads, d_ff,
                context_length, rope_theta, norm_eps,
                device=device, dtype=dtype,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model, norm_eps, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch_size, sequence_length)")
        seq_len = token_ids.shape[-1]
        if not (1 <= seq_len <= self.context_length):
            raise ValueError("sequence length must be between 1 and context_length")
        if token_positions is not None:
            if token_positions.ndim == 0 or token_positions.shape[-1] != seq_len:
                raise ValueError("token_positions must have final dimension equal to sequence length")

        x = self.token_embedding(token_ids)
        for block in self.blocks:
            x = block(x, token_positions=token_positions)
        return self.lm_head(self.final_norm(x))
