import torch
import torch.nn as nn


_INTEGER_DTYPES = {
    torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
}


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, rope_theta: float, head_dim: int, context_length: int,
                 device: torch.device | None = None):
        super().__init__()
        if head_dim <= 0 or head_dim % 2 != 0:
            raise ValueError("head_dim must be a positive even integer")
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        if rope_theta <= 0:
            raise ValueError("rope_theta must be positive")

        self.head_dim = head_dim
        self.context_length = context_length
        half = head_dim // 2
        positions = torch.arange(context_length, device=device, dtype=torch.float32)
        frequencies = 1.0 / (
            rope_theta ** (torch.arange(half, device=device, dtype=torch.float32) * (2.0 / head_dim))
        )
        angles = torch.outer(positions, frequencies)
        self.register_buffer("cos_cached", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cached", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.head_dim:
            raise ValueError(f"expected final dimension {self.head_dim}, got {x.shape[-1]}")
        if token_positions.dtype not in _INTEGER_DTYPES:
            raise TypeError("token_positions must have an integer dtype")
        if token_positions.ndim == 0 or token_positions.shape[-1] != x.shape[-2]:
            raise ValueError("token_positions final dimension must equal sequence length")

        x_batch_ndim = x.ndim - 2
        pos_batch_ndim = token_positions.ndim - 1
        if pos_batch_ndim > x_batch_ndim:
            raise ValueError("token_positions has too many leading dimensions")

        # Align position batch dimensions with x's batch-like dimensions and
        # leave singleton dimensions for attention-head axes.
        pos_shape = tuple(token_positions.shape[:-1])
        pos_shape = pos_shape + (1,) * (x_batch_ndim - pos_batch_ndim) + (x.shape[-2],)
        positions = token_positions.reshape(pos_shape)

        if torch.any(positions < 0) or torch.any(positions >= self.context_length):
            raise ValueError("token_positions contains positions outside the supported context")

        cos = self.cos_cached[positions]
        sin = self.sin_cached[positions]
        # cos/sin: batch-like dims + sequence + half_head_dim
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rotated = torch.empty_like(x)
        rotated[..., 0::2] = x_even * cos - x_odd * sin
        rotated[..., 1::2] = x_even * sin + x_odd * cos
        return rotated
