import math
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        std = math.sqrt(2.0 / (in_features + out_features))
        trunc_normal_(self.weight, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(f"expected final dimension {self.in_features}, got {x.shape[-1]}")
        return torch.matmul(x, self.weight.transpose(-1, -2))


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()
        if num_embeddings <= 0 or embedding_dim <= 0:
            raise ValueError("num_embeddings and embedding_dim must be positive")
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError("token_ids must have an integer dtype")
        if torch.any(token_ids < 0) or torch.any(token_ids >= self.num_embeddings):
            raise IndexError("token id is outside the embedding vocabulary")
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, norm_eps: float = 1e-5,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if norm_eps < 0:
            raise ValueError("norm_eps must be non-negative")
        self.norm_eps = norm_eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.weight.shape[0]:
            raise ValueError("input final dimension does not match RMSNorm dimension")
        in_dtype = x.dtype
        work = x.to(torch.float32) if x.dtype in (torch.float16, torch.bfloat16) else x
        rms_inv = torch.rsqrt(work.pow(2).mean(dim=-1, keepdim=True) + self.norm_eps)
        result = work * rms_inv * self.weight.to(work.dtype)
        return result.to(in_dtype)


class SiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.w_gate = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w_up = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w_down = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.silu = SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(self.silu(self.w_gate(x)) * self.w_up(x))
