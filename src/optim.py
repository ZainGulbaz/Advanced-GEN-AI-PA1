import math
import numbers
import torch
from torch.optim import Optimizer


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.ndim < 1:
        raise ValueError("logits must have at least one dimension")
    if targets.shape != logits.shape[:-1]:
        raise ValueError("targets must match logits.shape[:-1]")
    if targets.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError("targets must have an integer dtype")
    if torch.any(targets < 0) or torch.any(targets >= logits.shape[-1]):
        raise IndexError("target index is outside the vocabulary")
    target_logits = torch.gather(logits, -1, targets.unsqueeze(-1)).squeeze(-1)
    return (torch.logsumexp(logits, dim=-1) - target_logits).mean()


def get_lr_cosine_schedule(step: int, alpha_max: float, alpha_min: float,
                           s_w: int, s_c: int) -> float:
    if not isinstance(step, numbers.Integral) or isinstance(step, bool):
        raise TypeError("step must be an integer")
    if not isinstance(s_w, numbers.Integral) or isinstance(s_w, bool):
        raise TypeError("warmup endpoint must be an integer")
    if not isinstance(s_c, numbers.Integral) or isinstance(s_c, bool):
        raise TypeError("cosine endpoint must be an integer")
    if step < 0:
        raise ValueError("step must be non-negative")
    if s_w < 0 or s_w >= s_c:
        raise ValueError("require 0 <= warmup endpoint < cosine endpoint")
    if alpha_min < 0 or alpha_max < 0 or alpha_min > alpha_max:
        raise ValueError("require 0 <= alpha_min <= alpha_max")

    if s_w > 0 and step < s_w:
        return (step / s_w) * alpha_max
    if step <= s_c:
        progress = (step - s_w) / (s_c - s_w)
        return alpha_min + 0.5 * (1.0 + math.cos(math.pi * progress)) * (alpha_max - alpha_min)
    return alpha_min


def gradient_clipping(parameters, max_norm: float, eps: float = 1e-6) -> float:
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")
    if eps < 0:
        raise ValueError("eps must be non-negative")

    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    if any(g.is_sparse for g in grads):
        raise RuntimeError("gradient clipping does not support sparse gradients")

    total_sq = torch.zeros((), device=grads[0].device, dtype=torch.float32)
    for grad in grads:
        total_sq = total_sq + grad.detach().float().pow(2).sum()
    total_norm = float(torch.sqrt(total_sq).item())

    if total_norm > max_norm:
        scale = max_norm / (total_norm + eps)
        for grad in grads:
            grad.detach().mul_(scale)
    return total_norm


class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        if lr < 0:
            raise ValueError("lr must be non-negative")
        if eps < 0:
            raise ValueError("eps must be non-negative")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if not (0 <= betas[0] < 1 and 0 <= betas[1] < 1):
            raise ValueError("betas must satisfy 0 <= beta < 1")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        self._validate_param_groups()

    @staticmethod
    def _validate_group(group):
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        weight_decay = group["weight_decay"]
        if lr < 0:
            raise ValueError("lr must be non-negative")
        if eps < 0:
            raise ValueError("eps must be non-negative")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if not (0 <= beta1 < 1 and 0 <= beta2 < 1):
            raise ValueError("betas must satisfy 0 <= beta < 1")

    def _validate_param_groups(self):
        for group in self.param_groups:
            self._validate_group(group)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self._validate_group(group)
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                grad = p.grad

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step

                if weight_decay != 0:
                    p.mul_(1.0 - lr * weight_decay)

                denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
                step_size = lr / bias_correction1
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
