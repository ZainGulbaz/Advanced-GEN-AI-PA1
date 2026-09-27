import torch
from src.attention import custom_softmax


def generate(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    context_length: int,
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eot_token_id: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if prompt_ids.ndim != 1 or prompt_ids.numel() == 0:
        raise ValueError("prompt_ids must be a non-empty one-dimensional tensor")
    if prompt_ids.dtype != torch.long:
        raise TypeError("prompt_ids must have torch.long dtype")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not (0 < top_p <= 1):
        raise ValueError("top_p must satisfy 0 < top_p <= 1")
    if not hasattr(model, "context_length"):
        raise AttributeError("model must expose context_length")
    if context_length != model.context_length:
        raise ValueError("supplied context_length must equal model.context_length")
    if generator is not None and not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator or None")

    result = prompt_ids.clone()
    if max_new_tokens == 0:
        return result

    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                context = result[-context_length:]
                logits = model(context.unsqueeze(0))[:, -1, :].squeeze(0)
                probs = custom_softmax(logits / temperature, dim=-1)

                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                if top_p < 1.0:
                    cumulative = torch.cumsum(sorted_probs, dim=-1)
                    keep = cumulative <= top_p
                    first_over = torch.nonzero(cumulative >= top_p, as_tuple=False)
                    if first_over.numel() > 0:
                        keep[first_over[0, 0]] = True
                    keep[0] = True
                    filtered = sorted_probs * keep.to(sorted_probs.dtype)
                else:
                    filtered = sorted_probs

                filtered = filtered / filtered.sum()
                sampled_rank = torch.multinomial(filtered, 1, generator=generator)
                next_token = sorted_indices[sampled_rank].reshape(1)
                result = torch.cat([result, next_token])

                if eot_token_id is not None and int(next_token.item()) == eot_token_id:
                    break
    finally:
        model.train(was_training)

    return result
