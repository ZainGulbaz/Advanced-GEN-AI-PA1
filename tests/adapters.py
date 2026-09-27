from src.layers import Linear, Embedding, RMSNorm, SiLU, SwiGLU
from src.rope import RotaryPositionalEmbedding
from src.attention import custom_softmax, scaled_dot_product_attention, GroupedQuerySelfAttention
from src.model import TransformerBlock, TransformerLM
from src.optim import cross_entropy, AdamW, get_lr_cosine_schedule, gradient_clipping
from src.data import load_token_array, get_batch, save_checkpoint, load_checkpoint


def run_linear(x, w):
    m = Linear(w.shape[1], w.shape[0], device=x.device, dtype=x.dtype)
    m.weight.data.copy_(w)
    return m(x)


def run_embedding(t, w):
    m = Embedding(w.shape[0], w.shape[1], device=w.device, dtype=w.dtype)
    m.weight.data.copy_(w)
    return m(t)


def run_rmsnorm(x, w, eps):
    m = RMSNorm(w.shape[0], eps, device=x.device, dtype=x.dtype)
    m.weight.data.copy_(w)
    return m(x)


def run_silu(x):
    return SiLU()(x)


def run_swiglu(x, wg, wu, wd):
    m = SwiGLU(wg.shape[1], wg.shape[0], device=x.device, dtype=x.dtype)
    m.w_gate.weight.data.copy_(wg)
    m.w_up.weight.data.copy_(wu)
    m.w_down.weight.data.copy_(wd)
    return m(x)


def run_rope(x, pos, t, hd, cl):
    return RotaryPositionalEmbedding(t, hd, cl, device=x.device)(x, pos)


def run_softmax(x, d):
    return custom_softmax(x, d)


def run_scaled_dot_product_attention(q, k, v, m=None):
    return scaled_dot_product_attention(q, k, v, m)


def run_grouped_query_self_attention(x, wq, wk, wv, wo, nq, nkv, cl, rt, pos=None):
    m = GroupedQuerySelfAttention(
        x.shape[-1], nq, nkv, cl, rt, device=x.device, dtype=x.dtype
    )
    m.w_q.weight.data.copy_(wq)
    m.w_k.weight.data.copy_(wk)
    m.w_v.weight.data.copy_(wv)
    m.w_o.weight.data.copy_(wo)
    return m(x, pos)


def run_transformer_block(x, sd, nq, nkv, dff, cl, rt, eps, pos=None):
    m = TransformerBlock(
        x.shape[-1], nq, nkv, dff, cl, rt, eps, device=x.device, dtype=x.dtype
    )
    m.load_state_dict(sd)
    return m(x, pos)


def get_transformer_lm(vs, cl, dm, nl, nq, nkv, dff, rt, eps):
    return TransformerLM(vs, cl, dm, nl, nq, nkv, dff, rt, eps)


def run_transformer_lm(t, sd, vs, cl, dm, nl, nq, nkv, dff, rt, eps, pos=None):
    m = get_transformer_lm(vs, cl, dm, nl, nq, nkv, dff, rt, eps)
    m.load_state_dict(sd)
    return m(t, pos)


def run_cross_entropy(l, t):
    return cross_entropy(l, t)


def get_adamw_cls():
    return AdamW


def run_get_lr_cosine_schedule(s, amax, amin, sw, sc):
    return get_lr_cosine_schedule(s, amax, amin, sw, sc)


def run_gradient_clipping(p, mn):
    return gradient_clipping(p, mn)


def run_load_token_array(p):
    return load_token_array(p)


def run_get_batch(d, b, sl, dev, g):
    return get_batch(d, b, sl, dev, g)


def run_save_checkpoint(m, o, ns, tg, vg, out):
    save_checkpoint(m, o, ns, tg, vg, out)


def run_load_checkpoint(src, m, o, tg, vg):
    return load_checkpoint(src, m, o, tg, vg)
