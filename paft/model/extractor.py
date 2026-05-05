"""
Extract per-head weight matrices from a GPT-2 model.

GPT-2 weight layout (HuggingFace Conv1D convention)
────────────────────────────────────────────────────
Conv1D computes:  output = input @ weight + bias
weight shape is [in_features, out_features]  (transpose of nn.Linear)

c_attn.weight  [n_embd, 3*n_embd]   — fused QKV projection
    columns  [0         : n_embd  ]  = W_Q  (all heads)
    columns  [n_embd    : 2*n_embd]  = W_K  (all heads)
    columns  [2*n_embd  : 3*n_embd]  = W_V  (all heads)

c_attn.bias    [3*n_embd]
    indices  [0         : n_embd  ]  = b_Q
    indices  [n_embd    : 2*n_embd]  = b_K
    indices  [2*n_embd  : 3*n_embd]  = b_V

c_proj.weight  [n_embd, n_embd]     — output projection
    rows  [h*d_head : (h+1)*d_head]  = W_O_h for head h

c_proj.bias    [n_embd]

Per-head shapes (GPT-2 small: n_embd=768, n_heads=12, d_head=64):
    W_V_h : [n_embd, d_head]   tall — right polar decomp   W_V_h = Q_V_h @ S_V_h
    W_O_h : [d_head, n_embd]   wide — left  polar decomp   W_O_h = S_O_h @ Q_O_h

Head grouping within W_V
─────────────────────────
The V block of c_attn.weight is [n_embd, n_embd].  Its columns are grouped
head-by-head: head h occupies columns h*d_head … (h+1)*d_head.

Bulk extraction exploits this layout via a single reshape+permute, avoiding a
Python loop over heads.  The two forms are numerically identical:

    Single:  w[:, 2*n_embd + h*d_head : 2*n_embd + (h+1)*d_head]  → [n_embd, d_head]
    Bulk:    w[:, 2*n_embd:].reshape(n_embd, H, d_head).permute(1,0,2)[h]  → [n_embd, d_head]

Analogously for W_O (c_proj.weight rows):
    Single:  w[h*d_head : (h+1)*d_head, :]     → [d_head, n_embd]
    Bulk:    w.reshape(H, d_head, n_embd)[h]    → [d_head, n_embd]

Calling convention
──────────────────
All functions return detached clones on CPU.  They do NOT cast dtype — the
caller is responsible (PAFTModel._build_paft_attention calls .float()).

All functions must be called on the ORIGINAL GPT2LMHeadModel before any
attention-layer surgery.  After PAFTAttention replaces GPT2Attention the
c_attn / c_proj attributes no longer exist on the replaced layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


# ──────────────────────────────────────────────────────────────────────────────
# Architecture dimensions
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GPT2Dims:
    """
    Immutable architecture dimensions for one GPT-2 variant.

    Created once per model via get_gpt2_dims() and reused across all extraction
    calls in _build_paft_attention.  Avoids repeated config attribute reads.
    """
    n_layers: int
    n_heads:  int
    n_embd:   int
    d_head:   int

    def __post_init__(self) -> None:
        if self.n_embd != self.n_heads * self.d_head:
            raise ValueError(
                f"n_embd={self.n_embd} must equal n_heads*d_head "
                f"({self.n_heads}*{self.d_head}={self.n_heads*self.d_head})"
            )

    def to_dict(self) -> Dict[str, int]:
        return {
            "n_layers": self.n_layers,
            "n_heads":  self.n_heads,
            "n_embd":   self.n_embd,
            "d_head":   self.d_head,
        }


def get_gpt2_dims(model) -> GPT2Dims:
    """
    Read architecture dimensions from a GPT-2 model's config.

    Works with GPT2LMHeadModel and any wrapper that exposes a .config with the
    standard GPT-2 config attributes (n_layer, n_head, n_embd).

    Returns:
        GPT2Dims — immutable dataclass; pass around instead of re-reading config.

    Raises:
        AttributeError if config is missing expected attributes.
        ValueError    if n_embd is not divisible by n_head.
    """
    cfg    = model.config
    n_embd = cfg.n_embd
    n_head = cfg.n_head

    if n_embd % n_head != 0:
        raise ValueError(
            f"n_embd={n_embd} is not divisible by n_head={n_head}"
        )

    return GPT2Dims(
        n_layers = cfg.n_layer,
        n_heads  = n_head,
        n_embd   = n_embd,
        d_head   = n_embd // n_head,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Validation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _check_layer(layer_idx: int, n_layers: int) -> None:
    if not (0 <= layer_idx < n_layers):
        raise IndexError(
            f"layer_idx={layer_idx} out of range for model with {n_layers} layers"
        )


def _check_head(head_idx: int, n_heads: int) -> None:
    if not (0 <= head_idx < n_heads):
        raise IndexError(
            f"head_idx={head_idx} out of range for model with {n_heads} heads"
        )


def _get_attn(model, layer_idx: int):
    """
    Return the attention module for one layer.
    Must be called before surgery — after PAFTAttention replacement, c_attn
    and c_proj attributes no longer exist on the replaced layer.
    """
    attn = model.transformer.h[layer_idx].attn
    if not hasattr(attn, "c_attn"):
        raise AttributeError(
            f"Layer {layer_idx} attention has no 'c_attn' attribute. "
            "extract_* functions must be called before PAFTAttention surgery. "
            f"Got: {type(attn).__name__}"
        )
    return attn


# ──────────────────────────────────────────────────────────────────────────────
# Single-head extraction  (used by analysis scripts, tests, and SVFModel)
# ──────────────────────────────────────────────────────────────────────────────

def extract_wv_head(
    model,
    layer_idx: int,
    head_idx:  int,
    dims:      GPT2Dims | None = None,
) -> torch.Tensor:
    """
    Extract W_V for one head from one transformer layer.

    Layout: c_attn.weight columns [2*n_embd + h*d_head : 2*n_embd + (h+1)*d_head]

    Returns:
        Tensor of shape [n_embd, d_head], detached, CPU, original dtype.

    Args:
        model:     GPT2LMHeadModel (or any wrapper exposing .config and .transformer).
        layer_idx: Transformer block index (0-indexed).
        head_idx:  Attention head index (0-indexed).
        dims:      Optional pre-computed GPT2Dims; avoids a config read if provided.
    """
    if dims is None:
        dims = get_gpt2_dims(model)
    _check_layer(layer_idx, dims.n_layers)
    _check_head(head_idx, dims.n_heads)

    attn      = _get_attn(model, layer_idx)
    w         = attn.c_attn.weight                        # [n_embd, 3*n_embd]
    col_start = 2 * dims.n_embd + head_idx * dims.d_head
    col_end   = col_start + dims.d_head

    return w[:, col_start:col_end].detach().clone()       # [n_embd, d_head]


def extract_wo_head(
    model,
    layer_idx: int,
    head_idx:  int,
    dims:      GPT2Dims | None = None,
) -> torch.Tensor:
    """
    Extract W_O for one head from one transformer layer.

    Layout: c_proj.weight rows [h*d_head : (h+1)*d_head]

    Each head h contributes:  output_h = v_out_h @ W_O_h
    where v_out_h is the attended value for head h, shape [B, T, d_head].

    Returns:
        Tensor of shape [d_head, n_embd], detached, CPU, original dtype.
    """
    if dims is None:
        dims = get_gpt2_dims(model)
    _check_layer(layer_idx, dims.n_layers)
    _check_head(head_idx, dims.n_heads)

    attn      = _get_attn(model, layer_idx)
    w         = attn.c_proj.weight                        # [n_embd, n_embd]
    row_start = head_idx * dims.d_head
    row_end   = row_start + dims.d_head

    return w[row_start:row_end, :].detach().clone()       # [d_head, n_embd]


# ──────────────────────────────────────────────────────────────────────────────
# Full-layer extraction  (frozen QK weights and biases)
# ──────────────────────────────────────────────────────────────────────────────

def extract_wq_wk_layer(
    model,
    layer_idx: int,
    dims:      GPT2Dims | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract the full (all-heads combined) W_Q and W_K for one transformer layer.

    Q and K projections are frozen in all PAFT variants — only W_V and W_O
    are decomposed.  These are extracted once and stored as frozen buffers in
    PAFTAttention.

    Layout:
        c_attn.weight[:, 0         : n_embd  ]  = W_Q  [n_embd, n_embd]
        c_attn.weight[:, n_embd    : 2*n_embd]  = W_K  [n_embd, n_embd]

    Returns:
        W_Q : [n_embd, n_embd], detached clone
        W_K : [n_embd, n_embd], detached clone
    """
    if dims is None:
        dims = get_gpt2_dims(model)
    _check_layer(layer_idx, dims.n_layers)

    w   = _get_attn(model, layer_idx).c_attn.weight       # [n_embd, 3*n_embd]
    W_Q = w[:, :dims.n_embd].detach().clone()             # [n_embd, n_embd]
    W_K = w[:, dims.n_embd : 2 * dims.n_embd].detach().clone()

    return W_Q, W_K


def extract_biases_layer(
    model,
    layer_idx: int,
    dims:      GPT2Dims | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract QKV and output projection biases for one transformer layer.

    c_attn.bias layout:
        indices [0         : n_embd  ] = b_Q
        indices [n_embd    : 2*n_embd] = b_K
        indices [2*n_embd  : 3*n_embd] = b_V

    PAFTAttention stores these as a single b_qkv [3*n_embd] parameter and
    slices internally in the forward pass, matching the original layout exactly.

    Returns:
        b_qkv : [3*n_embd]  — fused QKV bias (b_Q | b_K | b_V)
        b_o   : [n_embd]    — output projection bias
        Both are detached clones.

    Note:
        GPT-2 always has biases in c_attn and c_proj.  If a model variant
        has bias=False, an AttributeError will be raised here — the caller
        should handle this case if supporting such variants.
    """
    if dims is None:
        dims = get_gpt2_dims(model)
    _check_layer(layer_idx, dims.n_layers)

    attn  = _get_attn(model, layer_idx)
    b_qkv = attn.c_attn.bias.detach().clone()             # [3*n_embd]
    b_o   = attn.c_proj.bias.detach().clone()             # [n_embd]

    return b_qkv, b_o


# ──────────────────────────────────────────────────────────────────────────────
# Bulk head extraction  (all heads for one layer in a single slice/reshape)
# ──────────────────────────────────────────────────────────────────────────────
# These are used by PAFTModel._build_paft_attention and SVFModel to avoid a
# Python loop over heads.  They are numerically equivalent to calling the
# single-head functions in a loop and stacking.

def extract_wv_all_heads(
    model,
    layer_idx: int,
    dims:      GPT2Dims | None = None,
) -> torch.Tensor:
    """
    Extract W_V for ALL heads in one layer via a single slice+reshape.

    Equivalent to:
        torch.stack([extract_wv_head(model, layer_idx, h, dims) for h in range(H)])

    Implementation:
        V block = c_attn.weight[:, 2*n_embd:]    shape [n_embd, n_embd]
        reshape to [n_embd, H, d_head]           groups columns head-by-head
        permute to [H, n_embd, d_head]           head-first for bmm convention

    The reshape works because columns in the V block are laid out
    head-by-head: head h occupies columns h*d_head … (h+1)*d_head.
    Reshape groups them as [n_embd, H, d_head] in one step.

    Returns:
        Tensor of shape [H, n_embd, d_head], detached, CPU, original dtype.
    """
    if dims is None:
        dims = get_gpt2_dims(model)
    _check_layer(layer_idx, dims.n_layers)

    w    = _get_attn(model, layer_idx).c_attn.weight          # [n_embd, 3*n_embd]
    W_V  = w[:, 2 * dims.n_embd :].detach().clone()           # [n_embd, n_embd]

    # [n_embd, n_embd] → [n_embd, H, d_head] → [H, n_embd, d_head]
    return W_V.reshape(dims.n_embd, dims.n_heads, dims.d_head).permute(1, 0, 2).contiguous()


def extract_wo_all_heads(
    model,
    layer_idx: int,
    dims:      GPT2Dims | None = None,
) -> torch.Tensor:
    """
    Extract W_O for ALL heads in one layer via a single reshape.

    Equivalent to:
        torch.stack([extract_wo_head(model, layer_idx, h, dims) for h in range(H)])

    Implementation:
        c_proj.weight  shape [n_embd, n_embd]
        reshape to [H, d_head, n_embd]   rows are grouped head-by-head

    The reshape works because rows of c_proj.weight are laid out head-by-head:
    head h occupies rows h*d_head … (h+1)*d_head.

    Returns:
        Tensor of shape [H, d_head, n_embd], detached, CPU, original dtype.
    """
    if dims is None:
        dims = get_gpt2_dims(model)
    _check_layer(layer_idx, dims.n_layers)

    w   = _get_attn(model, layer_idx).c_proj.weight           # [n_embd, n_embd]
    W_O = w.detach().clone()                                   # [n_embd, n_embd]

    # [n_embd, n_embd] → [H, d_head, n_embd]
    return W_O.reshape(dims.n_heads, dims.d_head, dims.n_embd).contiguous()


# ──────────────────────────────────────────────────────────────────────────────
# Full-layer bulk extraction  (everything for one layer in one call)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LayerWeights:
    """
    All weight tensors needed to build one PAFTAttention or SVFAttention layer.

    Shapes (GPT-2 small, H=12, n=768, d=64):
        W_V   : [H, n_embd, d_head]    all heads' V weights
        W_O   : [H, d_head, n_embd]    all heads' O weights
        W_Q   : [n_embd, n_embd]       full Q projection (frozen in all PAFT)
        W_K   : [n_embd, n_embd]       full K projection (frozen in all PAFT)
        b_qkv : [3*n_embd]             fused QKV bias
        b_o   : [n_embd]               output projection bias
    """
    W_V:   torch.Tensor   # [H, n_embd, d_head]
    W_O:   torch.Tensor   # [H, d_head, n_embd]
    W_Q:   torch.Tensor   # [n_embd, n_embd]
    W_K:   torch.Tensor   # [n_embd, n_embd]
    b_qkv: torch.Tensor   # [3*n_embd]
    b_o:   torch.Tensor   # [n_embd]


def extract_layer(
    model,
    layer_idx: int,
    dims:      GPT2Dims | None = None,
) -> LayerWeights:
    """
    Extract all weight tensors for one transformer layer in a single call.

    Preferred over calling individual functions in a loop — accesses each
    underlying weight tensor only once and bundles the results.

    Used by PAFTModel._build_paft_attention and SVFModel's equivalent.

    Returns:
        LayerWeights dataclass with all tensors detached and CPU.
        Dtypes are preserved from the model (caller casts to fp32 as needed).
    """
    if dims is None:
        dims = get_gpt2_dims(model)
    _check_layer(layer_idx, dims.n_layers)

    attn    = _get_attn(model, layer_idx)
    w_attn  = attn.c_attn.weight.detach()    # [n_embd, 3*n_embd]
    w_proj  = attn.c_proj.weight.detach()    # [n_embd, n_embd]
    b_qkv   = attn.c_attn.bias.detach().clone()
    b_o     = attn.c_proj.bias.detach().clone()

    n = dims.n_embd
    H = dims.n_heads
    d = dims.d_head

    W_Q = w_attn[:, :n].clone()              # [n, n]
    W_K = w_attn[:, n:2*n].clone()           # [n, n]

    # V block: [n, n] → [H, n, d]
    W_V = (
        w_attn[:, 2*n:]
        .clone()
        .reshape(n, H, d)
        .permute(1, 0, 2)
        .contiguous()
    )

    # O block: [n, n] → [H, d, n]
    W_O = (
        w_proj
        .clone()
        .reshape(H, d, n)
        .contiguous()
    )

    return LayerWeights(W_V=W_V, W_O=W_O, W_Q=W_Q, W_K=W_K, b_qkv=b_qkv, b_o=b_o)