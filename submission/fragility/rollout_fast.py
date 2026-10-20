"""Sampled gpt-oss rollouts fast enough for the eval box: HF weights, our own decode step, one CUDA graph per step.

v1 (rollout.py) captures the stock HF forward (static cache, precomputed masks) in a CUDA graph. Same API and return dict as v1 (one prompt for every row; per-row prompts go to v1).
This version keeps the HF prefill but runs the decode step itself over the same weights:
- MoE: the same triton matmul_ogs kernels; on sm_100 with opt flags for decode-sized M (block_m 16, block_k 64,
  block_n 128, split_k 1; see _MoeFlags), since the library forces block_m >= 64 there (~4x slower at 4-16 tokens).
  Elsewhere (the eval GPU, sm_120) the library defaults, which are faster there.
- attention (Triton): reads only the min(pos+1, S) keys written so far (pos is a device tensor, so one graph serves
  every position); sliding layers keep a 128-slot ring buffer. Scores, max-subtraction and probabilities are rounded
  to bf16 where eager_attention_forward rounds them (expf, IEEE division), so outputs match it bitwise in 99.99% of
  elements (engine/evalrt/kernel_check.py). No masks, no repeat_kv, no cat with the sinks.
- q/k/v projections in one Triton launch; RMSNorm (+ the residual add before it) and RoPE (+ the K/V cache write) are
  single Triton kernels rounding like the eager bf16 code (RMSNorm bitwise equal).
- the prompt is prefilled once (batch 1) and its K/V broadcast to the batch.
~710 kernels per step instead of ~2250. Sampling (top-k 40 / top-p 0.95 inverse CDF on pre-drawn uniforms) and
bookkeeping are unchanged and stay inside the graph. The first decode step is checked against an HF forward; any
failure (e.g. a Triton config that does not compile on another GPU) falls back to the library MoE flags, then to v1.
feat_layer: the decode step adds the residual stream entering layer `feat_layer` (= HF hidden_states[feat_layer])
into a sum inside the graph, as v1's forward hook does.
Checks: engine/evalrt/{greedy_gate,logit_check,kernel_check,rollout_check,feat_check}.py.
"""
from __future__ import annotations

import dataclasses
import os
import re
import time

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from transformers.models.gpt_oss import modeling_gpt_oss

NEG = torch.finfo(torch.bfloat16).min
CHUNK = 256  # keys per split of the full-attention decode kernel
CHECK_TV = 0.2  # total variation between the first decode step's next-token distribution and the HF forward's


# ----------------------------------------------------------------------------------------------- triton kernels
@triton.jit
def _add_rmsnorm_kernel(a_ptr, b_ptr, res_ptr, w_ptr, out_ptr, eps, N: tl.constexpr, BLOCK: tl.constexpr,
                        HAS_B: tl.constexpr):
    """res = a + b (bf16); out = bf16(w * (res * rsqrt(mean(res^2) + eps))) with fp32 math, as GptOssRMSNorm."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    m = offs < N
    x = tl.load(a_ptr + row * N + offs, mask=m, other=0.0)
    if HAS_B:
        x = (x.to(tl.float32) + tl.load(b_ptr + row * N + offs, mask=m, other=0.0).to(tl.float32)).to(tl.bfloat16)
        tl.store(res_ptr + row * N + offs, x, mask=m)
    xf = x.to(tl.float32)
    var = tl.sum(xf * xf, axis=0) / N
    y = xf * tl.math.rsqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=m, other=0.0).to(tl.float32)
    tl.store(out_ptr + row * N + offs, (w * y).to(tl.bfloat16), mask=m)


@triton.jit
def _rot(x1, x2, c, s):
    """gpt-oss _apply_rotary_emb on bf16 halves, rounding to bf16 after every op like the eager code."""
    a = (x1.to(tl.float32) * c).to(tl.bfloat16).to(tl.float32)
    b = (x2.to(tl.float32) * s).to(tl.bfloat16).to(tl.float32)
    o1 = (a - b).to(tl.bfloat16)
    a = (x2.to(tl.float32) * c).to(tl.bfloat16).to(tl.float32)
    b = (x1.to(tl.float32) * s).to(tl.bfloat16).to(tl.float32)
    o2 = (a + b).to(tl.bfloat16)
    return o1, o2


@triton.jit
def _rope_store_kernel(q_ptr, k_ptr, v_ptr, cos_ptr, sin_ptr, pos_ptr, qo_ptr, kc_ptr, vc_ptr, S,
                       H: tl.constexpr, G: tl.constexpr, D: tl.constexpr):
    """Rotate q (B, H*D) into qo (B, H, D); rotate k and copy v of (B, G*D) into the caches (B, G, S, D) at slot
    pos % S (S = window for the ring buffers, >= every position for the full caches)."""
    b = tl.program_id(0)
    h = tl.program_id(1)
    half = tl.arange(0, D // 2)
    c = tl.load(cos_ptr + half).to(tl.float32)
    s = tl.load(sin_ptr + half).to(tl.float32)
    if h < H:
        base = q_ptr + b * H * D + h * D
        o1, o2 = _rot(tl.load(base + half), tl.load(base + D // 2 + half), c, s)
        out = qo_ptr + b * H * D + h * D
        tl.store(out + half, o1)
        tl.store(out + D // 2 + half, o2)
    else:
        g = h - H
        slot = tl.load(pos_ptr) % S
        base = k_ptr + b * G * D + g * D
        o1, o2 = _rot(tl.load(base + half), tl.load(base + D // 2 + half), c, s)
        out = kc_ptr + ((b * G + g) * S + slot) * D
        tl.store(out + half, o1)
        tl.store(out + D // 2 + half, o2)
        d = tl.arange(0, D)
        tl.store(vc_ptr + ((b * G + g) * S + slot) * D + d, tl.load(v_ptr + b * G * D + g * D + d))


@triton.jit
def _gemv_seg(x_ptr, w_ptr, b_ptr, o_ptr, M, pid, N: tl.constexpr, K: tl.constexpr, BN: tl.constexpr,
              BK: tl.constexpr):
    """o[:M, pid*BN:(pid+1)*BN] = bf16(x @ w^T + b) for M <= 16 rows (fp32 accumulation, bias added in fp32)."""
    rn = pid * BN + tl.arange(0, BN)
    rm = tl.arange(0, 16)
    nm = rn < N
    acc = tl.zeros([16, BN], tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        km = rk < K
        x = tl.load(x_ptr + rm[:, None] * K + rk[None, :], mask=(rm < M)[:, None] & km[None, :], other=0.0)
        w = tl.load(w_ptr + rn[:, None] * K + rk[None, :], mask=nm[:, None] & km[None, :], other=0.0)
        acc += tl.dot(x, tl.trans(w))
    b = tl.load(b_ptr + rn, mask=nm, other=0.0).to(tl.float32)
    tl.store(o_ptr + rm[:, None] * N + rn[None, :], (acc + b[None, :]).to(tl.bfloat16),
             mask=(rm < M)[:, None] & nm[None, :])


@triton.jit
def _qkv_kernel(x_ptr, wq, bq, oq, wk, bk, ok, wv, bv, ov, M, NQ: tl.constexpr, NKV: tl.constexpr, K: tl.constexpr,
                BN: tl.constexpr, BK: tl.constexpr):
    """q, k, v projections of M <= 16 rows in one launch (cuBLAS takes three GEMMs + split-K reductions)."""
    pid = tl.program_id(0)
    tq: tl.constexpr = NQ // BN
    tkv: tl.constexpr = NKV // BN
    if pid < tq:
        _gemv_seg(x_ptr, wq, bq, oq, M, pid, NQ, K, BN, BK)
    elif pid < tq + tkv:
        _gemv_seg(x_ptr, wk, bk, ok, M, pid - tq, NKV, K, BN, BK)
    else:
        _gemv_seg(x_ptr, wv, bv, ov, M, pid - tq - tkv, NKV, K, BN, BK)


@triton.jit
def _bf(x):
    """Round fp32 to bf16 (and back), i.e. what storing an eager bf16 intermediate does."""
    return x.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _scores(q, kb, cols, cm, d, scaling, D: tl.constexpr):
    """bf16(bf16(q . k) * scaling) as eager `matmul(q, k^T) * scaling`; -inf beyond the written keys."""
    k = tl.load(kb + cols[:, None] * D + d[None, :], mask=cm[:, None], other=0.0)
    s = _bf(_bf(tl.dot(q, tl.trans(k))) * scaling)
    return tl.where(cm[None, :], s, float("-inf"))


@triton.jit
def _attn_window_kernel(q_ptr, k_ptr, v_ptr, sink_ptr, pos_ptr, out_ptr, scaling, S: tl.constexpr, G: tl.constexpr,
                        R: tl.constexpr, D: tl.constexpr):
    """Sliding layers, one (batch row, K/V head): the R query heads against the <= S ring slots, eager numerics:
    c = bf16(s - max(s, sink)), p = bf16(expf(c) / sum), out = bf16(p @ v)."""
    bg = tl.program_id(0)
    n = tl.minimum(tl.load(pos_ptr) + 1, S)
    rows = tl.arange(0, 16)
    rmask = rows < R
    d = tl.arange(0, D)
    q = tl.load(q_ptr + (bg * R + rows)[:, None] * D + d[None, :], mask=rmask[:, None], other=0.0)
    cols = tl.arange(0, S)
    cm = cols < n
    s = _scores(q, k_ptr + bg * S * D, cols, cm, d, scaling, D)
    sink = _bf(tl.load(sink_ptr + (bg % G) * R + rows, mask=rmask, other=0.0).to(tl.float32))
    mx = tl.maximum(tl.max(s, axis=1), sink)
    e = tl.where(cm[None, :], libdevice.exp(_bf(s - mx[:, None])), 0.0)
    z = tl.sum(e, axis=1) + libdevice.exp(_bf(sink - mx))
    p = tl.math.div_rn(e, z[:, None]).to(tl.bfloat16)
    v = tl.load(v_ptr + bg * S * D + cols[:, None] * D + d[None, :], mask=cm[:, None], other=0.0)
    o = tl.dot(p, v)
    tl.store(out_ptr + (bg * R + rows)[:, None] * D + d[None, :], o.to(tl.bfloat16), mask=rmask[:, None])


@triton.jit
def _attn_scores_kernel(q_ptr, k_ptr, pos_ptr, sc_ptr, pm_ptr, scaling, S, NS, SCW, R: tl.constexpr,
                        D: tl.constexpr, CH: tl.constexpr, BN: tl.constexpr):
    """Full layers, pass 1 per (batch row, K/V head, key chunk): bf16 scores into sc (row stride SCW), chunk max."""
    bg = tl.program_id(0)
    sp = tl.program_id(1)
    n = tl.minimum(tl.load(pos_ptr) + 1, S)
    rows = tl.arange(0, 16)
    d = tl.arange(0, D)
    q = tl.load(q_ptr + (bg * R + rows)[:, None] * D + d[None, :], mask=(rows < R)[:, None], other=0.0)
    m = tl.full([16], float("-inf"), tl.float32)
    start = sp * CH
    for off in range(start, tl.minimum(start + CH, n), BN):
        cols = off + tl.arange(0, BN)
        cm = cols < n
        s = _scores(q, k_ptr + bg * S * D, cols, cm, d, scaling, D)
        tl.store(sc_ptr + (bg * 16 + rows)[:, None] * SCW + cols[None, :], s.to(tl.bfloat16), mask=cm[None, :])
        m = tl.maximum(m, tl.max(s, axis=1))
    tl.store(pm_ptr + (bg * NS + sp) * 16 + rows, m)


@triton.jit
def _attn_pv_kernel(v_ptr, sink_ptr, pos_ptr, sc_ptr, pm_ptr, out_ptr, pacc_ptr, S, NS, SCW, G: tl.constexpr,
                    R: tl.constexpr, D: tl.constexpr, CH: tl.constexpr, BN: tl.constexpr, NSP: tl.constexpr,
                    ZB: tl.constexpr):
    """Full layers, pass 2 per (batch row, K/V head, key chunk): global max (with the sink) and normaliser over all
    scores (every chunk recomputes the same sum in the same order), then this chunk's bf16(p) @ v into pacc."""
    bg = tl.program_id(0)
    sp = tl.program_id(1)
    n = tl.minimum(tl.load(pos_ptr) + 1, S)
    rows = tl.arange(0, 16)
    rmask = rows < R
    d = tl.arange(0, D)
    spl = tl.arange(0, NSP)
    pm = tl.load(pm_ptr + (bg * NS + spl)[:, None] * 16 + rows[None, :], mask=(spl < NS)[:, None],
                 other=float("-inf"))
    sink = _bf(tl.load(sink_ptr + (bg % G) * R + rows, mask=rmask, other=0.0).to(tl.float32))
    mx = tl.maximum(tl.max(pm, axis=0), sink)
    sc = sc_ptr + (bg * 16 + rows)[:, None] * SCW
    z = tl.zeros([16], tl.float32)
    for off in range(0, n, ZB):
        cols = off + tl.arange(0, ZB)
        cm = cols < n
        s = tl.load(sc + cols[None, :], mask=cm[None, :], other=float("-inf")).to(tl.float32)
        z += tl.sum(tl.where(cm[None, :], libdevice.exp(_bf(s - mx[:, None])), 0.0), axis=1)
    z += libdevice.exp(_bf(sink - mx))
    acc = tl.zeros([16, D], tl.float32)
    start = sp * CH
    for off in range(start, tl.minimum(start + CH, n), BN):
        cols = off + tl.arange(0, BN)
        cm = cols < n
        s = tl.load(sc + cols[None, :], mask=cm[None, :], other=float("-inf")).to(tl.float32)
        e = tl.where(cm[None, :], libdevice.exp(_bf(s - mx[:, None])), 0.0)
        p = tl.math.div_rn(e, z[:, None]).to(tl.bfloat16)
        v = tl.load(v_ptr + bg * S * D + cols[:, None] * D + d[None, :], mask=cm[:, None], other=0.0)
        acc += tl.dot(p, v)
    tl.store(pacc_ptr + ((bg * NS + sp) * 16 + rows)[:, None] * D + d[None, :], acc)


@triton.jit
def _attn_sum_kernel(out_ptr, pacc_ptr, NS, R: tl.constexpr, D: tl.constexpr, NSP: tl.constexpr):
    """Full layers, pass 3 per (batch row, query head): bf16 of the summed chunk partials of p @ v."""
    bh = tl.program_id(0)
    spl = tl.arange(0, NSP)
    d = tl.arange(0, D)
    acc = tl.load(pacc_ptr + (((bh // R) * NS + spl) * 16 + bh % R)[:, None] * D + d[None, :],
                  mask=(spl < NS)[:, None], other=0.0)
    tl.store(out_ptr + bh * D + d, tl.sum(acc, axis=0).to(tl.bfloat16))


# ----------------------------------------------------------------------------------------------- prefill (HF path)
def gqa_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """eager_attention_forward of gpt-oss (sinks included) with the query heads of one K/V head folded into rows;
    the (1, 1, q, n) mask is broadcast instead of repeated."""
    b, h, q, d = query.shape
    g = key.shape[1]
    r = h // g
    w = torch.matmul(query.reshape(b, g, r * q, d), key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        w = (w.view(b, g, r, q, -1) + attention_mask[:, :, None]).view(b, g, r * q, -1)
    sinks = module.sinks.reshape(1, g, r, 1, 1).expand(b, g, r, q, 1).reshape(b, g, r * q, 1).to(w.dtype)
    c = torch.cat([w, sinks], dim=-1)
    c = c - c.max(dim=-1, keepdim=True).values
    p = torch.softmax(c, dim=-1, dtype=c.dtype)[..., :-1].to(value.dtype)
    return torch.matmul(p, value).reshape(b, h, q, d).transpose(1, 2).contiguous(), None


class _PrefillCache:
    """Takes the prompt's K/V (batch 1) from the HF forward and broadcasts it into the decode caches."""
    is_compileable = False

    def __init__(self, kc, vc):
        self.kc, self.vc = kc, vc

    def update(self, k, v, layer_idx, *args, **kwargs):
        kc, vc = self.kc[layer_idx], self.vc[layer_idx]
        P, S = k.shape[2], kc.shape[2]
        p = torch.arange(max(0, P - S), P, device=k.device)
        kc[:, :, p % S] = k[:, :, p]
        vc[:, :, p % S] = v[:, :, p]
        return k, v

    def get_seq_length(self, layer_idx=0):
        return 0


# ----------------------------------------------------------------------------------------------- decode step
class _Decoder:
    """Decode step over the HF modules' weights for `batch` rows; all inputs and outputs are static buffers.
    With feat_layer set, on_feat(h) is called with h (B, hidden) bf16 = the input of layer feat_layer, i.e. HF
    hidden_states[feat_layer] (the output of decoder layer feat_layer - 1, residual included)."""
    feat_layer: int | None = None
    on_feat = None

    def __init__(self, model, batch, length):
        cfg, dev = model.config, model.device
        self.model, self.B, self.dev = model, batch, dev
        self.H, self.G, self.D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        self.R = self.H // self.G
        self.hid = cfg.hidden_size
        self.layers = model.model.layers
        self.win = win = cfg.sliding_window
        self.S = [win if t == "sliding_attention" else length for t in cfg.layer_types]
        bf = torch.bfloat16
        self.kc = [torch.zeros(batch, self.G, s, self.D, dtype=bf, device=dev) for s in self.S]
        self.vc = [torch.zeros_like(t) for t in self.kc]
        self.pos = torch.zeros(1, dtype=torch.long, device=dev)
        n = batch * self.G * triton.cdiv(length, CHUNK) * 16
        self.pm = torch.empty(n, dtype=torch.float32, device=dev)
        self.pacc = torch.empty(n, self.D, dtype=torch.float32, device=dev)
        self.scw = length
        self.sc = torch.empty(batch * self.G * 16, length, dtype=bf, device=dev)
        self.res = [torch.empty(batch, self.hid, dtype=bf, device=dev) for _ in range(2)]
        self.qr = torch.empty(batch, self.H, self.D, dtype=bf, device=dev)
        self.q = torch.empty(batch, self.H * self.D, dtype=bf, device=dev)
        self.k = torch.empty(batch, self.G * self.D, dtype=bf, device=dev)
        self.v = torch.empty(batch, self.G * self.D, dtype=bf, device=dev)
        self.att = torch.empty(batch, self.H * self.D, dtype=bf, device=dev)
        self.xn = torch.empty(batch, self.hid, dtype=bf, device=dev)

    def _norm(self, a, b, res, norm):
        _add_rmsnorm_kernel[(self.B,)](a, b if b is not None else a, res, norm.weight, self.xn, norm.variance_epsilon,
                                       N=self.hid, BLOCK=triton.next_power_of_2(self.hid), HAS_B=b is not None,
                                       num_warps=4)
        return self.xn

    def _attention(self, i, attn, x, cos, sin):
        if self.B <= 16:
            q, k, v = self.q, self.k, self.v
            nq, nkv = self.H * self.D, self.G * self.D
            _qkv_kernel[((nq + 2 * nkv) // 16,)](x, attn.q_proj.weight, attn.q_proj.bias, q, attn.k_proj.weight,
                                                 attn.k_proj.bias, k, attn.v_proj.weight, attn.v_proj.bias, v, self.B,
                                                 NQ=nq, NKV=nkv, K=self.hid, BN=16, BK=128, num_warps=2, num_stages=4)
        else:
            q, k, v = attn.q_proj(x), attn.k_proj(x), attn.v_proj(x)
        _rope_store_kernel[(self.B, self.H + self.G)](q, k, v, cos, sin, self.pos, self.qr, self.kc[i], self.vc[i],
                                                      self.S[i], H=self.H, G=self.G, D=self.D, num_warps=1)
        self.attn_core(self.kc[i], self.vc[i], self.S[i] == self.win, attn.sinks, attn.scaling)
        return attn.o_proj(self.att)

    def attn_core(self, kc, vc, window, sinks, scaling):
        """self.qr (B, H, D) against the first min(pos+1, S) slots of kc/vc (B, G, S, D) -> self.att (B, H*D)."""
        B, H, G, D, R, S = self.B, self.H, self.G, self.D, self.R, kc.shape[2]
        if window:
            _attn_window_kernel[(B * G,)](self.qr, kc, vc, sinks, self.pos, self.att, scaling, S=S, G=G, R=R, D=D,
                                          num_warps=4)
            return
        ns, nsp = triton.cdiv(S, CHUNK), max(16, triton.next_power_of_2(triton.cdiv(S, CHUNK)))
        _attn_scores_kernel[(B * G, ns)](self.qr, kc, self.pos, self.sc, self.pm, scaling, S, ns, self.scw, R=R,
                                         D=D, CH=CHUNK, BN=64, num_warps=4)
        _attn_pv_kernel[(B * G, ns)](vc, sinks, self.pos, self.sc, self.pm, self.att, self.pacc, S, ns, self.scw,
                                     G=G, R=R, D=D, CH=CHUNK, BN=64, NSP=nsp, ZB=256, num_warps=4)
        _attn_sum_kernel[(B * H,)](self.att, self.pacc, ns, R=R, D=D, NSP=nsp, num_warps=1)

    def __call__(self, inp):
        """inp (B, 1) token ids at position self.pos -> logits (B, vocab) bf16; writes K/V at self.pos."""
        m = self.model.model
        h = m.embed_tokens(inp).view(self.B, self.hid)
        cos, sin = m.rotary_emb(h, self.pos[None])
        res, other = self.res
        res.copy_(h)
        delta = None
        for i, layer in enumerate(self.layers):
            x = self._norm(res, delta, other, layer.input_layernorm)
            if delta is not None:
                res, other = other, res
            if i == self.feat_layer and self.on_feat is not None:
                self.on_feat(res)  # res = bf16(residual + mlp out) of layer i-1, as HF's decoder layer returns it
            o = self._attention(i, layer.self_attn, x, cos, sin)
            x = self._norm(res, o, other, layer.post_attention_layernorm)
            res, other = other, res
            delta = layer.mlp(x.view(self.B, 1, self.hid))[0].view(self.B, self.hid)
        x = self._norm(res, delta, other, m.norm)
        return self.model.lm_head(x)


# ----------------------------------------------------------------------------------------------- MoE flags
class _MoeFlags:
    """matmul_ogs opt flags for decode-sized M on sm_100 (B200 sweeps over all OptFlags, NOTES.md §4.2): block_m 16,
    block_k 64 (128 at batch <= 2), block_n 128, split_k 1. The library default (block_k 256, block_n 256; block_m
    64 on sm_100) is ~4x slower at 4-16 tokens. split_k 2 would be another ~10% faster at batch 4 but sums the
    experts' outputs in fp32 instead of rounding each to bf16 first, which moves the logits away from the stock path
    (engine/evalrt/logit_check.py: drift 1.3-1.7x the reference's own batch-size noise; split_k 1 sits at it).
    restore() undoes everything."""

    def __init__(self):
        from transformers.integrations import mxfp4
        hub = mxfp4.triton_kernels_hub
        self.of, self.mo = hub.matmul_ogs_details.opt_flags, hub.matmul_ogs
        self.saved, self.orig = dict(self.of._opt_flags_constraints), self.mo.make_opt_flags

    def apply(self, batch) -> str:
        """Returns the label logged in "check": "decode" (our flags) or "default" (the library's)."""
        self.restore()
        if os.environ.get("ROLLOUT_MOE_DEFAULT"):
            return "default"
        if torch.cuda.get_device_capability()[0] != 10 and not os.environ.get("ROLLOUT_MOE_FLAGS"):
            # only sm_100 needs them: the library forces block_m >= 64 there. On the eval GPU (sm_120) its defaults
            # (block_m 16, 256x256 tiles, split_k 2-3) run 15.6 ms/step at B=12 against 28.3 with ours (NOTES §4.2)
            return "default"
        bm, bk, sk, bn = 16, 128 if batch <= 2 else 64, 1, 128
        if os.environ.get("ROLLOUT_MOE_FLAGS"):  # experiments: "block_m,block_k,split_k,block_n" (block_n 0 = default)
            bm, bk, sk, bn = map(int, os.environ["ROLLOUT_MOE_FLAGS"].split(","))
        self.of.reset_opt_flags_constraints()
        self.of.update_opt_flags_constraints({"block_m": bm, "block_k": bk, "split_k": sk})
        if bn:
            orig = self.orig
            self.mo.make_opt_flags = lambda *a, **k: dataclasses.replace(orig(*a, **k), block_n=bn)
        return "decode"

    def restore(self):
        self.mo.make_opt_flags = self.orig
        self.of.reset_opt_flags_constraints()
        self.of.update_opt_flags_constraints(self.saved)


# ----------------------------------------------------------------------------------------------- entry point
def _v1():
    """v1: rollout.py next to this file."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("rollout", os.path.join(os.path.dirname(__file__), "rollout.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rollouts(tok, model, prompt_ids: list[int] | list[list[int]], batch: int = 16, max_new: int = 3072,
             wall: float = 1e9, seed: int = 0, feat_layer: int | None = None, return_toks: bool = False,
             temperature: float = 1.0, greedy: bool = False) -> dict:
    """Drop-in for v1's rollouts (same arguments and return dict: len, steps, answers, setup_s, s, ms_per_step, feat,
    plus "check" = the first-step comparison, and "fallback" if v1 ran instead).
    `batch` samples (T=1, top_k=40, top_p=0.95, as the official labelling) of up to `max_new` tokens, stopping
    early after `wall` seconds. Returns per-sample lengths (-1 = not finished), the steps run, answers, timing.
    With `feat_layer`, also "feat": per sample, the residual stream after `feat_layer` decoder layers (= HF
    hidden_states[feat_layer]) averaged over generated tokens max_new-16 .. max_new-1 (the length probe's input).
    greedy=True takes the argmax instead (for checks against the HF eager path). Per-row prompts (a list of lists)
    go to v1. If the fast decode step fails to build or disagrees with the HF forward (an untested GPU), falls back to
    v1 next to this file."""
    t0 = time.monotonic()
    if isinstance(prompt_ids[0], list):
        r = _v1().rollouts(tok, model, prompt_ids, batch=batch, max_new=max_new, wall=wall, seed=seed,
                           feat_layer=feat_layer, return_toks=return_toks, temperature=temperature)
        return {**r, "fallback": "per-row prompts"}
    saved = modeling_gpt_oss.eager_attention_forward
    modeling_gpt_oss.eager_attention_forward = gqa_attention
    moe = _MoeFlags()
    try:
        r = _rollouts(tok, model, prompt_ids, batch, max_new, wall, seed, greedy, t0, moe, feat_layer, temperature)
        if not (return_toks or greedy):
            r.pop("toks")
        return r
    except Exception as ex:  # noqa: BLE001
        if greedy or os.environ.get("ROLLOUT_NO_FALLBACK"):
            raise
        why = f"{type(ex).__name__}: {ex}"[:300]
        print(f"rollout: fast decode failed ({why}); falling back to v1", flush=True)
    finally:
        modeling_gpt_oss.eager_attention_forward = saved
        moe.restore()
    torch.cuda.empty_cache()
    r = _v1().rollouts(tok, model, prompt_ids, batch=batch, max_new=max_new,
                       wall=max(1.0, wall - (time.monotonic() - t0)), seed=seed, feat_layer=feat_layer,
                       return_toks=return_toks, temperature=temperature)
    return {**r, "fallback": why}


def _rollouts(tok, model, prompt_ids, batch, max_new, wall, seed, greedy, t0, moe, feat_layer=None, temperature=1.0):
    """Everything per step lives in the graph: forward, sampling (inverse CDF on pre-drawn uniforms), bookkeeping,
    position update, feature sum. The host only replays and checks for completion every 128 steps."""
    if os.environ.get("ROLLOUT_TEST_FALLBACK"):
        raise RuntimeError("ROLLOUT_TEST_FALLBACK set")
    cfg, dev = model.config, model.device
    if feat_layer and not 0 < feat_layer < cfg.num_hidden_layers:
        raise ValueError(f"feat_layer {feat_layer} not in 1..{cfg.num_hidden_layers - 1}")
    extra = 1 if feat_layer else 0  # with features, token max_new-1 is also fed (one extra step), as in v1
    P, W = len(prompt_ids), cfg.sliding_window
    L = P + max_new + extra
    eos = model.generation_config.eos_token_id
    eos = torch.tensor(eos if isinstance(eos, list) else [eos], device=dev)
    gen = torch.Generator(device=dev).manual_seed(seed)
    U = torch.rand(max_new + 1, batch, generator=gen, device=dev)

    def sample(logits, u):
        if greedy:
            return logits.float().argmax(-1)
        v, i = (logits.float() / temperature).topk(40, dim=-1)
        p = torch.softmax(v, dim=-1)
        p = p * ((p.cumsum(-1) - p) < 0.95)
        c = p.cumsum(-1)
        j = (c < u[:, None] * c[:, -1:]).sum(-1, keepdim=True).clamp(max=39)
        return i.gather(-1, j).squeeze(-1)

    with torch.inference_mode():
        dec = _Decoder(model, batch, L)
        ids = torch.tensor([prompt_ids], device=dev)
        q = torch.arange(P, device=dev)
        k = torch.arange(P, device=dev)[None, :]
        full = torch.where(k <= q[:, None], 0.0, NEG).to(torch.bfloat16)[None, None]
        slide = torch.where((k <= q[:, None]) & (q[:, None] - k < W), 0.0, NEG).to(torch.bfloat16)[None, None]
        logits = model(input_ids=ids, past_key_values=_PrefillCache(dec.kc, dec.vc), use_cache=True,
                       position_ids=q[None], attention_mask={"full_attention": full, "sliding_attention": slide},
                       logits_to_keep=1).logits
        first = sample(logits[:, -1].expand(batch, -1), U[0])
        del logits, full, slide
        # reference for checking the hand-written decode step: the HF forward over the prompt + row 0's first token
        q1 = torch.arange(P + 1, device=dev)
        k1 = q1[None, :]
        full = torch.where(k1 <= q1[:, None], 0.0, NEG).to(torch.bfloat16)[None, None]
        slide = torch.where((k1 <= q1[:, None]) & (q1[:, None] - k1 < W), 0.0, NEG).to(torch.bfloat16)[None, None]
        ref = model(input_ids=torch.cat([ids, first[:1, None]], 1), position_ids=q1[None], use_cache=False,
                    attention_mask={"full_attention": full, "sliding_attention": slide},
                    logits_to_keep=1).logits[0, -1].float()
        del full, slide
        # warm-up and capture write positions P.. into the caches; the ring buffers still hold prompt keys there
        rings = [(i, dec.kc[i].clone(), dec.vc[i].clone()) for i, S in enumerate(dec.S) if S < L]

        pos = dec.pos  # position of the token fed next
        n_t = torch.zeros(1, dtype=torch.long, device=dev)  # tokens generated so far
        inp = torch.zeros(batch, 1, dtype=torch.long, device=dev)
        toks = torch.zeros(batch, max_new + extra, dtype=torch.long, device=dev)
        length = torch.zeros(batch, dtype=torch.long, device=dev)  # 0 = not finished
        done = torch.zeros(batch, dtype=torch.bool, device=dev)
        hsum = None
        if feat_layer:
            hsum = torch.zeros(batch, cfg.hidden_size, dtype=torch.float32, device=dev)

            def grab(h):  # the step feeds token n_t-1; keep tokens max_new-16 .. max_new-1 (as v1's hook)
                keep = ((n_t >= max_new - 15) & (n_t <= max_new)).float()
                hsum.add_(h.float() * keep)
            dec.feat_layer, dec.on_feat = feat_layer, grab

        def reset():
            for i, kc, vc in rings:
                dec.kc[i].copy_(kc)
                dec.vc[i].copy_(vc)
            pos.fill_(P)
            n_t.fill_(1)
            inp.copy_(first[:, None])
            toks.zero_()
            toks[:, 0] = first
            d = torch.isin(first, eos)
            done.copy_(d)
            length.copy_(d.long())

        def step():
            nxt = sample(dec(inp), U.index_select(0, n_t).view(-1))
            toks.index_copy_(1, n_t, nxt[:, None])
            new = torch.isin(nxt, eos) & ~done
            length.copy_(torch.where(new, n_t + 1, length))
            done.logical_or_(new)
            inp.copy_(nxt[:, None])
            n_t.add_(1)
            pos.add_(1)

        # eager first step (compiles the Triton kernels), checked against the HF forward; with the MoE flags for this
        # GPU first (decode flags on sm_100), then the library defaults. Writes position P, which the replays write again.
        check = None
        for flags in dict.fromkeys((moe.apply(batch), "default")):
            if flags == "default":
                moe.restore()
            else:
                moe.apply(batch)
            reset()
            try:
                lg = dec(inp)[0].float()
            except Exception as ex:  # noqa: BLE001
                print(f"rollout: decode step with {flags} MoE flags failed ({type(ex).__name__}: {ex})", flush=True)
                continue
            dmax = (lg - ref).abs().max().item()
            tv = 0.5 * (torch.softmax(lg, -1) - torch.softmax(ref, -1)).abs().sum().item()
            if tv < CHECK_TV and bool((ref.topk(5).indices == lg.argmax()).any()):
                check = {"tv": round(tv, 4), "dlogit": round(dmax, 3), "moe_flags": flags}
                break
            print(f"rollout: decode step with {flags} MoE flags departs from the HF forward (total variation {tv:.3f}, "
                  f"max |dlogit| {dmax:.3f})", flush=True)
        if check is None:
            raise RuntimeError("rollout_fast: decode step disagrees with the HF forward")
        del ref, lg
        reset()
        step()
        reset()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                step()
        torch.cuda.current_stream().wait_stream(s)
        reset()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        reset()
        if hsum is not None:
            hsum.zero_()
        t_setup = time.monotonic() - t0

        n = 1
        while n < max_new + extra:
            graph.replay()
            n += 1
            if n % 128 == 0 and (bool(done.all()) or time.monotonic() - t0 > wall):
                break
        torch.cuda.synchronize()
        feat = (hsum / 16).cpu() if hsum is not None else None
        length = [v if 0 < v <= max_new else -1 for v in length.tolist()]
        answers = []
        for b, ln in enumerate(length):
            text = tok.decode(toks[b, :ln if ln > 0 else min(n, max_new)].tolist())
            m = re.findall(r"ANSWER:\s*\**\s*(-?\d+|NaN)", text)
            answers.append(m[-1] if (m and ln > 0) else None)
        out = {"len": length, "steps": n, "answers": answers, "check": check, "feat": feat,
               "toks": toks[:, :n].tolist() if greedy else toks[:, :max_new].cpu()}
        del dec, graph
        torch.cuda.empty_cache()
    dt = time.monotonic() - t0
    return {**out, "setup_s": round(t_setup, 1), "s": round(dt, 1),
            "ms_per_step": round(1000 * (dt - t_setup) / max(1, n - 1), 1)}
