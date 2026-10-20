"""Sampled gpt-oss rollouts fast enough for the eval box: HF model, CUDA-graph decode step.

Stock HF decode is launch-bound (~75 ms/step for gpt-oss-120b at any batch). Two things make one step graph-capturable
and cheap: a cache of our own (static K/V buffers, write position held in a device tensor, masks precomputed outside
the graph and passed as the per-layer-type mask dict), and GQA attention without repeat_kv (the stock eager path copies
the whole cache 8x per layer per step). Greedy tokens match the eager path exactly; sampled
lengths match the sglang reference (engine/evalrt/rollout_check.py). B200, sampled: 13 ms/step at batch 1, ~34 at 8.
"""
from __future__ import annotations

import re
import time

import torch
from transformers.models.gpt_oss import modeling_gpt_oss

NEG = torch.finfo(torch.bfloat16).min


def gqa_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """eager_attention_forward of gpt-oss (sinks included) with the query heads of one K/V head folded into rows."""
    b, h, q, d = query.shape
    g = key.shape[1]
    r = h // g
    w = torch.matmul(query.reshape(b, g, r * q, d), key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        w = w + attention_mask.repeat(1, 1, r, 1)
    sinks = module.sinks.reshape(1, g, r, 1, 1).expand(b, g, r, q, 1).reshape(b, g, r * q, 1).to(w.dtype)
    c = torch.cat([w, sinks], dim=-1)
    c = c - c.max(dim=-1, keepdim=True).values
    p = torch.softmax(c, dim=-1, dtype=c.dtype)[..., :-1].to(value.dtype)
    return torch.matmul(p, value).reshape(b, h, q, d).transpose(1, 2).contiguous(), None


class _Cache:
    is_compileable = True

    def __init__(self, cfg, batch, length, device):
        shape = (batch, cfg.num_key_value_heads, length, cfg.head_dim)
        self.k = [torch.zeros(shape, dtype=torch.bfloat16, device=device) for _ in range(cfg.num_hidden_layers)]
        self.v = [torch.zeros_like(t) for t in self.k]
        self.pos = torch.zeros(1, dtype=torch.long, device=device)

    def update(self, k, v, layer_idx, *args, **kwargs):
        idx = self.pos + torch.arange(k.shape[2], device=k.device)
        self.k[layer_idx].index_copy_(2, idx, k)
        self.v[layer_idx].index_copy_(2, idx, v)
        return self.k[layer_idx], self.v[layer_idx]

    def get_seq_length(self, layer_idx=0):
        return 0


def rollouts(tok, model, prompt_ids: list[int] | list[list[int]], batch: int = 16, max_new: int = 3072,
             wall: float = 1e9, seed: int = 0, feat_layer: int | None = None, return_toks: bool = False,
             temperature: float = 1.0) -> dict:
    """`batch` samples (T=1, top_k=40, top_p=0.95, as the official labelling) of up to `max_new` tokens, stopping
    early after `wall` seconds. Returns per-sample lengths (-1 = not finished), the steps run, answers, timing.
    `prompt_ids` is one prompt for every row, or one prompt per row (left-padded; padding is masked per row and
    positions start at 0 on every row, so a row computes what it would alone).
    With `feat_layer`, also "feat": per sample, the residual stream after `feat_layer` decoder layers (= HF
    hidden_states[feat_layer]) averaged over generated tokens max_new-16 .. max_new-1 (the input of the length probe;
    meaningful for samples still running at the cap)."""
    t0 = time.monotonic()
    saved = modeling_gpt_oss.eager_attention_forward
    modeling_gpt_oss.eager_attention_forward = gqa_attention
    try:
        rows = prompt_ids if isinstance(prompt_ids[0], list) else [prompt_ids] * batch
        assert len(rows) == batch
        r = _rollouts(tok, model, rows, batch, max_new, wall, seed, t0, feat_layer, temperature)
        if not return_toks:
            r.pop("toks")
        return r
    finally:
        modeling_gpt_oss.eager_attention_forward = saved


def _rollouts(tok, model, rows, batch, max_new, wall, seed, t0, feat_layer=None, temperature=1.0):
    """Everything per step lives in the graph: forward, sampling (inverse CDF on pre-drawn uniforms), bookkeeping,
    position and mask updates. The host only replays and checks for completion every 128 steps."""
    cfg, dev = model.config, model.device
    P, W = max(map(len, rows)), cfg.sliding_window
    L = P + max_new + 1  # one spare slot: with features, token max_new-1 is also fed (one extra step)
    eos = model.generation_config.eos_token_id
    eos = torch.tensor(eos if isinstance(eos, list) else [eos], device=dev)
    karange = torch.arange(L, device=dev)[None, :]
    gen = torch.Generator(device=dev).manual_seed(seed)
    U = torch.rand(max_new + 1, batch, generator=gen, device=dev)
    off = torch.tensor([P - len(r) for r in rows], device=dev)  # left padding per row
    pad = (karange < off[:, None])[:, None, None]  # (batch, 1, 1, L): padding columns, masked for every query

    def masks(q_pos):
        q = q_pos[:, None]
        full = torch.where(karange <= q, 0.0, NEG).to(torch.bfloat16)[None, None].masked_fill(pad, NEG)
        slide = torch.where((karange <= q) & (q - karange < W), 0.0, NEG).to(torch.bfloat16)
        return full, slide[None, None].masked_fill(pad, NEG)

    def sample(logits, u):
        v, i = (logits.float() / temperature).topk(40, dim=-1)
        p = torch.softmax(v, dim=-1)
        p = p * ((p.cumsum(-1) - p) < 0.95)
        c = p.cumsum(-1)
        j = (c < u[:, None] * c[:, -1:]).sum(-1, keepdim=True).clamp(max=39)
        return i.gather(-1, j).squeeze(-1)

    with torch.inference_mode():
        cache = _Cache(cfg, batch, L, dev)
        ids = torch.tensor([[0] * (P - len(r)) + r for r in rows], device=dev)
        q = torch.arange(P, device=dev)
        full, slide = masks(q)
        logits = model(input_ids=ids, past_key_values=cache, use_cache=True,
                       position_ids=(q[None] - off[:, None]).clamp(min=0),
                       attention_mask={"full_attention": full, "sliding_attention": slide}, logits_to_keep=1).logits
        first = sample(logits[:, -1], U[0])
        del logits, full, slide

        pos = cache.pos  # position of the token fed next; shared with the cache's write position
        n_t = torch.zeros(1, dtype=torch.long, device=dev)  # tokens generated so far
        inp = torch.zeros(batch, 1, dtype=torch.long, device=dev)
        toks = torch.zeros(batch, max_new + 1, dtype=torch.long, device=dev)
        length = torch.zeros(batch, dtype=torch.long, device=dev)  # 0 = not finished
        done = torch.zeros(batch, dtype=torch.bool, device=dev)
        full, slide = masks(torch.tensor([P], device=dev))

        def reset():
            pos.fill_(P)
            n_t.fill_(1)
            inp.copy_(first[:, None])
            toks.zero_()
            toks[:, 0] = first
            d = torch.isin(first, eos)
            done.copy_(d)
            length.copy_(d.long())
            f, sl = masks(torch.tensor([P], device=dev))
            full.copy_(f)
            slide.copy_(sl)

        def step():
            logits = model(input_ids=inp, past_key_values=cache, use_cache=True, position_ids=pos[None] - off[:, None],
                           attention_mask={"full_attention": full, "sliding_attention": slide}).logits
            nxt = sample(logits[:, -1], U.index_select(0, n_t).view(-1))
            toks.index_copy_(1, n_t, nxt[:, None])
            new = torch.isin(nxt, eos) & ~done
            length.copy_(torch.where(new, n_t + 1, length))
            done.logical_or_(new)
            inp.copy_(nxt[:, None])
            n_t.add_(1)
            pos.add_(1)
            full.index_fill_(-1, pos, 0.0)
            slide.index_fill_(-1, pos, 0.0)
            gone = pos - W  # key leaving the window of the next query (column L-1 is still masked: harmless)
            slide.index_fill_(-1, torch.where(gone >= 0, gone, L - 1), NEG)

        hook, hsum = None, None
        if feat_layer:
            hsum = torch.zeros(batch, cfg.hidden_size, dtype=torch.float32, device=dev)

            def grab(mod, args, out):  # the step feeds token n_t-1; keep tokens max_new-16 .. max_new-1
                h = out[0] if isinstance(out, tuple) else out
                keep = ((n_t >= max_new - 15) & (n_t <= max_new)).float()
                hsum.add_(h[:, -1].float() * keep)
            hook = model.model.layers[feat_layer - 1].register_forward_hook(grab)

        reset()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                step()
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        reset()
        if hsum is not None:
            hsum.zero_()
        t_setup = time.monotonic() - t0

        n = 1
        while n < max_new + (1 if feat_layer else 0):
            graph.replay()
            n += 1
            if n % 128 == 0 and (bool(done.all()) or time.monotonic() - t0 > wall):
                break
        torch.cuda.synchronize()
        if hook is not None:
            hook.remove()
        feat = (hsum / 16).cpu() if hsum is not None else None
        toks = toks[:, :max_new].cpu()
        length = [v if 0 < v <= max_new else -1 for v in length.tolist()]
        answers = []
        for b, ln in enumerate(length):
            text = tok.decode(toks[b, :ln if ln > 0 else n].tolist())
            m = re.findall(r"ANSWER:\s*\**\s*(-?\d+|NaN)", text)
            answers.append(m[-1] if (m and ln > 0) else None)
        del cache, graph
        torch.cuda.empty_cache()
    dt = time.monotonic() - t0
    return {"len": length, "steps": n, "answers": answers, "setup_s": round(t_setup, 1), "s": round(dt, 1),
            "ms_per_step": round(1000 * (dt - t_setup) / max(1, n - 1), 1), "feat": feat, "toks": toks}
