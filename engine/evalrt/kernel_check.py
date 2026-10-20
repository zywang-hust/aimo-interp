"""Bitwise agreement of rollout_fast's Triton kernels with the stock eager ops on real decode activations.

Runs a stock HF prefill + one decode step (DynamicCache, eager attention), records per layer the decode-step inputs
of eager_attention_forward and of the RMSNorms, then feeds the same tensors to our kernels.
"""
import os
import sys

sys.path.insert(0, os.environ["EVAL_EXTRA_SITE"])
sys.path.insert(0, "/user/wangzhiyao/projects/interp/submission/fragility")
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, DynamicCache  # noqa: E402
from transformers.models.gpt_oss import modeling_gpt_oss as M  # noqa: E402
import rollout_fast as F  # noqa: E402

model = AutoModelForCausalLM.from_pretrained(sys.argv[1], device_map="auto").eval()
dev = model.device
P = int(sys.argv[2]) if len(sys.argv) > 2 else 700
torch.manual_seed(0)
ids = torch.randint(1000, 50000, (1, P + 1), device=dev)
rec = {"attn": [], "norm": []}
stock = M.eager_attention_forward


def spy(module, q, k, v, mask, scaling, **kw):
    out = stock(module, q, k, v, mask, scaling, **kw)
    if q.shape[2] == 1:
        rec["attn"].append((module, q, k, v, mask, out[0]))
    return out


def nhook(mod, args, out):
    if args[0].shape[1] == 1:
        rec["norm"].append((mod, args[0], out))


M.eager_attention_forward = spy
for L in model.model.layers:
    L.input_layernorm.register_forward_hook(nhook)
    L.post_attention_layernorm.register_forward_hook(nhook)
with torch.inference_mode():
    cache = DynamicCache(config=model.config)
    model(input_ids=ids[:, :P], past_key_values=cache, use_cache=True)
    model(input_ids=ids[:, P:], past_key_values=cache, use_cache=True)
    tot = same = 0
    worst = 0
    for li, (mod, q, k, v, mask, ref) in enumerate(rec["attn"]):
        B, H, _, D = q.shape
        G, n = k.shape[1], k.shape[2]
        win = mod.sliding_window is not None
        S = 128 if win else n + 37  # some unwritten slots past pos
        d = F._Decoder.__new__(F._Decoder)
        d.B, d.H, d.G, d.D, d.R, d.win = B, H, G, D, H // G, 128
        kc = torch.zeros(B, G, S, D, dtype=torch.bfloat16, device=dev)
        vc = torch.zeros_like(kc)
        kc[:, :, :n] = k
        vc[:, :, :n] = v
        d.pos = torch.tensor([n - 1], device=dev)
        m = B * G * (-(-S // F.CHUNK)) * 16
        d.pm, d.pacc = torch.empty(m, device=dev), torch.empty(m, D, device=dev)
        d.scw = S
        d.sc = torch.empty(B * G * 16, S, dtype=torch.bfloat16, device=dev)
        d.qr = q.reshape(B, H, D).contiguous()
        d.att = torch.empty(B, H * D, dtype=torch.bfloat16, device=dev)
        d.attn_core(kc, vc, win, mod.sinks, mod.scaling)
        got = d.att.view(B, 1, H, D)
        eq = (got == ref).float().mean().item()
        tot += 1
        same += eq
        worst = max(worst, ((got.float() - ref.float()).abs() / ref.float().abs().clamp(min=1e-2)).max().item())
        if li < 4:
            print(f"layer {li} ({'slide' if win else 'full'}, n={n}): bitwise-equal fraction {eq:.4f}",
                  flush=True)
    print(f"attention: mean bitwise-equal fraction {same / tot:.4f}, worst rel diff {worst:.3g}",
          flush=True)
    eqs = []
    for mod, x, ref in rec["norm"]:
        out = torch.empty_like(x.view(-1, x.shape[-1]))
        F._add_rmsnorm_kernel[(x.shape[0],)](x.view(-1, x.shape[-1]), x, x, mod.weight, out, mod.variance_epsilon,
                                             N=x.shape[-1], BLOCK=4096, HAS_B=False, num_warps=4)
        eqs.append((out == ref.view_as(out)).float().mean().item())
    print(f"rmsnorm: mean bitwise-equal fraction {sum(eqs) / len(eqs):.5f}, min {min(eqs):.5f}", flush=True)
