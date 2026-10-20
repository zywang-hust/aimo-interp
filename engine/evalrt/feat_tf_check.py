"""Same-tokens check of v2's feat_layer: teacher-force one sampled stream through (a) one HF forward over the whole
sequence (how the probe's training features were made, trace_feats.py), (b) rollout_fast's decoder step by step and
(c) stock HF eager decode with a DynamicCache step by step (the stock decode path v1 captures), and compare the
layer-L states of the last 16 tokens, per token and averaged (the probe input).
  GPU=<id> engine/evalrt/run_eval.sh engine/evalrt/feat_tf_check.py <model-path> <pid> [max_new] [batch] [n_worst]
The streams are v2's own samples (rollouts(..., feat_layer=L), deterministic), the n_worst of them by cos(feature,
(a)) plus the best one. (b) and (c) run at `batch` identical rows, as in the rollout.
FT_MOE_DEFAULT=1: (c) is replaced by (b) with the library-default MoE flags (is a deviation from the decode flags?).
"""
import json
import os
import sys

sys.path.insert(0, os.environ["EVAL_EXTRA_SITE"])
ROOT = "/user/wangzhiyao/projects/interp"
sys.path.insert(0, f"{ROOT}/submission/fragility")
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache  # noqa: E402
from transformers.models.gpt_oss import modeling_gpt_oss  # noqa: E402
import rollout_fast as F  # noqa: E402

path, pid = sys.argv[1], sys.argv[2]
T = int(sys.argv[3]) if len(sys.argv) > 3 else 1536
B = int(sys.argv[4]) if len(sys.argv) > 4 else 12
NW = int(sys.argv[5]) if len(sys.argv) > 5 else 3
LAYER = 18
SOLVE = open(f"{ROOT}/ext/baselines/prompts/solve.txt").read().strip()
probs = {r["id"]: r["problem"] for r in map(json.loads, open(f"{ROOT}/bench/train36_input.jsonl"))}
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, device_map="auto").eval()
dev = model.device
x = tok.apply_chat_template([{"role": "system", "content": SOLVE}, {"role": "user", "content": probs[pid]}],
                            tokenize=True, add_generation_prompt=True, reasoning_effort="medium")
x = list(x["input_ids"] if hasattr(x, "keys") else x)
P = len(x)
cos = lambda a, b: torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1)  # noqa: E731


def full_ref(stream):
    """(a) states at the last 16 generated tokens, (16, hidden) fp32."""
    with torch.inference_mode():
        hs = model(input_ids=torch.tensor([x + stream], device=dev), output_hidden_states=True).hidden_states[LAYER]
    return hs[0, P + T - 16:P + T].float().cpu()


def fast_forced(stream, moe_default=False):
    """(b) rollout_fast's decoder (decode MoE flags), fed stream[j] at position P+j."""
    moe = F._MoeFlags()
    if not moe_default:
        moe.apply(B)
    saved = modeling_gpt_oss.eager_attention_forward
    modeling_gpt_oss.eager_attention_forward = F.gqa_attention
    got = []
    with torch.inference_mode():
        dec = F._Decoder(model, B, P + T + 1)
        q = torch.arange(P, device=dev)
        k = q[None, :]
        full = torch.where(k <= q[:, None], 0.0, F.NEG).to(torch.bfloat16)[None, None]
        slide = torch.where((k <= q[:, None]) & (q[:, None] - k < 128), 0.0, F.NEG).to(torch.bfloat16)[None, None]
        model(input_ids=torch.tensor([x], device=dev), past_key_values=F._PrefillCache(dec.kc, dec.vc), use_cache=True,
              position_ids=q[None], attention_mask={"full_attention": full, "sliding_attention": slide},
              logits_to_keep=1)
        modeling_gpt_oss.eager_attention_forward = saved
        dec.feat_layer = LAYER
        dec.pos.fill_(P)
        for j, t in enumerate(stream):
            dec.on_feat = (lambda h: got.append(h[0].float().cpu())) if j >= T - 16 else None
            dec(torch.tensor([[t]] * B, device=dev))
            dec.pos.add_(1)
    moe.restore()
    return torch.stack(got)


def hf_forced(stream):
    """(c) stock HF eager decode, DynamicCache, B identical rows; the layer L-1 output via a hook."""
    got, cur = [], {"keep": False}
    hook = model.model.layers[LAYER - 1].register_forward_hook(
        lambda m, a, o: got.append((o[0] if isinstance(o, tuple) else o)[0, -1].float().cpu()) if cur["keep"] else None)
    cache = DynamicCache(config=model.config)
    with torch.inference_mode():
        model(input_ids=torch.tensor([x] * B, device=dev), past_key_values=cache, use_cache=True, logits_to_keep=1)
        for j, t in enumerate(stream):
            cur["keep"] = j >= T - 16
            model(input_ids=torch.tensor([[t]] * B, device=dev), past_key_values=cache, use_cache=True,
                  logits_to_keep=1)
    hook.remove()
    return torch.stack(got)


r = F.rollouts(tok, model, x, batch=B, max_new=T, feat_layer=LAYER, return_toks=True)
print(f"{pid} P={P} v2 rollout: len {r['len']} check {r.get('check')} fallback {r.get('fallback')}", flush=True)
cands = [b for b in range(B) if r["len"][b] < 0]
refs = {b: full_ref(r["toks"][b].tolist()) for b in cands}
c_roll = {b: cos(r["feat"][b], refs[b].mean(0)).item() for b in cands}
order = sorted(cands, key=c_roll.get)
for b in order[:NW] + order[-1:]:
    stream = r["toks"][b].tolist()
    a = refs[b]
    fb = fast_forced(stream)
    fc = fast_forced(stream, moe_default=True) if os.environ.get("FT_MOE_DEFAULT") else hf_forced(stream)
    if os.environ.get("FT_MOE_DEFAULT"):
        print("   (c) below = v2 decoder with the default MoE flags, not HF eager", flush=True)
    tb, tc = cos(fb, a), cos(fc, a)
    print(f"sample {b}: mean-of-16 cos vs (a): rollout feat {c_roll[b]:.5f} | (b) v2 forced {cos(fb.mean(0), a.mean(0)).item():.5f}"
          f" | (c) HF eager decode {cos(fc.mean(0), a.mean(0)).item():.5f} | (b) vs (c) {cos(fb.mean(0), fc.mean(0)).item():.5f}"
          f" | rollout feat vs (b) {cos(r['feat'][b], fb.mean(0)).item():.5f}", flush=True)
    print(f"   per-token cos vs (a): (b) min {tb.min().item():.4f} mean {tb.mean().item():.5f} | "
          f"(c) min {tc.min().item():.4f} mean {tc.mean().item():.5f} | token norms {a.norm(dim=-1).min().item():.0f}-"
          f"{a.norm(dim=-1).max().item():.0f}", flush=True)
    print("   (b) " + " ".join(f"{v:.3f}" for v in tb.tolist()), flush=True)
    print("   (c) " + " ".join(f"{v:.3f}" for v in tc.tolist()), flush=True)
