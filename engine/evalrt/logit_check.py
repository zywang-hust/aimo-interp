"""Teacher-forced decode logits: rollout_fast's decoder vs stock HF eager decode (DynamicCache), same token stream.

  run_eval.sh engine/evalrt/logit_check.py <model> <effort> <batch> <n_tokens> id [id ...]
The token stream is HF batch-1 greedy. Prints, per problem, max and mean |logit diff| over the steps (rows of a
batch are identical inputs, row 0 used) for: HF batch B vs HF batch 1 (the reference's own kernel noise), fast vs
HF batch 1, fast with the default MoE flags vs HF batch 1; and how often the argmax differs.
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

path, effort, B, T, ids = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5:]
SOLVE = open(f"{ROOT}/ext/baselines/prompts/solve.txt").read().strip()
probs = {r["id"]: r["problem"] for r in map(json.loads, open(f"{ROOT}/bench/train36_input.jsonl"))}
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, device_map="auto").eval()
dev = model.device


def hf_forced(x, stream, nb):
    cache = DynamicCache(config=model.config)
    out = []
    with torch.inference_mode():
        lg = model(input_ids=torch.tensor([x] * nb, device=dev), past_key_values=cache, use_cache=True).logits[:, -1]
        out.append(lg[0].float())
        for t in stream[:-1]:
            lg = model(input_ids=torch.tensor([[t]] * nb, device=dev), past_key_values=cache, use_cache=True).logits
            out.append(lg[0, -1].float())
    return torch.stack(out)


def fast_forced(x, stream, nb, moe_default):
    moe = F._MoeFlags()
    if not moe_default:
        moe.apply(nb)
    saved = modeling_gpt_oss.eager_attention_forward
    modeling_gpt_oss.eager_attention_forward = F.gqa_attention
    P = len(x)
    out = []
    with torch.inference_mode():
        dec = F._Decoder(model, nb, P + len(stream) + 1)
        q = torch.arange(P, device=dev)
        k = q[None, :]
        full = torch.where(k <= q[:, None], 0.0, F.NEG).to(torch.bfloat16)[None, None]
        slide = torch.where((k <= q[:, None]) & (q[:, None] - k < 128), 0.0, F.NEG).to(torch.bfloat16)[None, None]
        lg = model(input_ids=torch.tensor([x], device=dev), past_key_values=F._PrefillCache(dec.kc, dec.vc),
                   use_cache=True, position_ids=q[None], attention_mask={"full_attention": full,
                                                                         "sliding_attention": slide},
                   logits_to_keep=1).logits
        modeling_gpt_oss.eager_attention_forward = saved
        out.append(lg[0, -1].float())
        dec.pos.fill_(P)
        for t in stream[:-1]:
            out.append(dec(torch.tensor([[t]] * nb, device=dev))[0].float())
            dec.pos.add_(1)
    moe.restore()
    return torch.stack(out)


def cmp(name, a, ref):
    d = (a - ref).abs().max(-1).values
    flips = (a.argmax(-1) != ref.argmax(-1)).float().mean().item()
    print(f"   {name:34s} |dlogit| max {d.max().item():.3f} mean-of-max {d.mean().item():.3f}  "
          f"argmax differs at {flips:.3%} of steps", flush=True)


for pid in ids:
    x = tok.apply_chat_template([{"role": "system", "content": SOLVE}, {"role": "user", "content": probs[pid]}],
                                tokenize=True, add_generation_prompt=True, reasoning_effort=effort)
    x = list(x["input_ids"] if hasattr(x, "keys") else x)
    with torch.inference_mode():
        g = model.generate(torch.tensor([x], device=dev), do_sample=False, max_new_tokens=T,
                           attention_mask=torch.ones(1, len(x), dtype=torch.long, device=dev))
    stream = g[0, len(x):].tolist()
    print(f"{pid} P={len(x)} stream {len(stream)} tokens, batch {B}", flush=True)
    h1 = hf_forced(x, stream, 1)
    cmp(f"HF batch {B} vs HF batch 1", hf_forced(x, stream, B), h1)
    cmp(f"fast batch {B} vs HF batch 1", fast_forced(x, stream, B, False), h1)
    cmp(f"fast default-MoE batch {B} vs HF b1", fast_forced(x, stream, B, True), h1)
    if os.environ.get("LC_BATCH1", "1") == "1":
        cmp("fast batch 1 vs HF batch 1", fast_forced(x, stream, 1, False), h1)
        cmp("fast default-MoE batch 1 vs HF b1", fast_forced(x, stream, 1, True), h1)
    for fl in os.environ.get("LC_FLAGS", "").split():
        os.environ["ROLLOUT_MOE_FLAGS"] = fl
        cmp(f"fast batch {B} MoE {fl} vs HF b1", fast_forced(x, stream, B, False), h1)
        os.environ.pop("ROLLOUT_MOE_FLAGS")
