"""Problem-fragility predictor.

Official robustness labels are a property of the problem (the same problem gets the same label from a 4B model
and from gpt-oss-120b), and mostly of how hard it is: an easy problem has no accuracy to lose. So every config is
judged by how long gpt-oss-120b has to think about the problem.

Decision (decision.json "rollout"): gpt-oss-120b medium samples the problem `batch` times (rollout.py); the score is
the mean log length, capped at max_new, plus c x the asked config's weakness, plus `twin` if the problem is a changed
copy of a public contest problem (is_twin). Spurious if score > thr. With feature "probe", samples still thinking at
max_new count at the white-box length probe's prediction (probe.npz: ridge on the layer-`feat_layer` residual stream
averaged over the last 16 tokens, predicting the final log length) instead of the cap.
Codabench calls are_robust once per problem in a fresh process, so each problem is judged on its own. The prefill
reader (artifacts/, `z_thr`) decides only if the rollout did not run or finish.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast

HERE = Path(__file__).resolve().parent
SOLVE_PROMPT = (
    "You are a precise math problem solver. Solve the given problem step by step, then output your final "
    "answer on the last line in the exact format:\nANSWER: <your answer>\nThis result should be an integer "
    "value or NaN if the given problem does not have a solution."
)
MODEL_IDS = {  # artifact name prefix -> official model id
    "qwen3.5-4b": "Qwen/Qwen3.5-4B",
    "skywork-or1-7b": "Skywork/Skywork-OR1-Math-7B",
    "olmo3-7b-think": "allenai/Olmo-3-7B-Think",
    "dsr1-qwen3-8b": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    "gpt-oss-120b": "openai/gpt-oss-120b",
}
DECISION = json.loads((HERE / "decision.json").read_text())
# Codabench runs each case in a fresh process under one ingestion.py (pid 1). Of the writable places only $HOME keeps
# files between cases (/tmp, /dev/shm and the read-only package dir do not). State: the first call's start time, and
# finished rollouts, so a problem asked for several configs is rolled out once.
STATE_DIR = Path(os.environ.get("FRAGILITY_STATE_DIR") or Path.home() / ".cache" / "fragility_state")
JOB_TIMEOUT = 4200  # a state file older than one Codabench job is not ours
CORPUS = [json.loads(l) for l in (HERE / "corpus.jsonl").open()]  # public contest problems (MathArena, AIMO refs)
TIMES: dict[str, float] = {}  # seconds per stage of this call (model loads, reader forwards, rollouts), for the log
T_IMPORT = time.time()  # module import done (torch, transformers already imported above)


def _match(problem: str) -> tuple[str, float]:
    """Closest public contest problem (source, difflib similarity) among the 5 best by shared words/numbers."""
    norm = re.sub(r"\s+", " ", problem).strip()
    toks = set(re.findall(r"[A-Za-z]+|\d+", norm))
    jac = [len(toks & set(re.findall(r"[A-Za-z]+|\d+", c["problem"]))) / max(1, len(toks)) for c in CORPUS]
    best = sorted(range(len(CORPUS)), key=lambda i: -jac[i])[:5]
    sim, i = max((difflib.SequenceMatcher(None, norm, CORPUS[i]["problem"]).ratio(), i) for i in best)
    return CORPUS[i]["source"], sim


def is_twin(problem: str, lo: float = 0.6, hi: float = 0.9995) -> bool:
    """A non-verbatim copy of a public contest problem (numbers or wording changed); Spurious 7 of the 9 such hard
    official problems (bench/scripts/fit_probe_rule.py --twin). Below `lo` the problem matches nothing."""
    return lo < _match(problem)[1] < hi


def describe(problem: str) -> str:
    """Input-derived scalars only (no text): size, digits, and the closest public contest problem's source."""
    norm = re.sub(r"\s+", " ", problem).strip()
    src, sim = _match(problem)
    return (f"chars={len(norm)} digits={sum(ch.isdigit() for ch in norm)} dollars={norm.count('$')} "
            f"match={src} sim={sim:.3f}")


def _model_path(model_id: str) -> str:
    # local testing: FRAGILITY_MODEL_ROOT=/path/to/<org>/<name> tree; on Codabench the HF id resolves offline
    root = os.environ.get("FRAGILITY_MODEL_ROOT")
    return os.path.join(root, model_id) if root else model_id


def _load(model_id: str):
    path = _model_path(model_id)
    local = not os.environ.get("FRAGILITY_MODEL_ROOT")
    # transformers 5.13 mis-loads this tokenizer as a Metaspace LlamaTokenizer (drops all spaces)
    tok_cls = PreTrainedTokenizerFast if "DeepSeek-R1-0528" in model_id else AutoTokenizer
    tok = tok_cls.from_pretrained(path, local_files_only=local)
    model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", local_files_only=local).eval()
    return tok, model


def _hidden(tok, model, problem: str, effort: str | None, layer: int) -> np.ndarray:
    msgs = [{"role": "system", "content": SOLVE_PROMPT}, {"role": "user", "content": problem}]
    kw = {"reasoning_effort": effort} if effort else {}
    x = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, **kw)
    x = list(x["input_ids"] if hasattr(x, "keys") else x)
    with torch.inference_mode():
        out = model(input_ids=torch.tensor([x], device=model.device), output_hidden_states=True, use_cache=False)
    return out.hidden_states[layer][0, -1].float().cpu().numpy()


def _state_read(name: str) -> str | None:
    f = STATE_DIR / name
    try:
        if f.exists() and time.time() - f.stat().st_mtime < JOB_TIMEOUT:
            return f.read_text()
    except OSError:
        pass
    return None


def _state_write(name: str, text: str) -> None:
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        (STATE_DIR / name).write_text(text)
    except OSError:
        pass


def _proc_start(pid) -> float:
    btime = next(float(l.split()[1]) for l in open("/proc/stat") if l.startswith("btime"))
    st = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
    return btime + int(st[19]) / os.sysconf("SC_CLK_TCK")


def _proc_cmd(pid) -> list[str]:
    return [os.path.basename(a) for a in open(f"/proc/{pid}/cmdline").read().split("\0")[:3] if a]


def _start_time() -> tuple[float, str]:
    """Start of the whole prediction run: the earlier of the first call's time (if the state dir survives between
    cases) and the ingestion parent's start (if this process is its child); else this call (the guard never fires)."""
    now = time.time()
    s = _state_read("start")
    if s is None:
        _state_write("start", str(now))
    cands = [(float(s), "state")] if s else []
    try:
        pp = os.getppid()
        if any("ingestion" in a for a in _proc_cmd(pp)):
            cands.append((_proc_start(pp), "parent"))
    except (OSError, StopIteration, ValueError):
        pass
    cands = [c for c in cands if now - c[0] < JOB_TIMEOUT]
    return min(cands) if cands else (now, "none")


def _chat_ids(tok, msgs: list[dict], effort: str) -> list[int]:
    x = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, reasoning_effort=effort)
    return list(x["input_ids"] if hasattr(x, "keys") else x)


def _rollout_name(problem: str, spec: dict) -> str:
    return f"rollout_{hashlib.sha256(json.dumps([problem, spec], sort_keys=True).encode()).hexdigest()[:16]}.json"


def _rollout_feats(tok, model, problem: str, spec: dict) -> dict:
    """gpt-oss rollouts of the problem (spec: effort, batch, max_new, wall; engine "v1" (default, rollout.py) or "v2"
    (rollout_fast.py, own decode step, falls back to v1 by itself)); cached per problem in the state dir."""
    if spec.get("engine", "v1") == "v2":
        from rollout_fast import rollouts
    else:
        from rollout import rollouts
    name = _rollout_name(problem, spec)
    cached = _state_read(name)
    if cached:
        return {**json.loads(cached), "cached": True}
    start, src = _start_time()
    elapsed = time.time() - start
    if elapsed > spec["guard_s"]:
        return {"skipped": round(elapsed), "clock": src}
    ids = _chat_ids(tok, [{"role": "system", "content": SOLVE_PROMPT}, {"role": "user", "content": problem}],
                    spec["effort"])
    r = rollouts(tok, model, ids, batch=spec["batch"], max_new=spec["max_new"], wall=spec["wall"],
                 seed=spec.get("seed", 0), feat_layer=spec.get("feat_layer"))
    feat = r.pop("feat", None)
    if spec["feature"] == "probe" and feat is not None:
        m = np.load(HERE / "probe.npz")
        if (int(m["T"]), int(m["layer"])) != (spec["max_new"], spec["feat_layer"]):
            raise ValueError("probe.npz was fitted at another (T, layer) than decision.json asks for")
        r["probe"] = [round(float(v), 4) for v in
                      (((feat.numpy() - m["mu"]) / m["sd"] - m["pmean"]) @ m["comps"].T) @ m["coef"] + m["b"]]
    _state_write(name, json.dumps(r))
    return r


def rollout_score(r: dict, max_new: int, feature: str) -> float | None:
    """`loglen`: mean log length with unfinished samples counted at `max_new`; `probe`: the same with unfinished
    samples at the probe's predicted log length (not below log max_new). None if the run stopped early (wall clock)
    with samples unfinished."""
    if "len" not in r or (r["steps"] < max_new and not all(v > 0 for v in r["len"])):
        return None
    if feature == "probe":
        if not all(v > 0 for v in r["len"]) and len(r.get("probe") or []) != len(r["len"]):
            return None
        pr = r.get("probe") or [0.0] * len(r["len"])
        return float(np.mean([np.log(v) if v > 0 else max(p, np.log(max_new)) for v, p in zip(r["len"], pr)]))
    return float(np.mean([np.log(v if v > 0 else max_new) for v in r["len"]]))


def _rollouts(problems: list[str], spec: dict) -> list[dict]:
    """gpt-oss rollouts of every problem; the model is only loaded if some problem is not cached yet and the time
    guard still allows rollouts."""
    start = _start_time()[0]
    if all(_state_read(_rollout_name(p, spec)) for p in problems) or time.time() - start > spec["guard_s"]:
        return [_rollout_feats(None, None, p, spec) for p in problems]
    t = time.monotonic()
    tok, model = _load(MODEL_IDS["gpt-oss-120b"])
    TIMES["load:gpt-oss-120b"] = round(time.monotonic() - t, 1)
    t = time.monotonic()
    rolls = [_rollout_feats(tok, model, p, spec) for p in problems]
    TIMES["rollout"] = round(time.monotonic() - t, 1)
    del model
    torch.cuda.empty_cache()
    return rolls


def _reader_z(cfgs: list[str], problems: list[str]) -> dict[str, np.ndarray]:
    """Standardised predicted log reasoning length of each problem under each reader config `<name>__<effort>`.
    Configs of the same model share one load."""
    out = {}
    for name in dict.fromkeys(c.split("__")[0] for c in cfgs):
        t = time.monotonic()
        tok, model = _load(MODEL_IDS[name])
        TIMES[f"load:{name}"] = round(time.monotonic() - t, 1)
        for cfg in (c for c in cfgs if c.startswith(name + "__")):
            t = time.monotonic()
            art = np.load(HERE / "artifacts" / f"{cfg}.npz")
            eff = cfg.split("__")[1] if name == "gpt-oss-120b" else None
            s = np.array([((_hidden(tok, model, p, eff, int(art["layer"])) - art["mu"]) / art["sd"]) @ art["w"]
                          + art["ym"] for p in problems])
            out[cfg] = (s - art["ref"].mean()) / art["ref"].std()
            TIMES[f"read:{cfg}"] = round(time.monotonic() - t, 1)
        del model
        torch.cuda.empty_cache()
    return out


def predict(model_id: str, reasoning_effort: str | None, problems: list[str]) -> list[bool]:
    t0 = time.monotonic()
    try:  # process start -> this module imported (python, torch, transformers, corpus)
        TIMES["import"] = round(T_IMPORT - _proc_start(os.getpid()), 1)
    except (OSError, StopIteration, ValueError):
        pass
    name = {v: k for k, v in MODEL_IDS.items()}[model_id]
    cfg = f"{name}__{(reasoning_effort or 'medium').lower() if name == 'gpt-oss-120b' else 'default'}"
    w = DECISION.get("weakness", {}).get(cfg, 0.0)
    spec = DECISION.get("rollout")
    rolls = _rollouts(problems, spec) if spec else [{} for _ in problems]
    # the rollout score decides; the prefill reader is loaded only for problems without one (time guard, wall clock)
    f = [rollout_score(r, spec["max_new"], spec["feature"]) if spec and "thr" in spec else None for r in rolls]
    f = [v + spec["twin"] * is_twin(p) if v is not None and spec.get("twin") else v for v, p in zip(f, problems)]
    robust, zs = [bool(v + spec["c"] * w <= spec["thr"]) if v is not None else None for v in f], ""
    if None in robust:
        z_all = _reader_z(DECISION["readers"], problems)
        # harder relative to the asked model -> more fragile: add the model's standardised weakness on the bench
        z = np.mean([z_all[c] for c in DECISION["readers"]], axis=0) + DECISION.get("beta", 0.0) * w
        robust = [bool(z[i] <= DECISION["z_thr"]) if v is None else v for i, v in enumerate(robust)]
        zs = " ".join(f"{c}={np.round(v, 3).tolist()}" for c, v in z_all.items()) + " | "
    zs += f"rule={['rollout' if v is not None else 'reader' for v in f]}"
    if spec:
        T = spec["max_new"]
        zs += " | rollout " + " ; ".join(
            f"loglen={rollout_score(r, T, 'loglen')} probe={rollout_score(r, T, 'probe')} "
            + json.dumps({k: v for k, v in r.items() if k != "answers"})
            + f" answers={sorted(map(str, r.get('answers') or []))}"
            for r in rolls)
    zs += f" | times={json.dumps(TIMES)}"
    print(f"[fragility] {model_id}/{reasoning_effort}: {len(problems)} problems, {robust.count(False)} spurious, "
          f"{time.monotonic() - t0:.1f}s | {zs} | {' ; '.join(describe(p) for p in problems)}",
          file=sys.stderr, flush=True)
    return robust
