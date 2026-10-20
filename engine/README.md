# engine/: sampling servers and eval-runtime replica

Two venvs (both uv, Python 3.12.14):

| venv | purpose | key versions |
|---|---|---|
| `.venv` | SGLang sampling servers | sglang 0.5.22 (PyPI wheel), torch 2.14.1+cu130, flashinfer 0.7.1, transformers 5.19.0 |
| `.venv-eval` | replica of the official evaluation image | torch 2.12.1+cu132, transformers 5.13.0, accelerate 1.13.0, huggingface-hub 1.22.0, tokenizers 0.22.2, safetensors 0.8.0, scikit-learn 1.8.0, pandas 3.0.3, joblib 1.5.3 (triton 3.7.1 comes with torch) |

Driver 580.126 (CUDA 13.0) runs both the cu130 and the cu132 wheels.

## Starting and stopping servers

```bash
. /user/wangzhiyao/projects/interp/engine/env.sh        # optional for clients; serve.sh sources it
engine/serve.sh <model-key> <gpu-ids> <port> [extra sglang args]
engine/serve.sh dsr1-qwen3-8b 3 30004                   # 1 GPU
engine/serve.sh skywork-or1-7b 0,1 30002                # small models: N GPUs -> --dp-size N (one replica per GPU)
engine/serve.sh gpt-oss-120b 0 30005                    # 1 GPU is enough (61 GiB weights, 1.44M-token KV); 0,1 -> TP2
engine/stop.sh 30004 [more ports]   |   engine/stop.sh all
```

- Model keys: `qwen3.5-4b`, `skywork-or1-7b`, `olmo3-7b-think`, `dsr1-qwen3-8b`, `gpt-oss-120b`
  (and `qwen3.5-4b-dflash` and `gpt-oss-120b-dflash`, which are not recommended, see below). All per-model
  settings and notes are in `models.yaml`. The served model name is the key.
- serve.sh runs `tools/train/stop.sh --yield` on the current host (the card occupier holds about 110 GiB
  per card), starts the server in its own process group, and waits until `/v1/models` answers.
  `WAIT=0` returns immediately. `MEM_FRAC` sets the static memory fraction (default 0.88).
- **Host-scoped:** pid files go in `run/$(hostname)/<port>.{pid,key}` and logs in `logs/$(hostname)/<key>-<port>.log`.
  stop.sh only touches the current host's pid files and kills by PID/process group, never by pattern.
- JIT caches go in `/local/sglang-cache/interp-engine` (yrfs is unsafe for concurrent compiles).
  Cold start: small models take 1.5–4 min, gpt-oss about 4 min.
- Endpoint: `http://<eth0 ip>:<port>/v1` (OpenAI-compatible; serve.sh prints it). `engine/client.py`
  is a minimal async client (`smoke` and `bench` modes).

## Request format (official sampling)

```python
client.chat.completions.create(model="dsr1-qwen3-8b",
    messages=[{"role": "system", "content": open(".../ext/baselines/prompts/solve.txt").read()},
              {"role": "user", "content": problem}],
    temperature=1.0, top_p=0.95, max_tokens=100000,
    extra_body={"top_k": 40},                          # top_k as an extra body field
    reasoning_effort="high")                           # gpt-oss only: low | medium (default) | high
```

- Every server runs with `--sampling-defaults openai`, so no `generation_config.json` default is ever
  applied (Olmo ships T=0.6 and max_new_tokens=32768). Verified: no such log line, and top_k=1 gives
  identical samples while top_k=40 gives distinct ones.
- gpt-oss effort: the top-level `reasoning_effort` and `chat_template_kwargs={"reasoning_effort": ...}` both
  render "Reasoning: <effort>" into the harmony system block (checked on the prompt token ids).
- A reasoning parser is on for every model, so `message.reasoning_content` holds the thinking and
  `message.content` holds the answer. `usage.completion_tokens` counts both. On truncation
  (`finish_reason == "length"`) everything is in `reasoning_content`.
- For all 5 models, the prompt token ids SGLang renders equal HF 5.13 `apply_chat_template` output
  (system + user, `add_generation_prompt=True`), except that HF's own DSR1 tokenizer is broken (below).
  `check_template.py <port> <model-path> [effort]` re-checks this.

## Measured throughput (one B200 per server, 2026-10-18)

Bench: `client.py bench -n 256 --max-tokens 16384`: 256 requests in flight from the start, cycling over
4 AIME problems with a `\boxed{}` user prompt, official sampling. "wall avg" is total output tokens divided by
wall time, including the tail where few requests remain. "steady" is the median of the server's
reported decode throughput while the batch was at least 75% of its maximum running size.

| model | wall avg tok/s/GPU | steady tok/s | mean len (tok) | truncated at 16k | correct | KV pool (tok) |
|---|---|---|---|---|---|---|
| qwen3.5-4b | 10.8k | ~13k | 13.7k | 127/256 | 129 | 2.55M (+ mamba states) |
| skywork-or1-7b | 12.6k | ~29k | 7.4k | 26/256 | 225 | 2.66M |
| olmo3-7b-think | 2.6k | ~5.8k | 9.2k | 46/256 | 209 | 0.29M (full MHA) |
| dsr1-qwen3-8b | 4.8k | ~12.7k | 14.0k | 117/256 | 139 | 1.03M |
| gpt-oss-120b low | 8.8k | | 1.6k | 0 | 225 | 1.44M full + 1.16M SWA |
| gpt-oss-120b medium | 7.2k | ~16k (≥200 running) | 2.8k | 0 | 238 | |
| gpt-oss-120b high | 8.1k | | 6.1k | 11 | 241 | |

- Olmo and DSR1 are KV-bound: Olmo has 32 KV heads, about 0.5 MB/token. At 256 in flight the scheduler
  retracts and recomputes (195 and 158 retractions), but capping concurrency on the client was not better
  (Olmo c=32: 2.26k tok/s, DSR1 c=64: 4.20k tok/s).
- **At max_tokens 100k, KV capacity is the limit:** DSR1 fits about 10 full-length sequences per GPU, Olmo about 4
  (Olmo's context is 64k anyway). Plan sampling with many GPUs (DP) rather than deep queues.
- gpt-oss responses are short, so the wall average is dominated by the tail. Steady state is about 16k tok/s.

### DFlash drafters: not used

Both drafters are exact under sampling: SGLang's DFLASH verify accepts a draft token with the target
probability and resamples from the residual, with top-k/top-p applied to the target. They speed up a
single stream, but they cut throughput at high concurrency:

| | single stream | 256 requests (wall avg) |
|---|---|---|
| qwen3.5-4b | ~540 tok/s with DFlash | 3.47k (48 running, spec default), 3.58k (67 running) vs 10.8k without |
| gpt-oss-120b medium | | 6.7k vs 7.2k without |
| gpt-oss-120b high | ~600 vs ~390 tok/s without | 4.7k vs 8.1k without |

Accept length: Qwen about 5.8 of 16, gpt-oss about 3.9 of 10. With DFlash, Qwen3.5 is also capped at 67
running requests by the mamba state cache. The keys stay in `models.yaml` for low-concurrency use only.

## Eval-runtime replica (`.venv-eval`), one B200

`evalrt/run_small.sh <org/name>...` (env `GPU=<id>`) runs `eval_check.py`, which loads with
`AutoModelForCausalLM.from_pretrained(path, device_map="auto")` offline and times a forward pass over 32
prompts of exactly 300 tokens (bf16 / native dtype, `inference_mode`). Logs are in `evalrt/logs/eval_check_*.log`.

| model | loads as | load s | weights GiB | first fwd s | warm fwd s (32×300) | peak GiB |
|---|---|---|---|---|---|---|
| Qwen3.5-4B | Qwen3_5ForCausalLM bf16 | 2.8 | 7.8 | 0.96 | 0.367 | 19.0 |
| Skywork-OR1-Math-7B | Qwen2ForCausalLM bf16 | 3.3 | 14.2 | 1.57 | 0.145 | 20.8 |
| Olmo-3-7B-Think | Olmo3ForCausalLM bf16 | 3.2 | 13.6 | 0.65 | 0.231 | 17.3 |
| DeepSeek-R1-0528-Qwen3-8B | Qwen3ForCausalLM bf16 | 3.3 | 15.3 | 0.74 | 0.192 | 23.4 |
| gpt-oss-120b (with `kernels`, see below) | GptOssForCausalLM, MXFP4 experts (Mxfp4GptOssExperts) | 11.8 | 60.8 | 8.2 | 0.506 | 69.4 |

Caveats for writing `solution.py`:

1. **The DSR1 tokenizer is broken in transformers 5.13** (and 5.19). `tokenizer_class` is
   `LlamaTokenizerFast`, which v5 maps to `LlamaTokenizer` with a Metaspace pre-tokenizer: every space and newline
   is dropped ("You are a precise" -> `You|are|ap|rec|ise…`). Use `PreTrainedTokenizerFast.from_pretrained(path)`
   or `AutoTokenizer.from_pretrained(path, tokenizer_type="qwen2")`. Both are verified identical to
   `tokenizer.json` and to SGLang. The other 4 models are fine with AutoTokenizer. The organizers know:
   the `uncertainty-profiling` baseline loads with `PreTrainedTokenizerFast` and has a comment about it.
   The `trained-probe` baseline still uses plain AutoTokenizer, so its DSR1 features come from mangled tokens.
2. **gpt-oss MXFP4 needs the `kernels` package** (transformers 5.13 accepts 0.15.2 ≤ v < 0.16) and a complete cached
   snapshot of `kernels-community/gpt-oss-triton-kernels` with `refs/v1`. Without `kernels`, transformers dequantizes to
   bf16: about 230 GB with CPU offload, and the second forward OOMs even on a 178 GiB B200. The Codabench image has both
   (`../docs/eval_image.md`); the public Dockerfile has neither. Locally the overlay `EVAL_EXTRA_SITE=evalrt/kernels-overlay`
   adds `kernels==0.15.2`, `kernels-data 0.16.2`, `tomlkit` and `pyyaml` (no-deps, so hub stays 1.22) without touching
   `.venv-eval`; the snapshot (commit c039a37b…, 734 KB) is in `/user/wangzhiyao/cache/huggingface/hub`. A partial
   cuda-only snapshot fails offline.
3. Qwen3.5's GatedDeltaNet layers fall back to the torch implementation (no flash-linear-attention /
   causal-conv1d in the image). That is fine for forward passes but slow for long HF `generate`.
4. Olmo-3-7B-Think: `max_position_embeddings` is 65536, so 100k tokens is impossible. The server uses context 65536.

The Codabench machine matches this replica plus `kernels`: gpt-oss-120b loads natively in MXFP4 there (60.8 GiB, 4.5 s)
and runs on sm_120 (`../NOTES.md` §6).

## Eval-time generation (`evalrt/`)

The package samples gpt-oss-120b on the eval box, so decode speed under HF is what matters. Design notes and
numbers are in `../NOTES.md` §4. Every script runs in `.venv-eval` with the kernels overlay and imports the engines
straight from `submission/fragility/`: `GPU=<id> engine/evalrt/run_eval.sh <script.py> [args]`.

| file | what it checks |
|---|---|
| `rollout_check.py` | sampled length distribution vs the sglang samples of `bench/runs/train36all` (`ROLLOUT_MOD=rollout_fast` for v2, `FEAT_LAYER=18`) |
| `logit_check.py`, `greedy_gate.py`, `kernel_check.py` | v2 against HF eager: teacher-forced logits, greedy departures, bitwise kernel agreement, each measured against HF's own batch-size noise floor |
| `feat_check.py` | the in-graph probe feature vs teacher forcing (`ROLLOUT_MOD=rollout_fast` for v2), with the neighbouring layers and the probe output |
| `feat_tf_check.py` | one sampled stream teacher-forced through v2's decoder, HF eager decode and one HF forward: per-token layer-18 states |
| `batch_check.py` | per-row prompts in one batch vs each prompt alone (v1) |
| `pkg_check.py` | the whole package the way Codabench runs it: one fresh process per case (`<pid>:<model_id>:<effort>`), per-case wall time and decision; `--config <decision.json>` swaps the rules, `PKG_COLD_TRITON=1` gives every case an empty Triton cache, `ROLLOUT_TEST_FALLBACK=1` forces v2's fallback to v1 |
| `trace_feats.py`, `run_trace_feats.sh` | teacher-forced mid-trace features (hidden states, entropy, logprob) of sglang traces, for `bench/scripts/len_probe.py` and `trace_probe.py` |
| `logs/` | outputs of the checks above (`pkg_<config>.log` per final config) and of `eval_check.py` |

## Files

`env.sh` · `models.yaml` · `serve.sh` · `stop.sh` · `client.py` · `check_template.py` · `eval_check.py` ·
`download_models.sh` / `verify_models.py` (models in `/user/wangzhiyao/models/aimo`, sha256-checked) ·
`evalrt/` (above, plus `kernels-overlay/` and `run_small.sh`) · `logs/<host>/` · `run/<host>/`
