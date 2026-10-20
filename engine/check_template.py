"""Compare SGLang's rendered prompt ids with HF apply_chat_template (run with .venv-eval python)."""
import json, sys, urllib.request
from transformers import AutoTokenizer
port, path = sys.argv[1], sys.argv[2]
effort = sys.argv[3] if len(sys.argv) > 3 else None
sysp = open("/user/wangzhiyao/projects/interp/ext/baselines/prompts/solve.txt").read()
msgs = [{"role": "system", "content": sysp}, {"role": "user", "content": "Find the remainder when $2^{100}$ is divided by 7."}]
body = {"messages": msgs, "max_tokens": 1, "temperature": 1.0, "top_k": 40, "top_p": 0.95, "return_prompt_token_ids": True}
models = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models"))
body["model"] = models["data"][0]["id"]
if effort:
    body["reasoning_effort"] = effort
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
d = json.load(urllib.request.urlopen(req))
sg = d["choices"][0].get("prompt_token_ids") or d.get("prompt_token_ids") or d["usage"]
tok = AutoTokenizer.from_pretrained(path)
kw = {"reasoning_effort": effort} if effort else {}
hf = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, **kw)
hf = hf["input_ids"] if isinstance(hf, dict) or hasattr(hf, "keys") else hf
print(body["model"], "match" if list(sg) == list(hf) else "MISMATCH", len(hf), "tokens")
if list(sg) != list(hf):
    print("sglang:", repr(tok.decode(sg))); print("hf    :", repr(tok.decode(hf)))
