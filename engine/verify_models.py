"""Verify downloaded model files against ModelScope's published sha256 (direct, no proxy)."""
import hashlib, json, os, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor

B = "/user/wangzhiyao/models/aimo"
REPOS = {  # local dir -> modelscope repo
    "Qwen/Qwen3.5-4B": "Qwen/Qwen3.5-4B",
    "Skywork/Skywork-OR1-Math-7B": "Skywork/Skywork-OR1-Math-7B",
    "allenai/Olmo-3-7B-Think": "allenai/Olmo-3-7B-Think",
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    "openai/gpt-oss-120b": "openai-mirror/gpt-oss-120b",
    "z-lab/gpt-oss-120b-DFlash": "z-lab/gpt-oss-120b-DFlash",
    "z-lab/Qwen3.5-4B-DFlash": "z-lab/Qwen3.5-4B-DFlash",
}
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


jobs = []
for local, repo in REPOS.items():
    url = f"https://www.modelscope.cn/api/v1/models/{repo}/repo/files?Recursive=true"
    files = json.load(opener.open(url, timeout=30))["Data"]["Files"]
    for x in files:
        if x["Type"] != "blob" or x["Path"].startswith(("original/", "metal/")):
            continue
        jobs.append((local, x["Path"], x.get("Size"), x.get("Sha256")))


def check(job):
    local, rel, size, want = job
    p = os.path.join(B, local, rel)
    if not os.path.isfile(p):
        return f"MISSING {local}/{rel}"
    if size is not None and os.path.getsize(p) != size:
        return f"SIZE    {local}/{rel} {os.path.getsize(p)} != {size}"
    if want and sha256(p) != want:
        return f"SHA     {local}/{rel}"
    return None


with ThreadPoolExecutor(16) as ex:
    bad = [r for r in ex.map(check, jobs) if r]
print(f"checked {len(jobs)} files, {len(bad)} bad")
print("\n".join(bad))
sys.exit(1 if bad else 0)
