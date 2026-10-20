"""Package submission/fragility/ into a Main-track Codabench ZIP (files at the ZIP root, no caches):
  python3 -I submission/build.py [submission/configs/<name>.json]
With a config, it goes into the ZIP as decision.json and the ZIP is dist/<name>.main.zip; without, the package's own
decision.json is used and the ZIP is dist/fragility.main.zip.
"""
import os
import sys
import zipfile

S = "/user/wangzhiyao/projects/interp/submission"
src = f"{S}/fragility"
config = sys.argv[1] if len(sys.argv) > 1 else None
name = os.path.splitext(os.path.basename(config))[0] if config else "fragility"
files = sorted(os.path.relpath(os.path.join(d, f), src) for d, ds, fs in os.walk(src) for f in fs
               if "__pycache__" not in d)
os.makedirs(f"{S}/dist", exist_ok=True)
out = f"{S}/dist/{name}.main.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        z.write(config if config and f == "decision.json" else f"{src}/{f}", f)
print(out, os.path.getsize(out), "bytes:", zipfile.ZipFile(out).namelist())
