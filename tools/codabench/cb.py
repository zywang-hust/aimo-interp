"""Minimal Codabench client (direct connection, no proxy) for competition 16180.

Credentials come from /user/wangzhiyao/.wzy/secrets.env (CODABENCH_USER / CODABENCH_PASSWORD); never printed.
  python3 -I tools/codabench/cb.py whoami
  python3 -I tools/codabench/cb.py list
  python3 -I tools/codabench/cb.py submit <zip> main|small
  python3 -I tools/codabench/cb.py show <submission-id>        # status, scores, log URLs
  python3 -I tools/codabench/cb.py logs <submission-id> [dir]  # download all logs
Submissions: 10 per day.
"""
import http.cookiejar
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.error
import urllib.request

BASE = "https://www.codabench.org"
COMPETITION, PHASE = 16180, 30038
TASKS = {"main": 36237, "small": 36241}
SECRETS = "/user/wangzhiyao/.wzy/secrets.env"


def _creds():
    env = dict(l.strip().split("=", 1) for l in open(SECRETS) if "=" in l and not l.startswith("#"))
    return env["CODABENCH_USER"], env["CODABENCH_PASSWORD"]


class Client:
    def __init__(self):
        self.jar = http.cookiejar.CookieJar()
        self.op = urllib.request.build_opener(urllib.request.ProxyHandler({}),
                                              urllib.request.HTTPCookieProcessor(self.jar))
        self.op.addheaders = [("User-Agent", "Mozilla/5.0 aimo-interp-client")]

    def _csrf(self):
        return next((c.value for c in self.jar if c.name == "csrftoken"), "")

    def req(self, method, path, data=None, raw=None, headers=None, timeout=120):
        url = path if path.startswith("http") else BASE + path
        h = {"X-CSRFToken": self._csrf(), "Referer": BASE + "/"}
        body = None
        if data is not None:
            body = json.dumps(data).encode()
            h["Content-Type"] = "application/json"
        elif raw is not None:
            body = raw
        h.update(headers or {})
        r = urllib.request.Request(url, data=body, method=method, headers=h)
        try:
            with self.op.open(r, timeout=timeout) as resp:
                out = resp.read()
        except urllib.error.HTTPError as e:
            raise SystemExit(f"{method} {path}: HTTP {e.code}: {e.read()[:500]!r}")
        try:
            return json.loads(out)
        except ValueError:
            return out

    def login(self):
        page = self.op.open(BASE + "/accounts/login/", timeout=60).read().decode()
        tok = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', page).group(1)
        user, pw = _creds()
        form = urllib.parse.urlencode({"csrfmiddlewaretoken": tok, "username": user, "password": pw}).encode()
        r = urllib.request.Request(BASE + "/accounts/login/", data=form, method="POST",
                                   headers={"Referer": BASE + "/accounts/login/",
                                            "Content-Type": "application/x-www-form-urlencoded"})
        self.op.open(r, timeout=60).read()
        if not any(c.name == "sessionid" for c in self.jar):
            raise SystemExit("login failed (no session cookie)")
        return self


def main():
    cmd = sys.argv[1]
    c = Client().login()
    if cmd == "whoami":
        print(json.dumps(c.req("GET", "/api/my_profile/"), indent=1)[:600])
    elif cmd == "list":
        subs = c.req("GET", f"/api/submissions/?phase={PHASE}")
        for s in (subs if isinstance(subs, list) else subs.get("results", [])):
            print(s.get("id"), s.get("status"), s.get("created_when"), s.get("filename"),
                  [t.get("id") if isinstance(t, dict) else t for t in (s.get("tasks") or [])],
                  json.dumps(s.get("scores"))[:200])
    elif cmd == "submit":
        path, track = sys.argv[2], sys.argv[3]
        name = time.strftime("%m%d-%H%M%S-") + os.path.basename(path)  # dataset names must be unique per user
        ds = c.req("POST", "/api/datasets/", data={"type": "submission", "name": name, "description": "",
                                                   "is_public": False, "request_sassy_file_name": name,
                                                   "file_size": os.path.getsize(path)})
        blob = open(path, "rb").read()
        sassy = ds["sassy_url"]
        hdr = {"Content-Type": "application/zip"}
        if "blob.core.windows.net" in sassy:
            hdr["x-ms-blob-type"] = "BlockBlob"
        put = urllib.request.Request(sassy, data=blob, method="PUT", headers=hdr)
        urllib.request.build_opener(urllib.request.ProxyHandler({})).open(put, timeout=600).read()
        c.req("PUT", f"/api/datasets/completed/{ds['key']}/")
        sub = c.req("POST", "/api/submissions/", data={"data": ds["key"], "phase": PHASE,
                                                       "tasks": [TASKS[track]], "fact_sheet_answers": None})
        print(json.dumps(sub)[:800])
    elif cmd == "leaderboard":  # public leaderboard of the phase: rank, team, primary score
        lb = c.req("GET", f"/api/phases/{PHASE}/get_leaderboard/")
        print(json.dumps(lb))
    elif cmd == "show":
        sid = sys.argv[2]
        print(json.dumps(c.req("GET", f"/api/submissions/{sid}/"), indent=1)[:3000])
        print(json.dumps(c.req("GET", f"/api/submissions/{sid}/get_details/"), indent=1)[:3000])
    elif cmd == "logs":
        sid = sys.argv[2]
        out = sys.argv[3] if len(sys.argv) > 3 else f"/user/wangzhiyao/projects/interp/submission/logs/{sid}"
        os.makedirs(out, exist_ok=True)
        d = c.req("GET", f"/api/submissions/{sid}/get_details/")
        urls = []

        def walk(x, k=""):
            if isinstance(x, dict):
                for kk, v in x.items():
                    walk(v, f"{k}.{kk}" if k else kk)
            elif isinstance(x, list):
                for i, v in enumerate(x):
                    walk(v, f"{k}.{i}")
            elif isinstance(x, str) and x.startswith("http"):
                urls.append((k, x))
        walk(d)
        for k, u in urls:
            try:
                data = urllib.request.build_opener(urllib.request.ProxyHandler({})).open(u, timeout=120).read()
                fn = os.path.join(out, re.sub(r"[^A-Za-z0-9_.-]", "_", k))
                open(fn, "wb").write(data)
                print(fn, len(data))
            except Exception as e:
                print("fail", k, e)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
