# AIMO Interpretability Challenge

给一道数学题和一个模型配置，预测它是 Robust 还是 Spurious，也就是题目被扰动后这个模型的准确率会不会大幅下降。
规则见 [docs/competition.md](docs/competition.md)，研究结论见 [NOTES.md](NOTES.md)，下一步见 [PLAN.md](PLAN.md)。

## 方法

标签几乎只取决于题目（同一道题跨配置一致率 97.5%），主要成分是难度：简单题没有分可掉。所以评测时让
gpt-oss-120b medium 对这道题真实生成 12 条，按平均 log 长度打分，加一点被问模型的弱度，和固定阈值比。两个版本互斥：

- 探针版：截到 1536，没写完的样本由第 18 层残差流上的长度探针估最终长度。准确率和跑到 3584 持平，时间省 60%，
  这也是白盒奖要的消融（NOTES §4.3）。
- 孪生版：截到 3072；题目如果是改过数字或措辞的竞赛原题，分数再加 0.4（NOTES §2.1）。

生成用自己写的解码：v1（`rollout.py`）把 HF forward 抓进 CUDA graph；v2（`rollout_fast.py`）自己写 decode step，
在评测机上比 v1 快约 25%，在 `decision.json` 里写 `"engine": "v2"` 才用（NOTES §4）。同一道题被问多个配置时，
rollout 结果缓存在评测机的 `$HOME` 里，只跑一次（NOTES §6）。

公开榜最好 0.762。同一条规则重交会在 0.62–0.76 之间跳，单次榜分的噪声约 ±0.2；bench 和公开榜对得上（NOTES §2.3）。
难题上长度信号的 AUC 约 0.71，和难度无关、又过了检验的只有孪生项（NOTES §0）。

## 目录

| 路径 | 内容 |
|---|---|
| `submission/fragility/` | 提交包：`solution.py` → `fragility.py`（判定）、`rollout.py`（v1）、`rollout_fast.py`（v2，编不过或第一步算错时退回 v1）、`decision.json`（规则和阈值）、`probe.npz`（长度探针）、`corpus.jsonl`（孪生题匹配用的竞赛原题）、`artifacts/`（prefill 读取，rollout 没跑时的后备） |
| `submission/configs/` | 规则配置：`probe1536.json`、`twin3072.json`（= `decision.json`），以及换 v2 引擎的 `*_v2.json` |
| `submission/build.py` | 打包到 `submission/dist/`；Codabench 日志下载到 `submission/logs/<id>/`（两者都不进 git） |
| `bench/` | 本地题库、采样、特征、探针和评估脚本（[bench/README.md](bench/README.md)） |
| `engine/` | SGLang 采样服务、评测镜像复刻、评测时生成的检查和整包测试（[engine/README.md](engine/README.md)） |
| `tools/codabench/cb.py` | Codabench 客户端：提交、查状态、下日志、看榜 |
| `docs/` | 比赛规则、评测镜像、标签理论与文献 |
| `data/`、`ext/` | 下载的数据集和官方仓库，只读，不在里面放脚本 |

## 常用命令

```bash
# 本地 bench：判定规则在官方行上的准确率（带评测采样噪声，阈值留一题）；已交规则和公开榜对照
python3 -I bench/scripts/evalbench.py
python3 -I bench/scripts/fit_probe_rule.py --twin     # 探针版、孪生版；--fit 给 decision.json 定阈值
python3 -I bench/scripts/lbcheck.py

# 按 Codabench 的方式跑整包（每个 case 一个新进程）；占卡进程会自己让卡
GPU=0 engine/evalrt/run_eval.sh engine/evalrt/pkg_check.py --config submission/configs/probe1536_v2.json \
    ca873e52:Qwen/Qwen3.5-4B:default 388d7602:openai/gpt-oss-120b:high

# 打包（可指定配置）、提交（每天 10 次）、看结果；cb.py 必须去掉代理
python3 -I submission/build.py submission/configs/probe1536_v2.json   # -> submission/dist/probe1536_v2.main.zip
CB="env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy python3 -I tools/codabench/cb.py"
$CB submit submission/dist/probe1536_v2.main.zip main
$CB list          # 或 show <id>、logs <id> submission/logs/<id>、leaderboard
```

## 约定

- 只写 `/user/wangzhiyao/...` 绝对路径，绝不写 `/root`。临时文件放 `/user/wangzhiyao/cache/tmp`。
- 停服务按 PID（`engine/stop.sh`），不用 `pkill -f`。让卡用 `/user/wangzhiyao/tools/train/stop.sh --yield`。
- Codabench 凭证只在 `/user/wangzhiyao/.wzy/secrets.env`。不做标签试探，不写死答案，不打印题面。
- 读下载的数据用 `python3 -I`。正在跑的 .sh 不要原地改。
