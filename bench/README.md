# bench：本地题库、采样、特征和评估

R/S 标签只来自官方的 80 行。我们复现不出官方的掉分（`../NOTES.md` §1），所以 bench 不造标签，只提供两样东西：
- 稠密监督：每道题在每个配置上的推理长度和 p_base，用来训练探针；
- 一个带评测采样噪声的本地评估，用来比较判定规则（`evalbench.py`）。

所有脚本用 `python3 -I bench/scripts/<name>.py` 跑，公共函数在 `common.py`（`ROOT`、`MODEL_KEYS`、`auc`、`load_samples`）。

## 数据

| 路径 | 内容 |
|---|---|
| `train_main_v2_rows.jsonl` | 官方 94 行，pid 是题面 sha1 的前 8 位 |
| `train36_{problems,input,answers,refmatch}.jsonl` | 官方 36 题：题面、采样输入、参考答案及来源、和竞赛原题的匹配 |
| `pool/` | bench 题库：`pool_input`（262 题）、`twins_verified`（225 道核验过的改数字孪生题）、合并答案表 `answers.jsonl` |
| `feats/{pool,twins,train36}/` | prefill 隐藏状态（每层末 token + 题面均值，float16） |
| `localeval/` | 官方 harness 的本地评测集，80 行（`input/`、`ref/`）；`small/` 是 43 行的小模型子集 |

## 采样和产物（`runs/`）

| 路径 | 内容 | 生成 |
|---|---|---|
| `train36all/` | 官方 36 题 × 7 个配置，每题 16 条（官方参数） | `train36all/run.sh` → `sample.py` |
| `step1/`、`step1_joined.csv` | 官方题的 p_base 采样，和官方行拼好的表 | `sample.py`、`join_step1.py` |
| `step2/`、`step2b/` | 官方题的自造扰动（`perturb_train36*.jsonl`）及采样；step2b 是全部变体上的 gpt-oss medium | `perturb.py`、`sample.py` |
| `train36x/` | 官方 36 题 gpt-oss medium 再采 48 条、上限 4096，和 train36all 合起来每题 64 条，给噪声模拟用 | `train36x/run.sh` |
| `recall/` | 记忆探针的输出（gpt-oss-120b） | `recall.py` |
| `pool/`、`twins/` | bench 题库和孪生题的采样，每题 8 条；`.rev.jsonl` 是从队尾反向采的副本 | `sample.py`、`run_reverse.sh` |
| `twin_drop.csv` | 孪生题相对原题的准确率下降 | `twin_drop.py` |
| `nosol/` | 无解变体及各模型的识别率 | `perturb.py`、`nosol/run_*.sh` |
| `probe/` | prefill 探针和纯文本基线在官方题上的预测，以及 bench CV 预测 | `train_probe.py`、`text_baseline.py` |
| `trace/` | gpt-oss medium 轨迹（bench 738 题 5904 条），teacher-forced 中途特征，长度探针结果 | `sample.py`、`engine/evalrt/run_trace_feats.sh`、`len_probe.py` 等 |
| `label_noise/` | max_drop 标签的有限样本模型 | `label_noise.py` |

## 脚本

| 脚本 | 作用 |
|---|---|
| `fetch_pool.py`、`match_refs.py`、`make_twins.py` | 建题池；官方题匹配竞赛原题；生成改数字孪生题（生成、多模型复解核验后只留 `twins_verified`） |
| `sample.py` | 异步、可续跑的采样器，官方参数和 solve 提示词。`--reverse --skip-done-in` 从队尾反向采 |
| `run_reverse.sh`、`meet.sh`、`after.sh` | 慢配置的第二副本；正反两路会合时停掉；等某个进程退出再接下一步 |
| `perturb.py` | 扰动生成：typos 用规则，其余由 LLM 生成、LLM 判等价；expert_modify 和 no_solution 由判官独立复解 |
| `twin_drop.py`、`label_noise.py` | 孪生题掉分；标签噪声模型 |
| `extract_feats.py`、`train_probe.py`、`text_baseline.py`、`export_artifacts.py` | prefill 探针这条线：抽特征，训练并迁移到官方题，纯文本对照，导出到提交包的 `artifacts/` |
| `rollout_reader.py`、`val_like.py` | 各配置的真实生成行为在全部官方行和难区间上的 AUC |
| `evalbench.py` | 判定规则的本地评估：从 train36all 的 16 条（或 train36all + train36x 合起来的 64 条，`medium64`）里重抽 n 条、截到 T，阈值留一题拟合，重复 R 次（NOTES §2.2） |
| `calibrate.py`、`val_median.py` | 给 `decision.json` 定阈值（官方行留一题，或公开验证集的中位数） |
| `len_probe.py`、`fit_len_probe.py` | 白盒剩余长度探针：bench 轨迹上训练、官方题上评估；写 `submission/fragility/probe.npz` |
| `fit_probe_rule.py` | 按包里的算法评估探针版和孪生版（`--twin`），`--fit` 给 `decision.json` 定阈值 |
| `term_perm.py` | 题目级加项（孪生、来源、题面长度）的置换检验（NOTES §2.1、§5） |
| `recall.py` | 记忆探针：不推理直接答的正确率（NOTES §5） |
| `lbcheck.py` | 已交规则在模拟验证集上的分数分布，和公开榜对照（NOTES §2.3） |
| `trace_probe.py` | 中途熵、logprob、隐藏状态在长度之外有没有 R/S 信息（结论是没有，NOTES §5） |
| `selfperturb.py` | 自扰动：gpt-oss 在自己改写的变体上比原题多想多少（和原题长度重复，NOTES §5） |

## 约定

- 只写 `/user/wangzhiyao/...` 绝对路径，不写 `/root`。访问内网采样服务绕过代理（`ProxyHandler({})`）。
- 读下载的数据用 `python3 -I`。正在跑的 .sh 不要原地改。
