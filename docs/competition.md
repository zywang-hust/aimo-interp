# 比赛规则与数据（AIMO Interpretability Challenge，Codabench 16180）

## 任务与接口
- 提交一个 ZIP，根目录放 `solution.py`，暴露 `are_robust(model_id, reasoning_effort, problems) -> list[bool]`。
  接口允许一次传整批题，但 Codabench 实测每道题单独调一次，每次是新进程；/tmp 不跨 case 保留，`$HOME` 保留（NOTES §6）。
- 根目录有 `small.txt` 就是 Small 轨，没有就是 Main 轨。评分看 accuracy；coverage 必须为 1，invalid_predictions 必须为 0。
- 评测机：1 张 RTX PRO 6000（96 GB，sm_120，与其他提交共享），不联网，HF cache 只读，挂在 `/app/data/huggingface`。
  时限 3600 s（Codabench 上设的是 4200 s）。
- 运行时：torch 2.12.1 / CUDA 13.2、transformers 5.13、kernels 0.15.2、sklearn 1.8、pandas、joblib、accelerate。
  没有 vLLM / sglang。实际镜像见 [eval_image.md](eval_image.md)；公开的 Dockerfile 已过时。
- 评测模型：Qwen/Qwen3.5-4B、Skywork-OR1-Math-7B、Olmo-3-7B-Think、DeepSeek-R1-0528-Qwen3-8B；
  Main 轨再加 gpt-oss-120b（low / medium / high）。
- 官方打标签时的采样参数：T=1.0、top_k=40、top_p=0.95，上限 100k token；
  提示词是 `ext/baselines/prompts/solve.txt`（作 system 消息），题目作 user 消息。

## 标签
- `max_drop` 是各扰动族中准确率下降最多的那一族的降幅。`≤ 0.10` 为 Robust，`≥ 0.25` 为 Spurious，中间不标注，
  val/test 里也不出现。官方说验证集“正负之间有足够大的间隔”“接近平衡”。
- 扰动族：domain / rename / rephrase / typos / distract（LLM 生成），以及人写的 expert_perturbations
  （改问法或改数字，答案随之变）和 expert_no_solution（改成无解，答案 NaN）。

## 数据
| 数据集 | 规模 | 用途 |
|---|---|---|
| train-main-v2 | 94 行 / 36 题；80 行有标签（59 R / 21 S） | 唯一覆盖正式模型的数据，字段 problem / is_robust / max_drop |
| val-sample | 28 行，旧模型 | 只用来检查提交契约 |
| augmented-sample-math*、challenge-sample-full | 54–558 行，旧模型 | 带逐扰动明细，能看到官方扰动长什么样 |
| baselines/data/math-robust-*.csv | 675 / 141 行，qwen3-8b:low | |

本地副本：`data/hf/<name>/data.parquet`；官方仓库浅克隆在 `ext/`（只读）。

## 时间与奖项
- 截止 2026-11-01 11:55 UTC，每天最多提交 10 次。最终排名用全新的未公开题，题数没公布；
  官方论坛（10-17）说最终可以选多个提交去跑测试集。
- 报告和开源代码 11-15 前交，11-15～11-30 互评 3 份报告。
- Main $2.5k / 1k / 500，Small $1.5k / 700 / 300。每轨另有 White-Box 奖 +$1k（报告里要用消融证明白盒信号有显著作用）。
  Writeup 奖共 $4k，不看排名。平分时比 3 次运行的平均耗时。
- 试探验证集标签算泄漏，直接取消资格。

## 公开榜（10-15 重置）
Main 验证 21 例（14 道不同的题），Small 约 14 例，分数步长 1/21、1/14。Main 一大片 0.71，不算 1.0 的最高 0.79；
头部的 1.0 多半是在小验证集上过拟合。
