# Session Handoff: Qwen Reward Prompt + Wandb Video Media

## 任务目的
缓解 Qwen-VL reward 的两个问题：
1. `enable_reason=True` 时第一段 reasoning 过长且会把第二段 Yes/No logit 锚到极端 0/1。
2. 引入 reason logging 后，I2V/V2V 主样本视频从 wandb Media 消失（`d8ibkzma` run 的 `train_samples`/`eval_samples` 表里 generation 单元格只剩 `"Video"` 占位字符串）。

## 执行内容
- `qwen_vl_video_reward.py`：`REASON_FINAL_TOKEN_MARGIN` 由 `256` 改为 `1024`，给 `</think>` 后的 final content 留更多 token 空间。
- `qwen_vl_video_reward.py`：`REASON_FIELD_PRIORITY` 由 `("reasoning", "reasoning_content", "content")` 改为 `("content", "reasoning", "reasoning_content")`，让第二段 Yes/No 和 wandb reason 表都看 final content 而不是内部 thinking。
- `qwen_vl_video_reward.py`：tex/shape 的 framework / yes_no_decision / reason_body 重写为同构版本：framework 明确 caption 是“语义描述、判断仍以视觉为准”；reason 改成 2-4 句证据观察、不下 pass/fail verdict；Yes/No 由 `successful generation` 改为 `reasonable visual/geometry match`。
- `logger/formatting.py`：把 `_concat_videos_grid` 加回来，`_process_i2v_samples` / `_process_v2v_samples` 恢复成 `List[LogVideo]`（composite condition+generation），不再走 `LogTable`。reason 仍由 `LogTableIncrement` 走独立 `*_reasons` 增量表，互不冲突。
- 验证：`LogFormatter.format_dict({"train_samples": [I2VSample]})` 现在返回 `list[LogVideo]`，且额外生成 `train_samples_reasons: LogTableIncrement`；`py_compile` / `ReadLints` 通过。

## 调试经验
- `d8ibkzma` run 表面是 “视频没了”，但实际不是 reason table 的事，而是 commit `e6678f7` 把 I2V/V2V 从 top-level `LogVideo` 列表改成了 `LogTable` 整体；wandb 把 `wandb.Video` 嵌进 `wandb.Table` 单元格后，`*.table.json` 落地只剩 `"Video"` 字符串占位，Media 自然就没视频。修复路径是恢复 List[LogVideo]，不要叠加 LogTable。
- `train_samples_reasons` 看起来“一直只有 8 行”是误读：本地 `media/table/train_samples_reasons_{0,3,7,11}_*.table.json` 行数是 8/16/24/32，是累计的；UI 上感觉每次都是 8 行，是因为 `AdvantageProcessor._build_*_log_data` 写入的 `train_samples=samples[:30]` 是 rank 0 本地 8 个 sample（`num_batches_per_epoch=4 * per_device_batch_size=2`），并未 gather 全 rank。如果想每轮 reason 表显示全局所有样本，需要在 logging 前 gather。
- Yes/No logit 极端塌成 0/1 的真正原因：第一段 thinking-conditioned 完成后，把模型自己的 final content 作为 assistant turn 塞回上下文再问 1-token Yes/No，content 里 `successfully captures...` / `fails to...` 这类 verdict 句子会强力锚定下一 token。本轮通过改 prompt（observation-only、reasonable match）缓解，没有动 reward 公式。
- 当前还没改的：`REASON_FINAL_TOKEN_MARGIN` 仍是硬编码常量；之前讨论过把它放进 YAML（和 `reason_thinking_token_budget` 同组），暂未做。

## 参考代码
| 文件 | 关键位置 | 说明 |
|------|---------|------|
| `src/flow_factory/rewards/qwen_vl_video_reward.py` | `REASON_FINAL_TOKEN_MARGIN`、`REASON_FIELD_PRIORITY`、`_TEX_*` / `_SHAPE_*` 常量 | reason token 预算、字段优先级、tex/shape 同构 prompt |
| `src/flow_factory/rewards/qwen_vl_video_reward.py` | `_query_reason_text`、`_query_yes_no_logprob`、`_score_single`、`_build_reason_conditioned_messages` | 第一段读 content，第二段把 reason 作为 assistant turn 后问 Yes/No |
| `src/flow_factory/rewards/vllm_evaluate.py` | `_get_yes_cond_prob` | 当前 reward 用 `sigmoid(logp_yes - logp_no)`；如果之后想做 `score * yes_prob`，从这里入手 |
| `src/flow_factory/logger/formatting.py` | `_concat_videos_grid`、`_process_i2v_samples`、`_process_v2v_samples` | 主样本恢复成 composite `LogVideo`，不再走 `LogTable` |
| `src/flow_factory/logger/formatting.py` | `LogTableIncrement.from_reason_samples`、`LogFormatter.format_dict` | reason 走 `*_reasons` 增量表，依赖 `sample.extra_kwargs["reward_reason"]` |
| `examples/grpo/lora/trellis2/{tex,shape}_qwen_vl_reward.yaml` | `extra_kwargs.reason_thinking_token_budget` | 当前 tex=2048，shape=1024，可以一起观测 |

## 最终方案（当前已采用）
- reward 形态保持 `reason -> reason-conditioned yes/no`，不引入 score * yes_prob 之类的复合公式。
- 通过两层联合压制 logit 极端化：
  1. 第一段 reason 读 final `content`（短、自然回答），不读 raw thinking；
  2. tex/shape prompt 同构化，reason 不下 verdict，Yes/No 用 “reasonable match” 这种连续阈值化措辞。
- I2V/V2V 主样本恢复 composite `LogVideo`（condition image + generation 拼一起），保证 wandb Media 仍有视频；reason 表独立走 `*_reasons` 增量表，不互相覆盖。

## 下一步任务
重新跑训练，观察：
- wandb Media 面板是否重新出现 `train_samples` / `eval_samples` 的视频。
- reward 的 Yes/No 概率分布是否不再塌成 0/1（可以画 reward 直方图或追踪 `reward_*_std`、`group_std_min/max`）。
- reason 表 `train_samples_reasons` 文本是否变成更短、更 observation 风格（不再有 frame-by-frame checklist 和 “Drafting the response”）。

## 初步方案
- 用 `examples/grpo/lora/trellis2/tex_qwen_vl_reward.yaml`（或 shape 那个）启动一个短 run，先确认 step 0 / step 4 / step 8 三轮即可。
- 观察项：
  - wandb Media → 找 `train_samples`、`eval_samples`，应能看到 condition+generation 拼接的视频；如果还没视频，看 `wandb/run-*/files/media/videos/` 目录是否落地 mp4。
  - wandb Charts → `train/reward_qwen_vl_side_by_side_mean/std/group_std_*`：std 不再恒等于 0.43 这类“接近 Bernoulli(0.7)”的值，且 `group_std_min` 不应一直贴 0。
  - wandb Tables → `train_samples_reasons` 文本应是 2-4 句 observation；`eval_samples_reasons` 同理。
- 风险点：
  - `REASON_FINAL_TOKEN_MARGIN=1024` 加 `reason_thinking_token_budget=2048`（tex）会让单次 reason 请求 max_tokens=3072；如果显存或吞吐紧张，先把 thinking_token_budget 降到 1024。
  - vLLM 必须用 `--reasoning-parser qwen3` 启动，否则 `content` 里会带 `<think>...</think>`，第二段 Yes/No 又会被原始 thinking 锚定。
  - 如果 Yes/No 仍然塌成 0/1，下一步候选：(a) 把 `REASON_FINAL_TOKEN_MARGIN` 配置化到 YAML 单独控制 final content 长度；(b) 改 reward 公式为 `score * sigmoid(logit)`，让模型同时输出连续 score；(c) 直接去掉 reason-conditioned 链路，回到 direct Yes/No + reason-only logging。
- 回归参考：`.scratch/smoke_qwen_vl_reward.py` 仍然能在不依赖真实 vLLM 服务的前提下覆盖 logging 链路；新跑前可以先跑一遍确认 caption / reason table IR 没回退。
