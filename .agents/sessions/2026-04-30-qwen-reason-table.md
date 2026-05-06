# Session Handoff: Qwen Reason Table Logging

## 任务目的
继续完善 Qwen-VL thinking-conditioned reward 的可观测性：当前 reason 已经进入 video caption，但 caption 会截断，下一步要评估是否额外写入 wandb table 以查看完整 reason/thinking。

## 执行内容
- `QwenVLSideBySideReward` 已实现 thinking-conditioned 链路：`reason -> reason-conditioned yes/no`，`enable_reason=True` 时 `RewardModelOutput.extra_info={"reasons": [...]}`。
- reward 失败路径已改为 sentinel：reason API 抛错时使用 `"[reason failed]"` 作为 assistant turn 继续 yes/no，不回退 direct yes/no，不制造 NaN。
- `RewardProcessor._store_reward_extra_info` 已被压扁：只把每条 reason 写入 `sample.extra_kwargs["reward_reason"] = reason`。
- `logger/formatting.py` 已被压扁：`_format_reward_reason_caption` 直接读取 `sample.extra_kwargs["reward_reason"]`，截断到 120 字符后拼入 caption。
- `.scratch/smoke_qwen_vl_reward.py` 已新增覆盖：processor 写入扁平 `reward_reason`，formatter 渲染 `reward | reason | prompt`。
- smoke 已在 `grpo3d_trellis2` 环境通过：`source /home/zhiyuan_ma/anaconda3/etc/profile.d/conda.sh && conda activate grpo3d_trellis2 && python .scratch/smoke_qwen_vl_reward.py`，输出 `ALL SMOKE TESTS PASSED`。

## 调试经验
- 用户实际看到的 caption 类似：`0.88 | qwen_vl_side_by_side: The user wants... | prompt`。这说明旧训练进程/旧 formatter 仍在运行，或者 wandb 展示的是旧 media；新扁平 formatter 重启后不应再带 `qwen_vl_side_by_side:` 前缀。
- 即使 parser 正常工作，caption 也不会显示 `<think>` 标签；vLLM `--reasoning-parser qwen3` 会把 `<think>...</think>` 剥离到 `message.reasoning`。
- 当前 caption 只保留 120 字符，Qwen3.5 thinking 起手很啰嗦，所以用户会觉得“看不到 thinking 的有效内容”。实际问题是 caption 太短，不是 reason 没传通。
- `ReadLints` 只报 `formatting.py` 里已有可选依赖 `av` 无法解析 warning，和本轮改动无关。
- 当前工作树曾检测到 `examples/grpo/lora/trellis2/qwen_vl_reward.yaml` 被删除；不要自动恢复，下一 session 先确认这是用户意图还是临时状态。

## 参考代码
| 文件 | 关键位置 | 说明 |
|------|---------|------|
| `src/flow_factory/rewards/qwen_vl_video_reward.py` | `__call__`、`_score_single`、`_query_reason_text` | 生成 reason、reason-conditioned yes/no、返回 `extra_info["reasons"]` |
| `src/flow_factory/rewards/reward_processor.py` | `_store_reward_extra_info` | 把 `extra_info["reasons"]` 写入扁平 `sample.extra_kwargs["reward_reason"]` |
| `src/flow_factory/logger/formatting.py` | `_format_reward_reason_caption`、`_build_sample_caption`、`LogFormatter.format_dict` | 当前 caption 截断 reason；若做 table，这里是 sample list 分流入口 |
| `src/flow_factory/logger/wandb.py` | `WandbLogger._convert_to_platform` | wandb backend 当前只处理 `LogImage`/`LogVideo`，若新增 table IR，需要在这里转 `wandb.Table` |
| `src/flow_factory/logger/abc.py` | `Logger.log_data`、`_recursive_convert` | logger IR 到平台对象的通用转换链路 |
| `.scratch/smoke_qwen_vl_reward.py` | `test_reward_reason_stored_flat_and_captioned` 等 | 无服务器 smoke，适合改 logging side-channel 后快速回归 |

## 最终方案（当前已采用）
- 训练用 reward 语义：`reason -> reason-conditioned yes/no`。
- reason 存储：极限扁平，`sample.extra_kwargs["reward_reason"] = reason`。
- caption：`reward | truncated reason | prompt`，其中 reason 当前只保留 120 字符。
- 不在 reward 名下嵌套 reason；这是用户明确要求的“极限扁平”版本。

## 下一步任务
在下个 session 评估并可能实现“full reason 写入 wandb table”的方案，解决 caption 太短导致无法查看完整 thinking/reason 的问题。

## 初步方案
- 优先做 **text-only table**，不要把视频重复嵌入 table：原 `train_samples`/`eval_samples` 继续负责视频展示，新加 `train_samples_reasons`/`eval_samples_reasons` 负责完整文本。
- 在 `formatting.py` 新增轻量 IR，例如 `LogReasonTable(samples: List[BaseSample])`，列建议为 `idx`, `prompt`, `reward`, `reason`。
- 在 `LogFormatter.format_dict()` 中特殊处理 key 为 `train_samples` / `eval_samples` 的 sample list：保留原 key 的 media 输出，同时额外产出 `f"{key}_reasons"`。
- 在 `wandb.py` 增加 `LogReasonTable -> wandb.Table` 转换；先仅支持 wandb，swanlab/tensorboard 可以透传或后续补。
- 风险点：`Logger.log_data` 对 dict 返回会 `final_dict.update(converted)`，所以最好在 `format_dict()` 层直接生成两个顶层 key，避免 backend conversion 时丢原 key 前缀。
- 回归测试：新增 smoke 覆盖 `LogFormatter.format_dict({"train_samples": samples})` 产出 `train_samples` 和 `train_samples_reasons`；wandb 转换可用 monkey-patch/fake table 轻测，或先人工 inspect。
