# Session Handoff: Qwen-VL Thinking-Conditioned Reward

## 任务目的
把 `QwenVLSideBySideReward` 从「JSON 打分」收敛到 `yes_no_logprob`，并在此基础上加一条可选的 **thinking-conditioned** 评分链路：先让 VLM 生成 reason，再用 reason 作为 assistant turn 让模型给出 1-token Yes/No；reason 同步透出到 wandb caption 便于排查。

## 执行内容
- 删除 reward 中所有 JSON 路径（`{score, reason}` 解析、`response_format`、`score_min/max`、`normalize`、`canonicalize` 等抽样参数与提示模板），保留 `yes_no_logprob` 单一打分模式。
- 在 `qwen_vl_video_reward.py` 加 `_query_reason_text`：单独走 `enable_thinking=True` + `max_tokens=reason_max_tokens` 的 chat completion；按 `reasoning > reasoning_content > content` 的优先级读 `message.model_dump()`，三者皆空抛 `ValueError`，无重试。
- 新增 `REASON_PROMPT` 与 `REASON_CONDITIONED_YES_NO_PROMPT`，配套 `_build_reason_messages` / `_build_reason_conditioned_messages`；`_score_single` 在 `enable_reason=True` 时按 reason -> yes/no 顺序串行两次 API（共用同一个 `max_concurrent` 信号量，依赖 vLLM `--enable-prefix-caching` 复用图像 prefill）。
- Reward 类去掉所有 instance-level cache（`_yes_no_cache` / `_reason_cache` / `_add_to_yes_no_cache` / `_build_cache_key` / `max_cache_size`），class 完全无状态，跨 `ThreadPoolExecutor` worker 不再有共享可变态。
- `__call__` 收集 `(score, reason)` 后：`enable_reason=True` 时返回 `extra_info={"reasons": [...]}`，`False` 时返回 `extra_info={}`（保证关闭 thinking 时与改造前完全等价）。
- `RewardProcessor` 加 `_store_reward_extra_info` 小函数，pointwise/groupwise 路径调用它把 `output.extra_info["reasons"]` 写到 `sample.extra_kwargs["reward_extra"][reward_name]["reasons"]`，全程显式 key 检查、不用 `getattr/setdefault`。
- `logger/formatting.py` 加 `_format_reward_reason_caption`：从 `reward_extra` 读单条 reason、做 `|` 转义和长度截断；`_build_sample_caption` 拼装顺序固定为 `reward | reason | prompt`（reason 为空时自然折叠回 `reward | prompt`）。
- 失败 reason 改为 sentinel `"[reason failed]"`：reason API 抛错 → log warning → 用 sentinel 当 assistant turn 继续走 reason-conditioned yes/no。整批语义保持一致，不再产生 NaN，也不会回退到 direct yes/no。
- 重写 smoke 测试 `.scratch/smoke_qwen_vl_reward.py`：覆盖 direct yes/no 回归、reason-conditioned 串行调用、reason 失败 sentinel、`enable_reason=False` 必须不调用 `_query_reason_text`、`extra_body`/`max_completion_tokens`/`logprobs` flag 契约、reason 字段优先级（`reasoning` > `reasoning_content` > `content`）。共 9 个 top-level 测试函数（其中 `test_reason_field_priority` 含 4 个 sub-case），全绿。
- YAML `examples/grpo/lora/trellis2/qwen_vl_reward.yaml` 加 `enable_reason: true` + `reason_max_tokens: 1024`，并补充 vLLM 启动前置：`--reasoning-parser qwen3`（旧版用 `deepseek_r1` 兜底）+ `--enable-prefix-caching`；顶部注释说明并发预算和 sentinel 行为。

## 调试经验
- vLLM 不开 `--reasoning-parser` 时 `<think>...</think>` 会被原样塞进 `message.content`，`reasoning` / `reasoning_content` 全为 null，captions 会出现裸 think 标签且 yes/no 第二轮拿到的是「思考脚本本身」当上下文 —— 必须开 parser。
- vLLM ≥ 0.17 把字段从 `reasoning_content` 改成 `reasoning`（RFC vllm-project/vllm#27755），所以 reward 用一张优先级表读字段，避免锁死单一版本。
- Qwen3.5 thinking 模式输出 800+ tokens 是常态，`reason_max_tokens` 设太小（< 1024）会让 reason 在中途被截断、`</think>` 不出现，第二轮 yes/no 拿到的是 dangling assistant turn，表现为 `reward_*_group_std_mean` 偏低。
- Reward 类挂 instance-level cache 在 `ThreadPoolExecutor` 下是共享可变态，多 worker 写同一份 dict 是 race。每个 rollout 渲染视频长得很像，cache key 也容易被复用为「假阳性命中」，所以直接整个去掉 cache，转而依赖 vLLM 的 prefix cache 来省图像 prefill。
- Reason 失败如果短路成 NaN+mean fill，整批就同时存在「reason-conditioned」和「无 reason」两套 reward 语义，`group_std` 看着会异常。改成 sentinel 后整批语义一致、`reasons` 列表对每个样本都非空，wandb caption 直接看到 `[reason failed]` 即可定位。

### 2026-04-29 PM 补充诊断（vLLM probe-driven）
- 现网 vLLM **没**带 `--reasoning-parser`，curl 探针返回 `"reasoning": null`、`content` 直接以 `Thinking Process: ...` 开头，没有任何 `<think>` 标签。一度误判 Qwen3.5 是「prose-mode、不用 `<think>` 协议」。
- 查 HF `Qwen/Qwen3.5-9B` 的 `chat_template.jinja` 后纠正：`<|im_start|>assistant\n<think>\n` 由 chat template 当作 prompt 前缀**注入**，开标签不会出现在模型生成里，所以 `content` 看不到 `<think>` 是预期行为；模型本应在某点输出 `</think>\n` 然后给 final answer。即"prose-mode"判断错误，Qwen3.5 完全走 `<think>...</think>` 协议。
- `--reasoning-parser` 名字考据：vLLM 0.9+ 提供 `qwen3` parser，覆盖 Qwen3 / Qwen3.5 / Qwen3-VL 全家族（参见 vLLM 官方 reasoning_outputs 文档与 vLLM Forums 上 Qwen3.5 thinking 帖）。`deepseek_r1` 是历史 fallback，对同样 `<think>...</think>` 格式也兼容，但 Qwen3.5 优先用 `qwen3`。`reasoning-parser deepseek_r1` 这个建议在最初的 yaml 注释里写过，已修正为 `qwen3` + 落后版本兜底。
- 探针实证 Qwen3.5-9B 的啰嗦：单条 "9.11 vs 9.8 which is greater?" 在 `max_tokens=512` 下还在第 8 步「Final Decision」，`finish_reason="length"`。所以 `reason_max_tokens` 从 384 上调到 1024 是必要的；如果跑 A/B 后看到 caption 末尾还是被截在中段（不像完整句号收尾），需要继续上调。

### vLLM 重启验证 checklist
1. **重启服务**（注意加上 `--reasoning-parser qwen3` 与 `--enable-prefix-caching`）：
   ```bash
   CUDA_VISIBLE_DEVICES=7 vllm serve Qwen/Qwen3.5-9B \
       --served-model-name Qwen/Qwen3.5-9B --host 0.0.0.0 --port 8000 \
       --trust-remote-code --gpu-memory-utilization 0.9 \
       --enable-prefix-caching \
       --reasoning-parser qwen3 \
       --limit-mm-per-prompt '{"image": 32}' \
       --max-num-seqs 512 --max-model-len 16384
   ```
2. **探针 1：`reasoning` 字段非空**（开启 thinking）：
   ```bash
   curl -s http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model":"Qwen/Qwen3.5-9B","messages":[{"role":"user","content":"hi"}],"extra_body":{"chat_template_kwargs":{"enable_thinking":true}},"max_tokens":1024}' \
     | python -m json.tool | grep -E '"(reasoning|reasoning_content|content)"'
   ```
   预期：`"reasoning"`（vLLM ≥ 0.17）或 `"reasoning_content"`（vLLM ≤ 0.16）非 null，`"content"` 是干净的 final answer，**不含 `<think>` 标签**。
3. **探针 2：关 thinking 立即出 yes/no**（reward 第二步契约）：
   ```bash
   curl -s http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model":"Qwen/Qwen3.5-9B","messages":[{"role":"user","content":"Answer Yes or No: is 9.8 > 9.11?"}],"extra_body":{"chat_template_kwargs":{"enable_thinking":false}},"max_completion_tokens":1,"logprobs":true,"top_logprobs":8}' \
     | python -m json.tool
   ```
   预期：`finish_reason="length"`、`content` 仅一个 token、`logprobs.content[0].top_logprobs` 列出 Yes/No 的 logprob。
4. **smoke 回归**（无服务器依赖）：`PYTHONPATH=src python .scratch/smoke_qwen_vl_reward.py` 应当 9 项全绿（含 `test_reason_field_priority` 的 4 个 sub-case）。
5. **训练首跑**：盯 wandb caption，前 1-2 个 epoch 抽查 reason 是否：(a) 不是 `[reason failed]`、(b) 末尾收尾完整、(c) 长度处于 `reason_max_tokens` 上限以内。如全部满足，进入 A/B 主跑。

## 参考代码
| 文件 | 关键位置 | 说明 |
|------|---------|------|
| `src/flow_factory/rewards/qwen_vl_video_reward.py` | `_query_reason_text` (L194)、`_score_single` (L358)、`__call__` (L424)、`REASON_FAILED_SENTINEL` (L356) | reward 实现主体；reason / yes/no 双路分发与 sentinel 失败兜底都在这里 |
| `src/flow_factory/rewards/reward_processor.py` | `_store_reward_extra_info` (L154) 及其在 pointwise/groupwise 的调用点 | 把 reward 的 `extra_info["reasons"]` 写进 `sample.extra_kwargs["reward_extra"]` |
| `src/flow_factory/logger/formatting.py` | `_format_reward_reason_caption` (L192)、`_build_sample_caption` (L220) | wandb caption 拼装为 `reward \| reason \| prompt` |
| `examples/grpo/lora/trellis2/qwen_vl_reward.yaml` | `rewards`/`eval_rewards` 块、顶部注释 | 训练配置；vLLM 启动前置和 thinking 调优要点都在注释里 |
| `.scratch/smoke_qwen_vl_reward.py` | 9 个 top-level 测试函数（`test_reason_field_priority` 含 4 个 sub-case） | 直接 `PYTHONPATH=src python .scratch/smoke_qwen_vl_reward.py` 验证 reward 行为 |

## 最终方案
- 评分链路：`reason -> reason-conditioned yes/no` 串行两次 API，reason 计入 reward 的语义定义。
- 失败处理：reason 失败用 sentinel `"[reason failed]"` 当 assistant turn 继续推 yes/no（不重试、不 NaN、不退回 direct yes/no）；yes/no 失败才退回 NaN 由 batch mean 填补。
- 状态：reward 类彻底无 cache（依赖 vLLM prefix cache 避免双倍图像 prefill）。
- 可观测性：reason 通过 `RewardModelOutput.extra_info["reasons"]` -> `sample.extra_kwargs["reward_extra"]` -> wandb caption 全链路打通；caption 顺序 `reward | reason | prompt`。
- 关闭 thinking 的回归：`enable_reason=False` 时整条 thinking 路径完全旁路，`extra_info={}`，与重构前等价。

## 下一步任务
跑 A/B 对比训练，量化 **enable_reason=true vs false** 在 Trellis2 tex GRPO 上的效果是否有提升。

## 初步方案
- **配置入口**：`examples/grpo/lora/trellis2/qwen_vl_reward.yaml`。两组分别置 `enable_reason: true / false`，其它 hyperparam 保持一致；`reason_max_tokens` 先用 1024，控制变量。
- **vLLM 服务**：thinking 组必须确认启动参数包含 `--reasoning-parser qwen3` + `--enable-prefix-caching`；可在服务起来后手动 curl 一次 `enable_thinking=true` 验证 `message.reasoning` 字段非 null（非 thinking 组无所谓，但建议沿用同一个 server 以排除变量）。
- **运行规模**：保持现 7-GPU 训练规格，先各跑 ~50 epoch 看趋势；关键指标对照：
  - reward 主信号：`train/reward_qwen_vl_side_by_side_mean` 收敛速度与终点
  - 探索性：`reward_*_std`、`reward_*_group_std_mean`（thinking 应让 group 内分得更开）
  - 训练动力：`policy_loss`、`grad_norm`、`clip_frac_*`
  - 服务侧：vLLM `requests_running` / token usage 是否翻倍（如果 prefix cache 没生效就会暴露）
- **wandb 验证**：`train_samples` / `eval_samples` 的 caption 应当出现真实 reason 文本（不是 `[reason failed]` 或空）；如果大量样本是 sentinel，先排查 vLLM 服务 `--reasoning-parser`、`reason_max_tokens` 截断、timeout。
- **风险点**：
  - 双倍 API 调用 → throughput 减半，看 `Sampling` 进度是否成为瓶颈，必要时下调 `num_workers` 或 `max_concurrent`。
  - reason 长度尾部分布如果超过 `reason_max_tokens`，第二轮 yes/no 会拿到 dangling 思考；先读几条 caption 确认尾部是否完整收尾。
  - thinking 组上下文更长 → vLLM `--max-model-len` 余量是否够（当前 16384，单样本 reason ≤ 1024 + yes/no prompt 应该绰绰有余，但 SBS 图像 token 也算在内）。
- **退出条件**：跑到两组 reward 曲线趋稳（或一组明显退化）即可下结论；若 thinking 组 group_std 显著更高、终点 reward 更高 → 收编为默认；反之则保留为 opt-in。
