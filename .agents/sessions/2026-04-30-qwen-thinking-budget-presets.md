# Session Handoff: qwen-vl thinking budget + shape/tex preset

## 任务目的
把 `QwenVLSideBySideReward` 的"思考力度"控制收敛到单旋钮 `reason_thinking_token_budget`，并把 prompt 按 tex/shape 两个阶段拆开（shape 阶段渲染是 gray-clay，规则不一样），顺手清理 YAML 里那一堆只在复读默认值的字段。

## 执行内容
- 确认 vLLM 0.19.1 对 Qwen3 的支持：`thinking_token_budget` 必须配合 `--reasoning-parser qwen3 --reasoning-config '{"reasoning_start_str":"<think>","reasoning_end_str":"</think>"}'` 才生效。
- 简化 reward 接口：删掉 `reason_max_tokens`，只暴露 `reason_thinking_token_budget`；`max_tokens` 内部派生为 `budget + REASON_FINAL_TOKEN_MARGIN`，给 `</think>` + 最终回答留余量。
- 把 prompt 拆成两段 (`EVALUATION_FRAMEWORK` + `YES_NO_DECISION`)，同时把 `_TEX_*` / `_SHAPE_*` 三组 fragment 收进模块顶层的 `_PROMPT_PRESETS` 字典。
- `__init__` 读 `extra_kwargs.prompt_preset`（默认 `"tex"`），调用新的 `_configure_prompts()` helper 写五个大写实例属性 (`self.EVALUATION_FRAMEWORK`、`self.YES_NO_DECISION`、`self.YES_NO_PROMPT`、`self.REASON_PROMPT`、`self.REASON_CONDITIONED_YES_NO_PROMPT`)。
- 三处构造 message 的方法 (`_build_user_text` / `_build_reason_user_text` / `_build_reason_conditioned_messages`) 全部改成读实例属性。
- `tex_qwen_vl_reward.yaml` 加 `prompt_preset: "tex"`，新建 `shape_qwen_vl_reward.yaml`（基于 `shape.yaml`，target_flow_model `shape_slat_1024`, 24 帧, 白底）写 `prompt_preset: "shape"`。
- 两份 YAML 都精简：删掉 6 个跟代码默认完全一致的字段 (`api_key` / `max_retries` / `timeout` / `max_frames` / `top_logprobs` / `canonicalize`)，只保留真正 override 的。
- vLLM screen 重启用上 `--reasoning-config`；smoke (`.scratch/smoke_qwen_vl_reward.py`) 全绿，两份 YAML 加载后能正确读出对应 preset 的 framework。

## 调试经验
- `reason_thinking_token_budget` 跟 `max_tokens` 不是同一个概念：前者只限制 `<think>` 内 token 数，后者是整体上限；vLLM 在思考超预算时会强制插入 `</think>`，所以 `max_tokens` 必须留 margin (我们用了 1024) 否则没空间出最终答案。
- 不开 `--reasoning-config` 的时候 `thinking_token_budget` 是哑参数，不会报错但也不生效；只看 `--reasoning-parser` 容易忽略。
- `RewardArguments.from_dict` 对 extra_kwargs 走"未知字段全部 capture + WARNING"路线，所以 YAML 里写错字段名不会报错；精简掉默认复读字段的副作用是 warning 行更干净，typo 保护反而更明显。

## 参考代码
| 文件 | 关键位置 | 说明 |
|------|---------|------|
| `src/flow_factory/rewards/qwen_vl_video_reward.py` | `_PROMPT_PRESETS`, `_configure_prompts`, `__init__`, `_query_reason_text` | tex/shape preset 选择 + thinking_token_budget 单旋钮逻辑 |
| `examples/grpo/lora/trellis2/tex_qwen_vl_reward.yaml` | `rewards` / `eval_rewards` block | tex 阶段配置；`prompt_preset: "tex"`, `budget: 2048` |
| `examples/grpo/lora/trellis2/shape_qwen_vl_reward.yaml` | `rewards` / `eval_rewards` block | shape 阶段配置；`prompt_preset: "shape"`, `budget: 1024`, gray-clay rubric |
| `src/flow_factory/hparams/reward_args.py` | `RewardArguments` 顶层字段 | 决定哪些字段不算 extra_kwargs |
| `src/flow_factory/rewards/unified_reward.py` | `UnifiedRewardAPIBase.__init__` (105-111) | api_base_url / api_key / max_concurrent / max_retries / timeout 默认值来源 |
| `.scratch/smoke_qwen_vl_reward.py` | 全文 | 离线 smoke：fake API，验证 budget→max_tokens、preset→framework 的路径 |

## 最终方案
- **思考力度只一个旋钮**：YAML 里就写 `reason_thinking_token_budget`；`max_tokens` 自动派生，杜绝两者不一致导致 thinking 被腰斩或 final 段没空间的坑。
- **stage rubric 用 preset 而不是子类**：`prompt_preset: "tex" | "shape"` 在同一个 reward 类里切换 prompt fragment，避免类继承层级膨胀；后续要加新阶段（比如 `pbr`）只需在 `_PROMPT_PRESETS` 加一行。
- **两段式 prompt (FRAMEWORK + DECISION)** 而不是更细的多段：`YES_NO_DECISION` 在直接 yes/no 路径和 reason-conditioned 路径用的是同一段文字，做 reason on/off ablation 时差异严格只来自 reasoning context，不会被 prompt 微差污染。
- **YAML 只保留 override**：所有跟 reward 类默认值相同的字段全部删掉，只留 `name`/`reward_model`/`api_base_url`/`vlm_model`/`max_concurrent`/`tile_resolution`/`enable_reason`/`reason_thinking_token_budget`/`prompt_preset` 这些真正区分本次配置的旋钮，加一行注释指明 fall-back 来源。
- **vLLM 启动命令固化**：在两份 YAML 顶部注释里写明 `--reasoning-parser qwen3 --reasoning-config ...` 是 thinking_token_budget 生效的前提，避免下次起服务时漏掉。
