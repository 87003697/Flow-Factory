# Session Handoff: Trellis2 打分改进方案设计

## 任务目的
为 Trellis2 GRPO 训练的打分流程设计并确认改进方案，涵盖：condition image 背景色替换、渲染背景色可配、起始 azimuth 修正、日志可视化增强、UniReward 合并。

## 执行内容
- 阅读 session handoff `2026-04-23-trellis2-dga-optimize-phase-bug.md` 中列出的 4 项下一步任务
- 调研 `preprocess_image`、`render_latents`、`inference`、`RewardProcessor`、`LogFormatter` 等关键路径的完整代码
- 发现 `pil_image_to_tensor` (image.py L612) 会静默丢弃 alpha 通道，导致 RGBA 数据不能直接通过 `condition_images` 字段存储
- 发现 `preprocess_image` 有两个调用点 (L870, L957)，都需要做 RGBA/RGB 分离
- 发现 `Trellis2Sample` 在 `LogFormatter` 中 fallback 到 `_process_base_samples`，condition_images 不会被记录
- 经过多轮方案迭代，最终确定：Trellis2 特有逻辑全封装在 `Trellis2Sample`（`__post_init__` + `set_render_bg_color()`），共享层零改动
- 排查了 `set_render_bg_color()` 改写 `condition_images` 导致 `unique_id` 缓存失效的风险，确认时序安全
- 产出完整 plan 文件，含所有改动点的 before/after 代码对比

## 调试经验
- `ImageConditionSample._id_fields` 包含 `condition_images`，任何对该字段的赋值都会触发 `_unique_id` 缓存清空。但排查所有 `unique_id` 读取点后确认：DGA 分组在 `inference()` 之前完成，reward 分组在之后执行，两者之间无交叉引用，时序安全。
- `pil_image_to_tensor` 对 RGBA 图片只做 `[:, :, :3]` 截断（不是 alpha 合成），所以不能依赖标准化流程保留 alpha 信息。
- `render_start_azimuth` 不值得参数化——从 YAML 到 `render_latents` 要穿过 4 层调用，对一个固定约定来说干扰面过大。

## 参考代码
| 文件 | 关键位置 | 说明 |
|------|---------|------|
| `src/flow_factory/models/trellis2/pipeline.py` | `preprocess_image` (L301-355) | 返回 RGBA 改动点 |
| `src/flow_factory/models/trellis2/trellis2.py` | `Trellis2Sample` (L110-300) | `__post_init__` + `set_render_bg_color()` |
| `src/flow_factory/models/trellis2/trellis2.py` | `preprocess_func` (L858-889, L940-974) | 两处 RGBA/RGB 分离 |
| `src/flow_factory/models/trellis2/trellis2.py` | `inference` (L1353-1519) | `render_bg_color` 透传 + 调 `set_render_bg_color` |
| `src/flow_factory/models/trellis2/trellis2.py` | `render_latents` (L2301-2363) | `bg_color` 参数 + 180 度起始 |
| `src/flow_factory/trainers/trellis2_grpo.py` | `_render_kwargs` (L88-94) | 增加 `render_bg_color` |
| `src/flow_factory/logger/formatting.py` | `_process_sample_list` (L636-660) | 注册 `ImageConditionSample` handler |
| `src/flow_factory/rewards/registry.py` | `_REWARD_MODEL_REGISTRY` (L28-36) | UniReward 6 条注册 |
| `src/flow_factory/samples/samples.py` | `ImageConditionSample` (L361-397) | `_id_fields` 含 `condition_images`，`unique_id` 机制 |
| `src/flow_factory/utils/image.py` | `pil_image_to_tensor` (L607-618) | alpha 通道被截断的位置 |

## 最终方案
**核心设计**：`Trellis2Sample.__post_init__()` 拦截 RGBA condition_images，存入私有属性 `_condition_images_rgba`，同时合成为黑底 RGB 供标准化。`set_render_bg_color(bg_color)` 从 RGBA 重新合成 `condition_images`，覆写为指定背景色的 3ch tensor。所有消费者（RewardProcessor、LogFormatter、stack）只读 `condition_images`，无需任何修改。

选择此方案而非 RewardProcessor hook 或 getter property，是因为：(1) 共享层零改动；(2) 所有消费者透明受益；(3) `unique_id` 时序排查通过。

## 下一步任务
在新 session 中执行 plan 文件 `.cursor/plans/trellis2_scoring_improvements_1fe901ab.plan.md`，按 8 个 todo 逐项实现，然后用 `ff-train examples/grpo/lora/trellis2_shape_dga_k4.yaml` 运行验证。

## 初步方案
- Plan 文件中每个改动点都有完整的 before/after 代码对比，可以直接按序执行
- 建议执行顺序：`preprocess-rgba` -> `sample-bg-composite` -> `render-bg-color` + `render-start-azimuth` -> `trainer-render-kwargs` -> `log-formatter` -> `uni-reward` -> `yaml-example`
- UniReward 需要先 `git fetch upstream` 确保 `upstream/main` 是最新的
- 运行验证时重点关注：(1) conditioning tensor 未变（`image_cond_512`/`1024` 编码结果不变）；(2) reward 拿到的 condition_images 背景色与 render 一致；(3) wandb 日志里 condition_images 和 video 并排显示
