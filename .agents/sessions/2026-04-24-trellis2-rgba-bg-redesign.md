# Session Handoff: Trellis2 RGBA 背景处理重构

## 任务目的
延续 4.23 session，重构 Trellis2 条件图像的 RGBA 背景处理流程，确保渲染视频与条件图像使用一致的白底背景，消除 reward 评分时的背景不一致问题。

## 执行内容
- 诊断并修复 `openai` 包缺失导致的 `unified_reward_video_aps` 加载失败
- 分析 4.23 方案（`_condition_images_rgba` 私有属性 + `set_render_bg_color()`）在分布式训练下因 sample 重建丢失私有状态而失效的根因
- 设计并实施新方案 "Scheme G"：删除旧的 tensor 版 `_composite_rgba` / `_split_rgba_image` / `set_render_bg_color()`，改为 PIL 级别的 `_composite_rgba_pil` + `_apply_bg_to_condition_images`
- `__post_init__` 简化：RGBA→黑底 RGB PIL（给 encoder 用），不再存储 `_condition_images_rgba`
- `inference()` 的 `decode_output` 块：用局部变量原始 RGBA 图 + `render_bg_color` 合成白底 RGB，覆写 `s.condition_images`（给 reward/日志用）
- 修改 `render_latents` 使用 `ret['alpha']` 做 numpy 级别的 alpha 合成，替代旧的直接取 `ret['shaded']`
- 将 `Trellis2Sample` 从继承 `ImageConditionSample` 改为继承 `I2VSample`
- 清理全部 6 处 debug 代码
- 移除 `pil_image_to_base64` 中的 RGBA→RGB 兜底

## 调试经验
- `_condition_images_rgba` 是私有非 dataclass 属性，在 `_run_owned_pilots` / `gather_samples` 等重建 sample 的场景下会丢失，导致 `set_render_bg_color()` 变成 no-op → 不要依赖 dataclass 外的私有状态跨进程传递
- Trellis2 的 `pipeline.preprocess_image()` 对 RGBA 输入**保留 RGBA 模式**返回（如果有有效 alpha 通道），所以 `_resolve_conditioning` 返回的 `condition_images` 是 RGBA PIL
- 移除 `pil_image_to_base64` 兜底后，eval 在第 11 步 crash：`OSError: cannot write mode RGBA as JPEG`，说明 `_apply_bg_to_condition_images` 设置的 RGB 结果在某处被覆盖或绕过

## 阻塞问题
eval 路径仍有 RGBA 条件图像到达 `pil_image_to_base64`。根据代码分析，`_apply_bg_to_condition_images` 应当在 `decode_output` 块中被调用并生成 RGB PIL，但实际运行仍触发 RGBA JPEG 错误。可能原因：
1. `_apply_bg_to_condition_images` 设置的 RGB PIL 被后续某步骤覆盖（如 `eval_reward_buffer` 内部重建 sample 触发 `__post_init__` → 但此时 condition_images 已是 RGB PIL/tensor，不应变回 RGBA）
2. reward processor 的 `_convert_media_format` 将 tensor 转回 PIL 时产生 RGBA（需确认 `tensor_to_pil_image` 对 4 通道 tensor 的行为）
3. 某些 batch 的 `condition_images` 在 `_resolve_conditioning` 后为 None，跳过了 `_apply_bg_to_condition_images`

## 参考代码
| 文件 | 关键位置 | 说明 |
|------|---------|------|
| `src/flow_factory/models/trellis2/trellis2.py` | `_composite_rgba_pil` (L83-98), `_apply_bg_to_condition_images` (L101-114), `Trellis2Sample.__post_init__` (L250-258), `inference()` decode_output 块 (L1540-1545), `render_latents` (L2327-2340) | 核心改动所在 |
| `src/flow_factory/rewards/unified_reward.py` | `_build_messages` (L634-662), `_prepare_video_inputs` (L570-598) | crash 发生点；condition_image 从 sample 提取后传入 `pil_image_to_base64` |
| `src/flow_factory/rewards/reward_processor.py` | `_convert_media_format` (L119-146), `_compute_pointwise_batch` (L149-164) | tensor/PIL 转换发生在此 |
| `src/flow_factory/utils/image.py` | `pil_image_to_base64` (L668-690), `standardize_image_batch` (L692-790) | RGBA 兜底已移除 |
| `src/flow_factory/samples/samples.py` | `ImageConditionSample.__post_init__` (L369-377) | 父类将 condition_images 标准化为 `List[torch.Tensor]` |
| `third_party/TRELLIS.2/trellis2/pipelines/trellis2_image_to_3d.py` | `preprocess_image` (L127-147) | 对 RGBA 输入保留 RGBA 返回 |

## 最终方案
采用 PIL 级别的 alpha 合成替代旧的 tensor 版 + 私有状态方案。核心设计：
- **encoder 输入**：`__post_init__` 中 RGBA→黑底 RGB PIL→tensor（由 `ImageConditionSample.__post_init__` 完成），不变
- **reward/日志输出**：`inference()` decode_output 时用原始 RGBA + `render_bg_color` 重新合成白底 RGB PIL，直接覆写 `s.condition_images`
- **视频渲染**：`render_latents` 用 numpy alpha 合成 `shaded + bg * (1-alpha)`

该方案不依赖任何私有非 dataclass 属性，跨进程/gather 安全。但当前 eval 仍有 RGBA 泄漏到 reward 链路，需要进一步调查。

## 下一步任务
1. 修复 eval 路径 RGBA 泄漏问题（阻塞）
2. 跑通训练，观察 UnifiedReward APS 分数是否合理
3. 根据训练效果，思考下一步优化方向

## 初步方案
**修复 RGBA 泄漏**：
- 在 `_apply_bg_to_condition_images` 入口加临时 log：打印 `condition_images_rgba` 是否为 None、每个 img 的 `type` 和 `mode`，确认函数是否被调用且 RGBA 正确转换
- 在 `pil_image_to_base64` 入口加 `assert image.mode != 'RGBA'` 定位 RGBA 的精确来源
- 检查 `eval_reward_buffer.add_samples` → `finalize` 路径是否有 sample 重建（触发 `__post_init__` 将 RGB PIL→tensor→PIL 的往返）
- 检查 `standardize_image_batch` 和 `tensor_to_pil_image` 对 4 通道 tensor 的处理
- 若确认是 `__post_init__` 重建导致的，考虑在 `_convert_media_format` 中加 mode 断言或在 reward 调用前再做一次 RGBA→RGB
