# Session Handoff: Trellis2 DGA optimize phase bug fix

## 任务目的
修复 `distributed_group_aligned` sampler 下 Trellis2 GRPO 训练在 `optimize()` 阶段因 `AttributeError: 'NoneType' object has no attribute 'shape'` 崩溃的 bug。

## 执行内容
- 在 `_broadcast_upstream_for_uid`、`_rollout_group`、`_stack_values`（`samples.py` 和 `trellis2.py`）、`optimize`（`grpo.py`）中插入 NDJSON debug 日志
- 用 `ff-train examples/grpo/lora/trellis2_shape_dga_k4.yaml` 复现 bug 并收集日志
- 日志证实：owner 进程调用了 `copy_stage_metadata_from` 导致 `dense_image_cond` 等字段为 Tensor，而 non-owner 为 None；stack 时混合类型触发崩溃
- 移除 `_broadcast_upstream_for_uid` 中的 `s.copy_stage_metadata_from(pilot, self._upstream_stages)` 一行
- 保留日志再次运行验证，训练完整通过（Sampling → Rewards → 4 gradient steps → Training completed）
- 清理所有 debug 日志和残留 import

## 调试经验
- `distributed_group_aligned` sampler 会让同一 rank 上同时存在 owner 和 non-owner 样本，`group_contiguous` 不会，因此后者不触发此 bug
- 上游元数据字段（`dense_image_cond` 等）对下游训练阶段不是必需的，只有 broadcast 字段（`sparse_coords`、`dense_final_latent`、`shape_final_latent`）是必要的

## 参考代码
| 文件 | 关键位置 | 说明 |
|------|---------|------|
| `src/flow_factory/trainers/trellis2_grpo.py` | `_broadcast_upstream_for_uid` (L332-366) | 修复点：移除 metadata copy，仅保留 broadcast 字段赋值 |
| `src/flow_factory/trainers/grpo.py` | `optimize()` (L180-212) | `BaseSample.stack()` 调用点，崩溃发生处 |
| `src/flow_factory/samples/samples.py` | `_stack_values()` (L312-325) | Tensor/None 混合时的直接报错位置 |
| `src/flow_factory/models/trellis2/trellis2.py` | `Trellis2Sample._stack_values()` (L290-299), `render_latents()` (L2301-2363) | sparse tensor 特殊 stack 逻辑；multiview 渲染（下一步改动点） |
| `src/flow_factory/models/trellis2/pipeline.py` | `preprocess_image()` (L301-355) | condition image 的 RGBA/去背景处理（下一步改动点） |

## 最终方案
移除 `_broadcast_upstream_for_uid` 中 owner-only 的 `copy_stage_metadata_from` 调用。这些上游元数据字段对下游训练不需要；移除后所有样本的这些字段保持一致的 None 状态，`stack()` 可正确处理。选择此方案而非在 `_stack_values` 中加防御性检查，是因为问题的根因在于不该拷贝这些字段，而非 stack 逻辑有缺陷。

## 下一步任务
调整 Trellis2 打分相关内容，具体包括四项：
1. **reward 用 condition image 背景色**：提交给 reward model 打分的 condition image 需要基于 RGBA alpha 通道将背景替换为指定颜色（pipeline 输入不动），使参考图背景与渲染背景一致
2. **multiview 渲染初始 azimuth**：从 0° 改为 180°（当前 `render_latents` 中 `yaws_rad` 从 0 开始）
3. **multiview 渲染背景色**：当前硬编码 `bg_color: (0,0,0)`，需可配置
4. **UniReward 适配**：从 `87003697/Flow-Factory` 的 main 分支（已添加为 `upstream` remote）将 `unified_reward.py`、`unified_reward_pairwise.py` 和 registry 更新合并到 trellis2 分支

## 初步方案

### 1. reward 用 condition image 背景色
- **关键点**：pipeline 输入不改动；仅在提交 reward model 打分时，将 condition image 的 RGBA alpha 通道做背景色替换
- **入口**：reward 打分阶段，`condition_images` 从 `Trellis2Sample` 取出送入 reward model 之前
- **改动**：提供一个 `reward_bg_color` 参数（如 `(0,0,0)` 或 `(1,1,1)`），在 reward processor 或 adapter 的 reward 前处理中做 `fg * alpha + bg_color * (1 - alpha)` 转换
- **传参链**：YAML `model.reward_bg_color` 或 `reward.bg_color` → reward 调用前的 condition image 预处理
- **注意**：需确保 reward 看到的参考图背景与 multiview 渲染背景（`render_bg_color`）一致

### 2. multiview 渲染初始 azimuth
- **入口**：`trellis2.py` L2344 `torch.linspace(0, 2*pi, num_frames+1)[:-1]`
- **改动**：加 `render_start_azimuth` 参数（默认 180°），改为 `torch.linspace(start, start + 2*pi, ...)[:-1]`
- **传参链**：YAML `model.render_start_azimuth` → `extra_kwargs` → `trellis2_grpo.py` `_render_kwargs` → `render_latents(start_azimuth=...)`

### 3. multiview 渲染背景色
- **入口**：`trellis2.py` L2351 `{'resolution': resolution, 'bg_color': (0,0,0)}`
- **改动**：加 `render_bg_color` 参数，替换硬编码值
- **传参链**：同上，通过 `_render_kwargs` 传入 `render_latents`

### 4. UniReward 适配
- **来源**：`upstream/main` 分支已 fetch（`git remote add upstream https://github.com/87003697/Flow-Factory.git`）
- **文件**：`unified_reward.py`（pointwise ACS/APS）、`unified_reward_pairwise.py`（groupwise think/flex）
- **改动**：cherry-pick 或直接复制这两个文件到 trellis2 分支；更新 `registry.py` 添加 6 个注册条目
- **依赖**：需要 `openai` 包；`abc.py` 两分支无差异，无基类兼容问题
- **Trellis2 适配要点**：当前 UnifiedReward 的 `required_fields` 用 `video`（List[List[PIL.Image]]），Trellis2Sample 的 `video` 是 `torch.Tensor (T,C,H,W)`，需要在 reward processor 或 adapter 中做 tensor→PIL 转换
