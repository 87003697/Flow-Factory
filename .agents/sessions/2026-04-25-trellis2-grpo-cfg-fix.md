# Session Handoff: trellis2-grpo step-0 ratio CFG fix

## 任务目的

修复 Trellis2 GRPO 训练 tex stage 时 step 0 `train/ratio≈0.96`、`clip_frac_low/high≈1.0` 的精度 bug（PPO ratio 在 policy 未更新时本应严格 = 1.0，否则梯度被 1e-4 clip range 全切，训练无效）。

## 执行内容

- 读 `.cursor/plans/fix-trellis2-grpo-step0-ratio_0e7f6fd8.plan.md`，按 §5 设计在 5 处插入 NDJSON 探针（P1~P5），统一写到 `.cursor/debug-d0f0de.log`，用 `_debug_emit` + `FF_DEBUG_RUN_ID` env 区分 baseline / postfix / postfix7gpu
- 单卡跑不通（`Trellis2GRPOTrainer._distributed_upstream_stages` 调 `dist.broadcast` 强依赖 PG），降级到 2 GPU 并把 `/tmp/trellis2_tex_debug.yaml` 调小（dataset 16、render 4、save/eval 全关）
- 撞了 `num_inference_steps=4` 与 tex scheduler 真实 12 steps 的不一致 → `tex_log_probs=None`，恢复 12
- baseline 跑通，jq 分析判定 H1/H2/H3/H5 全部 REJECTED：所有 dtype 实测都是 fp32（plan 推测的 bf16 cast 因为 LoRA 把 fp32 参数顶到 `next(parameters())` 前端而变成 no-op）
- 由 P2/P3 数值对比发现 train velocity_norm=1021 vs rollout velocity_norm=849 — CFG 行为不一致
- 翻 `pretrained_weights/TRELLIS.2-4B/pipeline.json` 确认 `tex_slat_sampler.guidance_strength=1.0, guidance_interval=[0.6, 0.9]`，rollout 通过 `_get_stage_guidance` 读到这些值并跳过 CFG，但 train `forward()` 用 `training_args.guidance_scale=7.5` 强行做了 CFG
- 在 `Trellis2Adapter.forward()` 解析 `t_val` 后补 `_get_stage_guidance(stage, ...)` 调用，对齐 rollout
- postfix 2 GPU 复跑：max_abs_diff 从 0.014 降到 1.19e-7
- postfix 7 GPU 全量跑通 epoch 0：7 ranks 聚合 max_abs_diff=3.99e-6（远 < 1e-4 阈值），wandb `train/ratio_mean=1.0000`、`clip_frac_total=0`、`grad_norm=0.0010`（健康）
- 用户 mark as fixed → 按 §5.7 清理所有探针、`_debug_emit`、debug log、临时 YAML；ff-review SAFE

## 调试经验

- **Plan 的所有 dtype 假设全部错了**：根因是 `model_dtype = next(self._unwrap(flow_model).parameters()).dtype` 在 LoRA 注入后返回 `torch.float32`（PEFT LoRA 参数排在 base bf16 参数前面），原 cast 是 no-op。**任何依赖 `next(model.parameters()).dtype` 的代码都该警惕 LoRA 改变迭代顺序**
- **代码看起来像 bug ≠ 是 bug**：照 plan 直接改 `_build_sparse_inputs` → 把一个 no-op cast 改成另一个 no-op cast，wandb ratio 还是 0.96。必须用探针拿 runtime evidence 才能拒掉假设
- **train/inference 不一致是 GRPO ratio≠1 的常见根因**：Trellis2 的 `_get_stage_guidance` 早就在 `inference()` 和 `_run_owned_pilots` 里被调用，唯独 `forward()` 漏调，造成 train 路径用 `training_args.guidance_scale=7.5` 而 rollout 用 pipeline.json 的 1.0
- **探针对偶比对**（rollout 与 train 同 (stage, t_val) 配对 x_t_norm 和 pred_v_norm）是发现 CFG 不一致的关键，单看任何一边都看不出
- **单卡 debug Trellis2GRPO 行不通**：`_distributed_upstream_stages` 强依赖 PG；至少 2 GPU
- **`num_inference_steps` YAML 字段对 sparse stage 没用**：tex/shape scheduler 直接读 pipeline.json 的 `params.steps`（固定 12），改 YAML 只会让 trajectory_indices 错位

## 参考代码

| 文件 | 关键位置 | 说明 |
|------|---------|------|
| `src/flow_factory/models/trellis2/trellis2.py` | `Trellis2Adapter.forward()` L1109-L1129 | 修复点：在解析 `t_val` 后调 `_get_stage_guidance` 对齐 rollout |
| `src/flow_factory/models/trellis2/trellis2.py` | `_get_stage_guidance()` L613 | 现成的 stage-aware guidance 解析；同时在 `inference()` L1542、`_run_owned_pilots` (trellis2_grpo.py:260) 被调 |
| `pretrained_weights/TRELLIS.2-4B/pipeline.json` | `tex_slat_sampler.params` | `guidance_strength=1.0, guidance_interval=[0.6, 0.9]` — tex stage 默认无 CFG |
| `src/flow_factory/trainers/grpo.py` | `optimize()` L266 ratio 计算 | 探针 P4 曾插在这里，已清理 |
| `src/flow_factory/trainers/trellis2_grpo.py` | `_distributed_upstream_stages` / `_broadcast_tensor` | 强依赖 `dist.init_process_group`，单卡不可用 |

## 最终方案

`Trellis2Adapter.forward()` 在解析 `t_val` 后、计算 `apply_cfg` 前，调用 `_get_stage_guidance(stage, ...)` 把 `training_args.guidance_scale/interval/rescale` 当 fallback，由 pipeline.json 的 stage-specific 值优先生效。这与 `inference()` / `_run_owned_pilots` 的解析路径完全对称，恢复了 train/inference 在 CFG 配置上的一致性契约。

为什么不在 trainer 层修改：trainer 通用，不该感知具体 adapter 的 stage-aware 配置；`forward()` 已经知道 `stage`，是最自然的修复点。

## 下一步任务

按 `.cursor/plans/trellis2-decode-oom-fix_80c90624.plan.md` 修复 rollout 解码 / 渲染路径的 OOM：从三个层面降低峰值显存与 cumesh 触发的 driver-level OOM。

## 初步方案

`src/flow_factory/models/trellis2/trellis2.py`（`load_pipeline()`、`decode_shape`、`decode_texture`、`decode_latents`）：

1. **真正启用 chunked decoder**：在 `load_pipeline()` 里 `decoder.convert_to_fp32()` 旁边对 `shape_decoder` 和 `tex_decoder` 各调一次 `ChunkedDecoderMixin.inject_to(decoder)`；删掉 `decode_shape` 里的延迟 inject。当前的延迟 inject 只挂方法不覆盖 `__call__`，`decoder(slat, ...)` 走的还是原 `forward()`，chunked 路径**从未生效**
2. **改用 `forward_chunked`**：`decode_shape` 用 `decoder.forward_chunked(slat, return_subs=True)`；`decode_texture` 用 `decoder.forward_chunked(slat, guide_subs=subs) * 0.5 + 0.5`。mixin 在每个分辨率层 merge 前都会 `torch.cuda.empty_cache()`，对碎片化场景有效
3. **清理 spatial cache + 删 fill_holes**：`decode_latents` 在 `decode_texture` 之后对 `subs` 里每个元素调 `clear_spatial_cache()` + `del subs` + `torch.cuda.empty_cache()`；删除 `mesh.fill_holes()`（用户已确认可接受 — 它只补 `max_hole_perimeter < 3e-2` 的小破洞，对 reward 影响可忽略，且 fill_holes 内部的 `cumesh.get_edges` 是 driver-level 申请，正是当前 OOM 现场）
4. **验证**：直接重跑 `ff-train examples/grpo/lora/trellis2_tex_unified_reward.yaml`，看 OOM 是否消失；若仍 OOM，看堆栈是否已经移到 nvdiffrast/pbr renderer（性质相同的 driver-level 申请，下一步可能要降 `render_resolution/render_num_frames`）；wandb 上看 reward 是否回退（一般不会）

潜在风险：
- `forward_chunked` 在某些边缘 sparse 拓扑上可能有数值差异（mixin 已存在但未真正用过），先做一次 dryrun 比对 reward
- `clear_spatial_cache()` 之后如果 `subs` 还被 `MeshWithVoxel` 等下游持引用会出错，需要确认 `decode_texture` 之后 `subs` 是否真的不再被使用

不在本次范围（plan 已注明）：`render_num_frames` / `render_resolution` 调小、`inference_modules` 多 stage 驻留优化、`mesh.simplify` target 调整。
