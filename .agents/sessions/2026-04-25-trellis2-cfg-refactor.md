# Session Handoff: trellis2 CFG 单一来源重构 + dtype 契约显式化

## 任务目的

承接 [2026-04-25-trellis2-grpo-cfg-fix](2026-04-25-trellis2-grpo-cfg-fix) 的临时 fix（在 `forward()` 里补 `_get_stage_guidance(...)` 兜底），把 Trellis2 各 stage 的 CFG 配置彻底收敛到 `pipeline.json` 单一来源；同时把 `_build_sparse_inputs` 与 `_inference_dense` 里依赖 `next(model.parameters()).dtype` 的脆弱写法改成显式 `torch.float32`，根除将来切到 `finetune_type=full + master_weight_dtype=bfloat16` 时会重新触发的 ratio≠1 隐患。

## 执行内容

- 按 `.cursor/plans/trellis2-cfg-pipeline-refactor_e2c0ba63.plan.md` 8 个 todo 顺序执行
- `_get_stage_guidance(stage)` 收敛为单参数严格读 `pipeline.json`：删 `sampler_params`/`guidance_*` 形参与 3 层 fallback；缺字段直接 `KeyError`（fail-fast）
- `_run_stage_inference` 内部调 `_get_stage_guidance(stage)`，签名删除 `guidance_scale/interval/rescale` 三个 kwargs；`_inference_dense/shape/tex` 仍接收（接收方而非配置源）
- `Trellis2Adapter.inference()` 签名删除 `guidance_scale/interval/rescale/sampler_params` 4 个参数 + 删 `sampler_params = ...` + 更新 docstring
- `Trellis2Adapter.forward()` 签名删除 `guidance_scale/interval/rescale` 3 个参数；唯一解析路径是开头的 `g = self._get_stage_guidance(stage)`；`pred_v.feats.float()` 加防御性注释
- `Trellis2GRPOTrainer._run_owned_pilots` 删 `_get_stage_guidance` 调用 + `_run_stage_inference` 的 guidance kwargs（净 -7 行）
- `_build_sparse_inputs` 删 `stage/stage_resolution` 形参 + `flow_model`/`model_dtype` 查找 + 3 处 cast 改显式 `torch.float32` + 扩 docstring 说明 fp32 契约来由；`_inference_dense` 删 dead code `model_dtype = next(...)` 行；唯一 caller `forward()` 同步去掉位置参数
- 3 个 trellis2 YAML（`trellis2_tex_unified_reward.yaml`、`trellis2_shape_unified_reward.yaml`、`trellis2_shape.yaml`）原计划是给 `train.guidance_scale: 7.5` / `eval.guidance_scale: 7.5` 加注释；第二轮验证 schema 后**直接删除**字段（共 6 处），各自留 1 行 NOTE 指向 `pipeline.json`
- 静态：`PYTHONPATH=src python -c "from flow_factory.models.trellis2.trellis2 import Trellis2Adapter; from flow_factory.trainers.trellis2_grpo import Trellis2GRPOTrainer"` 通过；`grep model_dtype|next(self._unwrap` 在 `trellis2.py` 0 命中
- 行为：2 GPU 冒烟（`shape_slat_1024` + `PickScore`，`max_dataset_size=16`、`render_num_frames=4`、save/eval 关）跑通 epoch 0：`train/ratio_mean=1.0000`、`ratio_std=0`、`clip_frac_total=0`；删字段版本同样通过
- diff：`5 files changed, 65 insertions(+), 76 deletions(-)`

## 调试经验

- **`field(default=...)` 等于"可省略"**：原 plan 说"不删 YAML 字段（schema 共用）"是基于"schema 强制"的假设；实际 `EvalArguments.guidance_scale` / `TrainingArguments.guidance_scale` 都是 `field(default=3.5)`，YAML omit 不会触发校验错误，且 trainer 通过 `**training_args` + `filter_kwargs` 透传时 Trellis2 路径根本不消费它。**唯一**直接读 `training_args.guidance_scale` 的 adapter 是 `qwen_image.py:312`
- **"配置即文档"优先于"YAML schema 视觉一致"**：留个带"我没用"注释的字段反而比缺失更让人困惑；删除 + 1 行 NOTE 强引导用户去看真单一来源（`pretrained_weights/TRELLIS.2-4B/pipeline.json`）
- **LoRA 把 fp32 adapter 参数顶到 `next(model.parameters())` 前端** 的副作用：当前 LoRA 下 `model_dtype = next(...).dtype = fp32`，cast 是 no-op；但切到 `finetune_type=full + master_weight_dtype=bfloat16` 后会真返回 bf16，cast 变成真下采，log_prob 漂移、ratio≠1 重新出现。**显式 fp32 契约才是稳的**（与官方 `pipelines/trellis2_image_to_3d.py::sample_shape_slat` 用 `torch.randn(...).to(device)` 默认 fp32 一致）

## 参考代码

| 文件 | 关键位置 | 说明 |
|------|---------|------|
| `src/flow_factory/models/trellis2/trellis2.py` | `_get_stage_guidance()` L613-L631 | 单参数严格读 `pipeline.json[stage].guidance_strength/interval/rescale`；唯一 CFG 解析入口 |
| `src/flow_factory/models/trellis2/trellis2.py` | `forward()` L1066+ | 签名无 `guidance_*`；进入即 `g = self._get_stage_guidance(stage)`；同时调显式 fp32 的 `_build_sparse_inputs(latents, sparse_coords, ...)` |
| `src/flow_factory/models/trellis2/trellis2.py` | `_build_sparse_inputs()` L1019+ | 显式 `torch.float32`；删 `stage/stage_resolution` 死参；docstring 说明 fp32 契约 |
| `src/flow_factory/models/trellis2/trellis2.py` | `_run_stage_inference()` L1340+ | 内部 `g = self._get_stage_guidance(stage)`；签名无 guidance kwargs |
| `src/flow_factory/models/trellis2/trellis2.py` | `inference()` L1415+ | 签名无 `guidance_*`/`sampler_params` |
| `src/flow_factory/trainers/trellis2_grpo.py` | `_run_owned_pilots()` L257+ | `_run_stage_inference` 调用无 guidance kwargs |
| `src/flow_factory/hparams/training_args.py` | L52, L178 | `guidance_scale: float = field(default=3.5)`，可省略 |
| `pretrained_weights/TRELLIS.2-4B/pipeline.json` | `*_slat_sampler.params` | CFG 真单一来源；tex `guidance_strength=1.0`（无 CFG），shape/dense 各有自己的值 |
| `examples/grpo/lora/trellis2_*.yaml` | `train` / `eval` 块 | 已删除 `guidance_scale` 字段，各留 1 行 NOTE |

## 最终方案

1. **CFG 单一来源**：`_get_stage_guidance(stage)` 只读 `pipeline.json`；`forward()`/`_run_stage_inference()` 在入口调用一次，分别向下游传具体值；`Trellis2GRPOTrainer._run_owned_pilots` 完全不再感知 guidance；YAML 删除 `guidance_scale` 字段
2. **dtype 契约显式化**：`_build_sparse_inputs` 把 sparse feats 显式 `.to(dtype=torch.float32)`，删除 `model_dtype = next(model.parameters()).dtype` 这种依赖 LoRA/full 切换不变性的写法；`_inference_dense` 同步删 dead code
3. **Fail-fast**：`pipeline.json` 缺字段 → `KeyError` 立即暴露，符合 `.cursor/rules/no-defensive-except.mdc`

为什么不在 trainer 层修改：trainer 通用，不该感知具体 adapter 的 stage-aware 配置；`forward()` 已经知道 `stage`，是最自然的修复点（与 fix session 结论一致）。

为什么删 YAML 字段而非保留注释：schema 允许、Trellis2 路径不消费、删除强引导用户去 `pipeline.json` 改配置；wandb 上看到 dataclass 默认 3.5 比看到 7.5 更明确"我不该改这里"。

## 下一步任务

按 `.cursor/plans/trellis2-decode-oom-fix_80c90624.plan.md` 修复 rollout 解码 / 渲染路径的 OOM（与上次 fix-session 列出的相同，本次未启动）。

## 初步方案

`src/flow_factory/models/trellis2/trellis2.py`（`load_pipeline()`、`decode_shape`、`decode_texture`、`decode_latents`）：

1. **真正启用 chunked decoder**：`load_pipeline()` 里 `decoder.convert_to_fp32()` 旁边对 `shape_decoder` 和 `tex_decoder` 各调一次 `ChunkedDecoderMixin.inject_to(decoder)`；删掉 `decode_shape` 里的延迟 inject。当前的延迟 inject 只挂方法不覆盖 `__call__`，`decoder(slat, ...)` 走的还是原 `forward()`，chunked 路径**从未生效**
2. **改用 `forward_chunked`**：`decode_shape` 用 `decoder.forward_chunked(slat, return_subs=True)`；`decode_texture` 用 `decoder.forward_chunked(slat, guide_subs=subs) * 0.5 + 0.5`。mixin 在每个分辨率层 merge 前都会 `torch.cuda.empty_cache()`
3. **清理 spatial cache + 删 fill_holes**：`decode_latents` 在 `decode_texture` 之后对 `subs` 里每个元素调 `clear_spatial_cache()` + `del subs` + `torch.cuda.empty_cache()`；删 `mesh.fill_holes()`（用户已确认可接受 — 它只补 `max_hole_perimeter < 3e-2` 的小破洞，对 reward 影响可忽略，且 `cumesh.get_edges` 是 driver-level 申请，正是当前 OOM 现场）
4. **验证**：直接重跑 `ff-train examples/grpo/lora/trellis2_tex_unified_reward.yaml`；OOM 若已转移到 nvdiffrast/pbr renderer，下一步降 `render_resolution/render_num_frames`；wandb 看 reward 是否回退（一般不会）

潜在风险：
- `forward_chunked` 在某些边缘 sparse 拓扑上可能与原 `forward` 有数值差异（mixin 已存在但未真正用过），先 dryrun 比对 reward
- `clear_spatial_cache()` 后若 `subs` 仍被 `MeshWithVoxel` 等下游持引用会出错，需确认 `decode_texture` 之后 `subs` 真的不再被使用

不在本次范围（plan 已注明）：`render_num_frames` / `render_resolution` 调小、`inference_modules` 多 stage 驻留优化、`mesh.simplify` target 调整。
