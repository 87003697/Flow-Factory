# Session Handoff: trellis2 rollout decode/render OOM 修复方案制定

## 任务目的

针对 `ff-train examples/grpo/lora/trellis2_tex_unified_reward.yaml` 训练 rollout 阶段在 `mesh.fill_holes()` → `cumesh.get_edges()` 触发的 CUDA OOM，制定一套"先 debug gate 再修复"的完整方案并落成 plan 文件。

## 执行内容

- 分析终端 OOM 堆栈，确认直接触发点是 `cumesh.get_edges()`（driver-level 显存申请），根因是 PyTorch caching allocator 持有大量碎片显存不还给 driver
- 阅读 `chunked_mixin.py` 全文，发现 `decode_shape` 里的 `ChunkedDecoderMixin.inject_to(decoder)` 只挂方法不覆盖 `__call__`，`decoder(slat, ...)` 走的是原版 `forward()`，chunked 路径**从未生效**
- 设计 4 层修复方案：(1) `load_pipeline()` 里一次性 inject，(2) 改用 `forward_chunked`，(3) `decode_latents` 清 `subs` spatial cache + `empty_cache()`，(4) 删除 `mesh.fill_holes()`
- 沿用 cfg-fix session 的 NDJSON 探针框架设计 debug gate，包含 5 处探测点（D1~D5）、4 个待证伪假设（H-A~H-D）、3 轮对偶比对（baseline / postfix-cache / postfix-full）
- 补充 6 条改进：(1) 探针同时记录 driver-level 与 torch-level 显存，(2) 多 rank 日志按 rank 分文件，(3) `FF_DEBUG_MAX_SAMPLES` 早退机制，(4) H-D 成立时的三级 fallback，(5) 三轮归因混淆说明表，(6) `forward_chunked` 0 点退化路径兼容性验证
- 对齐 cfg-refactor session 对 `trellis2.py` 和 3 个 YAML 的改动（行号、删除的 `guidance_scale` 字段等），刷新 plan 里所有代码 cite 到最新行号

## 调试经验

本 session 不涉及实际执行，但从前置 session 继承了以下关键教训：

- **只信 runtime evidence**：cfg-fix session 的 5 个 dtype 假设全部错了。OOM 路径上的峰值到底在 shape decoder、tex decoder、fill_holes 还是 renderer 同样不能靠猜
- **driver free vs torch reserved**：cumesh 走 CUDA driver 直接申请，`torch.cuda.memory_reserved()` ≠ 真实可用量；必须同时打 `mem_get_info()` 才能诊断"PyTorch 占着不还"的碎片场景
- **chunked mixin 内部已经做 `empty_cache()`**：`chunked_mixin.py:540, :568` 在每层 merge 前各做一次，归因时不能重复算入 decode_latents 末尾的 `empty_cache`
- **LoRA 改变 `next(parameters())` 顺序**：`_build_sparse_inputs` 已在 cfg-refactor session 改成显式 `torch.float32`
- **临时 yaml 保持 `num_inference_steps=12`**：sparse stage scheduler 直接读 `pipeline.json`，改 YAML 让 trajectory_indices 错位
- **至少 2 GPU**：`Trellis2GRPOTrainer._distributed_upstream_stages` 强依赖 PG

## 参考代码

| 文件 | 关键位置 | 说明 |
|------|---------|------|
| `src/flow_factory/models/trellis2/trellis2.py` | `load_pipeline()` L450-L453 | 现成 fp32 转换循环，改动点：加 `ChunkedDecoderMixin.inject_to(decoder)` |
| `src/flow_factory/models/trellis2/trellis2.py` | `decode_shape()` L2100-L2145 | 改动点：删延迟 inject、`decoder(slat,...)` → `decoder.forward_chunked(slat,...)` |
| `src/flow_factory/models/trellis2/trellis2.py` | `decode_texture()` L2148-L2186 | 改动点：`decoder(slat,...)` → `decoder.forward_chunked(slat,...)` |
| `src/flow_factory/models/trellis2/trellis2.py` | `decode_latents()` L2189-L2240 | 改动点：删 `fill_holes()`，加 `clear_spatial_cache + empty_cache` |
| `src/flow_factory/models/trellis2/trellis2.py` | `render_latents()` L2268+ | 可能的 fallback 改动点（H-D 成立时加 `empty_cache`） |
| `src/flow_factory/models/trellis2/chunked_mixin.py` | 全文 | `forward_chunked` 实现；`inject_to()` 只挂方法不覆盖 `__call__`；L540/L568 merge 前 `empty_cache` |
| `third_party/TRELLIS.2/trellis2/modules/sparse/basic.py` | `clear_spatial_cache()` L767 | `self._spatial_cache = {}` — subs 清理入口 |
| `third_party/TRELLIS.2/trellis2/representations/mesh/base.py` | `fill_holes()` L35-L57 | OOM 直接触发点：`cumesh.CuMesh().get_edges()` |
| `examples/grpo/lora/trellis2_tex_unified_reward.yaml` | 全文 | 临时 yaml 的 base；注意 `guidance_scale` 字段已在 cfg-refactor 中删除 |

## 最终方案

已落成 plan 文件：`.cursor/plans/trellis2-decode-oom-fix_80c90624.plan.md`，包含 11 个 todo，分为 debug gate（4 步）+ 业务改动（4 步）+ 验证与清理（3 步）。核心思路：

1. **Debug gate**：加 NDJSON 探针收集 3 轮 runtime evidence（baseline / postfix-cache / postfix-full），证伪 4 个假设后再动业务代码
2. **真正启用 chunked decoder**：`load_pipeline()` 里对 shape/tex decoder 各 inject 一次
3. **改用 `forward_chunked`**：decode_shape / decode_texture 的调用从 `decoder(slat,...)` 改成 `decoder.forward_chunked(slat,...)`
4. **清 cache + 删 fill_holes**：decode_texture 后清 subs spatial cache + `empty_cache()`；删除 `mesh.fill_holes()`
5. **Fallback**：如果 OOM 转移到 renderer，依次尝试渲染前 `empty_cache`、拆 16 帧为 2×8 sub-batch、最终才考虑 `inference_modules` 驻留优化

## 下一步任务

在新 session 中按 plan 的 11 个 todo 顺序执行，从 `debug-probes`（加 NDJSON 探针）开始。

## 初步方案

下一 session 直接读 plan 文件 `.cursor/plans/trellis2-decode-oom-fix_80c90624.plan.md` + 本 handoff + `ff-debug` SKILL，按以下顺序推进：

1. **debug-probes + debug-yaml**：在 `trellis2.py` 顶部加 `_debug_emit` 函数，在 `decode_latents`/`render_latents` 内加 5 处探针；从 base yaml 拷临时 yaml 到 `/tmp/`
2. **run-baseline**：`CUDA_VISIBLE_DEVICES=0,1 FF_DEBUG_RUN_ID=baseline FF_DEBUG_MAX_SAMPLES=4 ff-train /tmp/trellis2_tex_debug.yaml`，采集 D1~D5 NDJSON
3. **run-postfix-cache**：只加 `clear_spatial_cache + empty_cache`，重跑一轮验证 H-B/H-C
4. **inject + forward_chunked + drop fill_holes**：三步业务改动一起做
5. **run-postfix-full**：跑完 1 epoch，jq 对比三轮数据
6. **cleanup + verify-training**：删探针，用原 yaml 全量验证

潜在风险：
- `forward_chunked` 0 点退化路径可能让下游 marching cubes 报 channel mismatch（plan 已列 `verify-degenerate` todo）
- `clear_spatial_cache()` 后 `subs` 若仍被下游持引用会出错（代码分析确认 `decode_texture` 之后 `subs` 不再被使用，但需 runtime 验证）
- H-D 成立时 OOM 转移到 renderer，需要走 fallback 路径
