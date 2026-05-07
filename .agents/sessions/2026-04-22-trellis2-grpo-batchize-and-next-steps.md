# Session Handoff: Trellis2 GRPO Batch-ize & DDP Fix 完结

## 任务目的

本次 session 完成了两项核心工作：(1) 修复 DDP "ready twice" 错误，(2) 将 `_inference_shape` 和 `_inference_tex` 的 per-sample 循环改为单次 batched pass，提高推理吞吐。

## 执行内容

- 确认 DDP "ready twice" 崩溃的根因是 `use_reentrant=False` gradient checkpointing 与 DDP 的不兼容，通过 `_set_static_graph()` 修复。
- 移除了 negative CFG 路径上的 `torch.no_grad()`，使 Trellis2 与其他 Flow-Factory 模型（Wan2, Z-Image）行为一致——neg 梯度参与训练。
- 清理了 `_forward_dense()` 和 `_forward_sparse()` 中的 `bypass_ddp` 参数。
- 将 `_set_static_graph()` 从通用 `abc.py` 移到 `trellis2_grpo.py` 的 `__init__`，并添加了说明注释。
- 清理了 `grpo.py` 中 pre/post backward 的 debug 日志和 `trellis2_grpo.py` 中的 profiling 计时代码。
- Batch 化 `_inference_shape` 和 `_inference_tex`：将所有 K 个 sample 合并为一个 batched `SparseTensor`，经过单次 denoising loop 后再拆分回 per-sample 结果。

## 调试经验

- `_set_static_graph()` 和 `torch.no_grad()` 解决的是不同问题：前者修 DDP hook 冲突，后者控制梯度语义。二者不可互相替代。
- SparseTensor 的 `from_tensor_list` / `to_tensor_list` / `replace(feats=...)` 和 `.layout` 是 batch 化的关键 API，可安全拆分和重组。
- `CallbackCollector` 中的 `next_latents_mean` 在 sparse 阶段是 SparseTensor，拆分时需要用 `x_t.layout` 取 feats 再按 sample 切分。

## 参考代码

| 文件 | 关键位置 | 说明 |
|------|---------|------|
| `src/flow_factory/models/trellis2/trellis2.py` | `_inference_shape()` L1870-2010, `_inference_tex()` L2011-2156 | Batch 化后的推理入口 |
| `src/flow_factory/models/trellis2/trellis2.py` | `forward()` L1061-1120 | 训练前向，neg CFG 已移除 no_grad |
| `src/flow_factory/models/trellis2/trellis2.py` | `_forward_dense()`, `_forward_sparse()` | 已清理 bypass_ddp |
| `src/flow_factory/models/trellis2/trellis2.py` | `_stage_sde_kwargs()` L395 | 每个 stage 的 dynamics_type / noise_level 配置读取 |
| `src/flow_factory/models/trellis2/flow_match_euler_discrete.py` | `SparseFlowMatchEulerSDEScheduler` | Trellis2 sparse 专用 scheduler，支持 ODE 和 Flow-SDE |
| `src/flow_factory/scheduler/flow_match_euler_discrete.py` | `FlowMatchEulerDiscreteSDEScheduler` | Flow-Factory 通用 scheduler（dense），支持 Flow-SDE / Dance-SDE / CPS / ODE |
| `src/flow_factory/trainers/trellis2_grpo.py` | `Trellis2GRPOTrainer` | Upstream stage sharing + _set_static_graph |
| `src/flow_factory/data_utils/sampler.py` | `GroupContiguousSampler`, `DistributedKRepeatSampler` | 两种 sampler 策略 |
| `examples/grpo/lora/trellis2_shape.yaml` | 全文 | 当前训练配置，group_size=4, per_device_batch_size=2, 7 GPU |

## 最终方案

- **DDP 修复**：在 `Trellis2GRPOTrainer.__init__` 中对所有 DDP-wrapped target module 调用 `_set_static_graph()`。
- **neg CFG 梯度**：移除 `torch.no_grad()`，让 neg 梯度参与训练，与其他模型对齐。
- **Batch 化推理**：`_inference_shape` 和 `_inference_tex` 中，将 K 个 sample 的 coords/noise 合并为一个 batched SparseTensor，一次性走完 denoising loop，再拆分结果。避免了 K 次重复的 scheduler/model 调用开销。

## 下一步任务

### 任务 1：确定最优 K（batch size）

测量 batch 化后不同 `group_size` (K) 下的 GPU 显存峰值和推理速度，找到 K 的上限。

### 任务 2：探索 dynamics_type 替换

当前 shape stage 使用 `Flow-SDE`（`SparseFlowMatchEulerSDEScheduler._step_flow_sde`）。探索是否可以换成 FlowGRPO 通用 scheduler 中的其他 dynamics_type（如 Dance-SDE / CPS），或者是否当前 sparse 实现与 dense 版本存在差异需要对齐。

### 任务 3：探索 sampler_type 替换以支持更大 group_size

当前使用 `GroupContiguousSampler`，要求同一 group 的 K 个 sample 全在同一 rank 上。当 K 增大时，单 rank 显存压力线性增长。探索是否可以用 `DistributedKRepeatSampler`（跨 rank 分散）或混合方案来支持更大 group_size，同时保留 upstream stage sharing 的优化。

## 初步方案

### 任务 1：K 上限测量

- **入口**：`trellis2_grpo.py` 的 `sample()` 方法中 `K = self.training_args.group_size`。
- **改动**：在 `_inference_shape` 和 `_inference_tex` 的 batch 组装前后、denoising loop 各步、result splitting 后分别插入 `torch.cuda.max_memory_allocated()` 和 `time.perf_counter()` 记录。
- **输出**：打印每步的 peak GPU memory (GB) 和 wall time (s)，逐步增大 YAML 中的 `group_size`（4 → 8 → 16 → ...）直到 OOM。
- **风险**：K 增大时 SparseTensor 的 coords 拼接后 N_total 可能导致 attention 的 O(N²) 显存暴增（如果 sparse transformer 用了 dense attention）。需确认 sparse transformer 是否对 batch 维度独立处理。

### 任务 2：dynamics_type 探索

- **对比**：`SparseFlowMatchEulerSDEScheduler._step_flow_sde` 与 `FlowMatchEulerDiscreteSDEScheduler.step()` 的 Flow-SDE 分支数学公式应一致（已在本次 session 确认）。若要用 Dance-SDE 或 CPS，需在 sparse scheduler 中实现 `_step_dance_sde` / `_step_cps`。
- **入口**：`flow_match_euler_discrete.py` 的 `_step_impl` 中添加新分支。
- **参考**：dense 版本在 `src/flow_factory/scheduler/flow_match_euler_discrete.py` L373-405 (Dance-SDE) 和 L400-430 (CPS)。
- **风险**：SparseTensor 不支持 `to_broadcast_tensor`，需将 scalar sigma 保持为 Python float 做算术。

### 任务 3：sampler_type 探索

- **核心约束**：Trellis2 的 upstream stage sharing 要求同一 group 的 K 个 sample 共享同一个 dense/shape 前级结果，这在 `GroupContiguousSampler`（同 rank 连续）下自然满足。
- **方案 A**：保持 `GroupContiguousSampler`，通过任务 1 找到 K 上限即可。
- **方案 B**：使用 `DistributedKRepeatSampler`（跨 rank 分散），但需要将 upstream stage 结果通过 all-gather 或 broadcast 同步到所有持有该 group sample 的 rank。通信开销可能抵消或超过 upstream 共享的收益。
- **方案 C**：混合方案——每个 rank 上保留 per_device_batch_size 个 sample 做 upstream sharing，但 advantage 计算时跨 rank all-reduce 来获得更大 group 的统计量（有效 group_size = K × num_replicas 的子集）。需修改 `advantage_processor.py`。
- **建议**：先完成任务 1 确定单 rank K 上限，再决定是否需要任务 3。
