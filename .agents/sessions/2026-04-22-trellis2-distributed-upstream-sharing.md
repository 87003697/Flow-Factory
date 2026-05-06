# Session Handoff: Trellis2 Distributed Upstream Sharing

## 任务目的

解决 Trellis2 GRPO 训练中 `group_contiguous` sampler 导致的严重 rank 负载倾斜（profiling 显示最慢 rank 600s，其他 100s），通过重构 inference + 跨 GPU dense 共享 + 新 sampler 实现负载均衡。

## 执行内容

- 分析 K=4/8/16 的 profiling 数据，确认主瓶颈是 `wait_for_everyone()` 处的 rank 同步等待，而非计算或显存
- 确认 `per_device_batch_size` 控制训练 mini-batch，`group_size` 控制推理 batch，二者已解耦
- 分析增大 K + 减小 unique_sample_num_per_epoch 的 trade-off，推荐 K=8 + M=28
- 提出跨 GPU dense 共享方案，迭代多轮后确定最终架构：
  - inference 层：加 `samples` 参数 + stage-skip + `_run_stage_inference` / `_get_stage_conditioning` helpers
  - trainer 层：owner-broadcast 协议（all-gather uid → owner 选举 → 并行 pilot → per-uid broadcast）
  - sampler 层：新增 `DistributedGroupAlignedSampler`（K 副本跨 rank 打散但保持在同一 global iteration）
- 实现 Plan 1（Trellis2 Inference Unification）：6 个文件的完整改动
- 实现 Plan 2（Refactor Upstream Broadcast）：拆分巨石函数、修复 tex 训练 bug、抽出 `_broadcast_tensor`
- 在 `Trellis2Sample` 上新增 `_STAGE_BROADCAST_FIELDS` / `_STAGE_METADATA_FIELDS` + `copy_stage_metadata_from()`，将 stage 字段知识归还 Sample
- 修复 `copy_stage_from` 覆盖已广播字段的逻辑 bug

## 调试经验

- `inference_with_shared_dense` 的 pilot 逻辑本质上就是"按 uid 去重到 1 个，只跑一次 dense，结果复制给 K 个"——这和跨 GPU owner-broadcast 是同一个协议的两种拓扑
- `DistributedKRepeatSampler` 全局打散后，同一 uid 的 K 个副本可能跨 iteration，不适合做 same-step 的 owner-broadcast
- `_STAGE_COPY_FIELDS` 如果同时包含广播字段和元数据字段，owner rank 上的 `copy_stage_from` 会用 pilot 直接引用覆盖掉已 `.clone()` 的广播结果，导致多个 sample 共享同一个 tensor 对象
- `_elect_uid_owners` 用 "最低 rank" 策略不是严格负载均衡的，groups_per_iter 大时可考虑改 round-robin

## 之前的 profiling 速度数据（方便下一步对比）

**配置**：W=7 GPU (H20), K=4, B=2, M=56, shape 训练, group_contiguous

K=4 profiling（各 rank 第一个 group 的 elapsed）：
- Rank 0: ~100s, Rank 1: ~150s, Rank 2: ~80s ... 差异来自不同 prompt 的 sparse point 数量不同
- Dense 阶段约 1.8s/次
- Shape 阶段约 60-150s/组（取决于 N_total）
- Tex 阶段约 40-100s/组

K=8 profiling：某个 rank 超过 600s 触发 NCCL Timeout（NCCL_TIMEOUT 默认 600s）
K=16 profiling：NCCL_TIMEOUT 调到 1800s 后可完成，但最慢 rank 超 900s

## 参考代码

| 文件 | 关键位置 | 说明 |
|------|---------|------|
| `src/flow_factory/models/trellis2/trellis2.py` | `_get_stage_conditioning()`, `_run_stage_inference()` | 新增的 stage helpers |
| `src/flow_factory/models/trellis2/trellis2.py` | `inference()` 的 `samples` 参数 + stage-skip | 支持预填充 sample 的 stage 跳过 |
| `src/flow_factory/models/trellis2/trellis2.py` | `Trellis2Sample._STAGE_BROADCAST_FIELDS` / `_STAGE_METADATA_FIELDS` | 广播字段 vs 元数据字段的分离 |
| `src/flow_factory/models/trellis2/trellis2.py` | `copy_stage_metadata_from()` | 只拷贝非广播的辅助字段 |
| `src/flow_factory/trainers/trellis2_grpo.py` | `__init__`: `_upstream_stages`, `_inference_stages`, `_batches_to_merge` | 拓扑配置 |
| `src/flow_factory/trainers/trellis2_grpo.py` | `sample()` | 统一主循环，用 `_batches_to_merge` 控制窗口 |
| `src/flow_factory/trainers/trellis2_grpo.py` | `_distributed_upstream_stages()` | 12 行编排器：resolve → stubs → elect → pilots → broadcast |
| `src/flow_factory/trainers/trellis2_grpo.py` | `_broadcast_tensor()`, `_broadcast_upstream_for_uid()` | 跨 GPU tensor 广播原语 + per-uid 广播 |
| `src/flow_factory/data_utils/sampler.py` | `DistributedGroupAlignedSampler` | 新 sampler：K 副本跨 rank 但同 iteration |
| `src/flow_factory/data_utils/sampler_loader.py` | `_SAMPLER_MAP` | sampler 注册表 |
| `src/flow_factory/hparams/data_args.py` | `sampler_type` Literal | 新增 `distributed_group_aligned` 选项 |
| `src/flow_factory/hparams/args.py` | `_align_batch_geometry()` | 新增 `distributed_group_aligned` 约束分支 |

## 最终方案

三层架构：

1. **Adapter 层**（`trellis2.py`）：`inference()` 通过 `samples` 参数 + stage-skip 支持预填充，`_run_stage_inference` 统一 stage 路由。不感知 distributed 逻辑。

2. **Trainer 层**（`trellis2_grpo.py`）：`sample()` 用 `_batches_to_merge` 实现统一主循环（`group_contiguous` 时 `K//bs`，distributed 时 `1`）。`_distributed_upstream_stages()` 编排跨 GPU 共享协议：all-gather uid → owner 选举 → 并行 pilot 计算 → per-uid broadcast → 填充本地 sample。

3. **Sampler 层**（`sampler.py`）：`DistributedGroupAlignedSampler` 保证同 uid 的 K 副本在同一 global iteration 内，使 owner-broadcast 协议无需跨 iteration cache。

退化行为：`group_contiguous` 下所有 uid 副本在同一 rank → owner = self → broadcast 退化为自拷贝 → 行为与修改前完全一致。

## 下一步任务

1. **插入 debug/profiling 代码**验证跨 GPU broadcast 正确性（sparse_coords 一致性、dense_final_latent 无污染、非 owner rank sample 字段完整性）
2. **跑一轮 `group_contiguous` 回归测试**确认重构后行为不变
3. **跑 `distributed_group_aligned` 对比测试**量化负载均衡收益 vs broadcast 开销
4. 和之前的 profiling 数据对比 wall-clock 提速

## 初步方案

- 在 `_broadcast_upstream_for_uid` 的 broadcast 前后加 assertion：owner rank 的 pilot tensor 和 broadcast 后各 rank 收到的 tensor 做 shape/dtype/值 校验
- 在 `_distributed_upstream_stages` 返回前，对所有 local sample 做 `assert s.sparse_coords is not None` + `assert s.sparse_coords[:, 0].unique() == 0` 校验
- 在 `sample()` 的 profiling 分支（`_PROFILE_K`）里加 per-window 和 per-uid 的 elapsed 记录，方便和之前的 per-group 数据对比
- 准备两份 YAML：一份保持 `group_contiguous`（回归），一份改 `distributed_group_aligned`（需要调 `num_processes` 使 `W*B % K == 0`）
- 预期风险：`distributed_group_aligned` 要求 `W*B % K == 0`，当前配置 W=7, B=2, K=4 → 14%4=2 ≠ 0，需要改 W=8 或 K=2
