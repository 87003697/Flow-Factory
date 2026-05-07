# Session Handoff: Trellis2 GRPO optimize 阶段 device mismatch

## 任务目的

解决 7 卡训练中的 NCCL ALLREDUCE timeout 问题，并推进 Epoch 0 完整训练。

## 执行内容

- **解决 NCCL timeout**：将 `self.accelerator.wait_for_everyone()` 从 `sample()` 末尾移到 `for group_idx` 循环内部（per-group barrier），成功消除了 NCCL timeout。
- **Evaluation 阶段通过**：15/15 eval batches 全部完成，`eval/reward_pickscore_mean=0.7834`，`eval/reward_pickscore_std=0.0537`。
- **Sampling 阶段通过**：8/8 groups 全部完成，耗时约 42 分钟。per-group barrier 有效防止了时间差累积。
- **Reward 计算通过**：PickScore 4/4 batches 完成。
- **Optimize 阶段崩溃**：所有 7 个 rank 在 `grpo.py:261 optimize()` → `trellis2.py:1069 forward()` → `_forward_sparse()` 中报 `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:X and cpu!`。

## 调试经验

- **per-group barrier 有效**：将 `wait_for_everyone()` 放在 sampling 循环内而非循环后，成功防止了 NCCL timeout。sampling 每组耗时约 5 分钟，rank 间单组时间差远小于 600s timeout。
- **device mismatch 是 optimize 阶段新 bug**：sampling 阶段全程 `torch.no_grad()`，走的是 `pipeline.get_flow_model()` 返回的原始模型；optimize 阶段走 `self.transformer`（accelerate prepared PeftModel），但 `_forward_sparse` 中构建的 `cond`（image conditioning tensor）可能仍在 CPU 上。
- **错误调用栈关键路径**：`optimize()` → `adapter.forward()` → `_forward_sparse()` → `flow_model(x=x_t, t=t_tensor, cond=cond, concat_cond=concat_cond)` → `cross_attn` 中 `self.to_kv(context)` 的 `F.linear(input, self.weight, self.bias)` 报错 `mat1(cuda:X) and mat2(cpu)`。这说明 `cond` tensor（cross attention 的 context）在 CPU 上，而 LoRA 权重在 GPU 上。
- **accelerate 环境注意**：必须用 `grpo3d_trellis2` conda 环境运行，PATH 中 `accelerate` 必须来自该环境，否则子进程会使用错误的 Python 版本。

## 参考代码

| 文件 | 关键位置 | 说明 |
|------|---------|------|
| `src/flow_factory/trainers/trellis2_grpo.py` | `sample()` L101-124 | per-group barrier 已添加（本 session 修改） |
| `src/flow_factory/trainers/grpo.py` | `optimize()` L261 | 调用 `self.adapter.forward(**forward_inputs)` 的位置 |
| `src/flow_factory/models/trellis2/trellis2.py` | `forward()` L1069 | 调用 `_forward_sparse` 入口 |
| `src/flow_factory/models/trellis2/trellis2.py` | `_forward_sparse()` L1222 | `flow_model(x=x_t, t=t_tensor, cond=cond, ...)` — cond 可能在 CPU |
| `third_party/TRELLIS.2/trellis2/modules/sparse/attention/modules.py` | L130, L82 | `self.to_kv(context)` — `F.linear` 报错位置 |

## 最终方案

per-group barrier 修改已生效（NCCL timeout 问题已解决），但 optimize 阶段暴露了新的 device mismatch 问题。

## 下一步任务

修复 optimize 阶段的 device mismatch 错误：`_forward_sparse` 中 `cond` tensor 在 CPU 上，需要确保在调用 `flow_model` 前所有输入 tensor 都在正确的 CUDA device 上。

## 初步方案

1. **定位 cond 来源**：在 `_forward_sparse` 中，`cond` 来自 `forward()` 传入的参数。追溯 `optimize()` → `forward()` 的调用链，检查 `forward_inputs` 中 `image_cond` / `neg_image_cond` 的 device——这些是在 `sample()` 阶段预处理并存入 `Trellis2Sample` 的，可能保留了 CPU device。
2. **检查 `_forward_sparse` 中的 `.to(device)` 守卫**：确认 `cond = cond.to(device)` 是否存在且覆盖了所有路径（包括 `concat_cond`）。
3. **检查 `ref_param_device: cuda` 配置**：`training_args.yaml` 中 `ref_param_device: 'cuda'`，确认 reference model 参数也在 GPU 上，不会导致 ref forward 的 device 冲突。
4. **潜在风险**：`SparseTensor` 的 `.feats` 和 `.coords` 可能在不同 device 上；`image_cond` 在 preprocess 阶段可能被放到 CPU 以节省显存，但 optimize 时需要移回 GPU。
5. **参考之前的修复**：session `2026-04-20-trellis2-grpo-7gpu-launch.md` 中记录了类似的 device mismatch bug 和修复方式（`cond` 显式 `.to(device)`），需要检查该修复是否覆盖了 optimize 路径。
