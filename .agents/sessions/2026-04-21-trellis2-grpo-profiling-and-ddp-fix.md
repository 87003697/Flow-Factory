# Session Handoff: Trellis2 GRPO 采样 Profiling + DDP Ready-Twice 修复

## 任务目的

1. 在采样流程 8 个关键节点插入 profiling 计时和显存监控，定位性能瓶颈
2. 通过 `torch.cuda.empty_cache()` 缓解 decode 阶段 CUDA OOM
3. 修复 optimize 阶段 DDP "ready twice" 崩溃

## 执行内容

1. 在 `trellis2_grpo.py` sample() 的 group 循环内添加 per-group 计时 + 显存报告
2. 在 `trellis2.py` inference_with_shared_dense() 中为 dense/shape/tex/decode+render 四阶段添加计时
3. 在 `decode_latents()` 中添加 decode_shape/decode_texture/fill_holes 的细粒度计时 + `torch.cuda.memory_allocated()` / `max_memory_allocated()` 监控
4. 在 `render_latents()` 中添加 decode/simplify/render_frames 三步计时
5. 在 `decode_latents()` 入口和 decode 循环入口各添加 `torch.cuda.empty_cache()` 缓解 OOM
6. **OOM 已缓解**：加 `empty_cache()` 后 8 个 group 全部完成，fill_holes 显存 delta 仅 +1~3MB
7. 发现并修复 optimize 阶段 DDP "ready twice" 错误：在 `forward()` 中对 CFG negative forward 包裹 `torch.no_grad()`

## 调试经验

### DDP "ready twice" 根因分析

- **之前的修复**（Session 3 的 `bypass_ddp`）只在 sampling（无梯度）阶段有效，因为 sampling 在 `torch.no_grad()` 下运行没有 backward pass
- **optimize 阶段的问题**：`forward()` 中 CFG 做两次 forward（positive + negative），positive 通过 DDP-wrapped `self.transformer`，negative 通过 `self._unwrap(self.transformer)`（bypass_ddp=True）。虽然绕过了 DDP wrapper，但**两者共享同一组底层参数**。DDP 的 reduction hook 注册在参数的 `AccumulateGrad` 节点上，而非模块上。backward 时梯度同时从 positive 和 negative 两条路径流到同一参数 → hook 触发两次
- **修复**：对 negative forward 包裹 `torch.no_grad()`，切断 autograd graph。语义上也正确：GRPO 策略梯度只需要 ∇θ pred_pos，negative prediction 只作 CFG 基线不需要梯度回传
- **注意**：`bypass_ddp=True` 仍然保留——它在 sampling 阶段避免 DDP forward hook 的副作用

### Profiling 结论

采样一个 epoch（8 groups）总耗时约 42 分钟。显存远未占满，真正的瓶颈是 shape/tex 推理的 B=1 逐 sample 循环。

## Profiling 数据（Epoch 0, Rank 0, 7×H20 96GB）

### 各 Group 耗时

| Group | 耗时 | 说明 |
|-------|------|------|
| 0 | 180.92s | |
| 1 | 226.81s | rank 0 自身只花 ~46s，其余时间在 barrier 等其他 rank |
| 2 | 341.70s | |
| 3 | 444.42s | 最慢 group |
| 4 | 312.13s | |
| 5 | 365.87s | |
| 6 | 442.92s | |
| 7 | 189.65s | |

**平均**: 313s/group, **总计**: 41 分 44 秒

### 各阶段耗时分解（Rank 0, Group 0 — 典型）

| 阶段 | 耗时 | 占比 | 说明 |
|------|------|------|------|
| `_inference_dense` | 4.07s | 2.3% | 1 次 pilot, 12 步 ODE × 2 CFG |
| `_inference_shape` | **100.66s** | **55.7%** | 4 sample × 12 步 SDE × 2 CFG = 96 forwards（**B=1 逐 sample 循环**） |
| `_inference_tex` | 55.48s | 30.7% | 4 sample × 12 步 ODE × 2 CFG = 96 forwards（**B=1 逐 sample 循环**） |
| decode+render | 18.38s | 10.2% | 4 sample 逐个 decode+render |
| 空闲等 barrier | ~2.3s | 1.1% | `wait_for_everyone()` |

### 显存使用

| 指标 | 值 | 占 96GB |
|------|-----|---------|
| 采样稳态 | ~17.6GB | 18% |
| Rank 0 peak | 37.4GB | 39% |
| 全局最高 peak（Rank 6） | 45.4GB | 47% |
| decode 时 fill_holes delta | +1~3MB | 可忽略 |

### 性能瓶颈总结

1. **主瓶颈**：shape/tex 推理是 B=1 逐 sample 循环（`for b, sample in enumerate(samples):`），每个 sample 独立做 12 步 × 2 CFG = 24 次 forward。4 个 sample 就是 96 次。如果 batch 起来只需 24 次
2. **负载不均**：各 rank 的 sparse coords 数量差异巨大（279~30693），导致推理时间差数倍。最快的 rank 在 barrier 白白等
3. **显存大量空闲**：稳态仅占 18%，peak 仅占 39-47%，有充足空间做 batch 化

## 参考代码

| 文件 | 关键位置 | 说明 |
|------|---------|------|
| `src/flow_factory/models/trellis2/trellis2.py` | `forward()` L1061-1120 | optimize 的单步 forward，包含 CFG 双次 forward + `torch.no_grad()` 修复 |
| 同上 | `_forward_sparse()` L1225-1270 | sparse stage forward，含 `bypass_ddp` + device 修复 |
| 同上 | `_forward_dense()` L1102-1155 | dense stage forward，含 `bypass_ddp` + device 修复 |
| 同上 | `_inference_shape()` L1944-2070 | shape 推理，**B=1 逐 sample 循环**（L1971） |
| 同上 | `_inference_tex()` L2072-2210 | tex 推理，B=1 逐 sample 循环 |
| 同上 | `inference_with_shared_dense()` L1459-1635 | 采样入口（dense pilot → shape K → tex K → decode/render），含四阶段 profiling |
| 同上 | `decode_latents()` L2338-2446 | mesh 解码，含细粒度 profiling + empty_cache |
| 同上 | `render_latents()` L2471-2570 | mesh 渲染，含 decode/simplify/render_frames 计时 |
| `src/flow_factory/trainers/trellis2_grpo.py` | `sample()` L81-170 | 采样循环，含 per-group 计时 |
| `src/flow_factory/trainers/grpo.py` | `optimize()` L~280-340 | 训练循环，`backward(loss)` 是 DDP ready-twice 的触发点 |
| `examples/grpo/lora/trellis2_shape.yaml` | 全文件 | 训练配置 |

### 当前代码中已修复项的确切位置

```python
# forward() L1066-1067 — dense negative CFG，torch.no_grad() 修复
if apply_cfg:
    with torch.no_grad():
        pred_neg = self._forward_dense(t_val, latents, neg_image_cond, bypass_ddp=True)

# forward() L1097-1103 — sparse negative CFG，torch.no_grad() 修复
if apply_cfg:
    with torch.no_grad():
        pred_neg = self._forward_sparse(
            t_val, x_t, neg_image_cond, concat_cond=concat_cond,
            stage=stage, stage_resolution=stage_resolution,
            bypass_ddp=True,
        )

# _forward_sparse() L1225-1228 — bypass_ddp + device 修复（仍然保留）
if stage == self._training_stage:
    flow_model = self._unwrap(self.transformer) if bypass_ddp else self.transformer
else:
    flow_model = self.pipeline.get_flow_model(stage, stage_resolution)

# trellis2_grpo.py sample() L133 — per-group barrier
self.accelerator.wait_for_everyone()

# decode_latents() 入口 — empty_cache
torch.cuda.empty_cache()

# inference_with_shared_dense() decode 循环入口 — empty_cache
torch.cuda.empty_cache()
```

## 最终方案

1. **Profiling 插桩**：在两个文件的 8 个关键节点插入 `torch.cuda.synchronize()` + `time.time()` 计时 + `torch.cuda.memory_allocated()`/`max_memory_allocated()` 显存监控，所有输出通过 `logger.info("[PROFILE] ...")` 打印
2. **OOM 缓解**：在 `decode_latents()` 入口和 decode 循环入口各调用 `torch.cuda.empty_cache()`
3. **DDP 修复**：在 `forward()` 中对 CFG negative forward 包裹 `torch.no_grad()`，切断反向传播时的 autograd graph，阻止 DDP reduction hook 重复触发

## 下一步任务

### 任务 1：修复 DDP "ready twice"（验证 `torch.no_grad()` 修复）

当前 `torch.no_grad()` 修复已添加（L1066 和 L1098），但 `logs/trellis2_grpo_profile_run3.log` 仍然在 optimize 阶段崩溃。这说明：
- **可能原因 A**：run3 的日志可能是在添加 `torch.no_grad()` 之前运行的（检查时间线确认）
- **可能原因 B**：`torch.no_grad()` 不够，gradient checkpointing（`use_reentrant=False`）导致 recompute 时仍然注册 hook。可能需要改用 `_set_static_graph()` 或其他方案

需要重新运行训练，确认修复是否生效。如果仍然崩溃，备选方案：
1. 在 DDP model 上调用 `self.transformer._set_static_graph()` — 告知 DDP 计算图不变
2. 改用 `find_unused_parameters=True` — 允许部分参数在某些 forward 中不被使用
3. 将 gradient checkpointing 切换为 `use_reentrant=True` — 避免 non-reentrant 模式与 DDP 的交互问题

### 任务 2：采样阶段 batch 化加速

将 `_inference_shape()` 和 `_inference_tex()` 从 B=1 逐 sample 循环改为 batch forward，核心改动：

**入口文件**：`src/flow_factory/models/trellis2/trellis2.py`

**关键改动点**：
- `_inference_shape()` L1971 的 `for b, sample in enumerate(samples):` → 将所有 sample 的 SparseTensor concat 成一个 batch，一次 forward（12 步 × 2 CFG = 24 次，而非 4 × 24 = 96 次）
- `_inference_tex()` 同理
- `_forward_sparse()` 需要支持 B>1 的 SparseTensor 输入（当前 `_inference_shape` 传 B=1，但 `forward()` 在 optimize 阶段已经支持 B>1）
- 注意 log_prob 收集逻辑需要按 batch dim 拆分回各 sample

**潜在风险**：
- 显存从 ~17GB 增加到估计 30-50GB（仍然在 96GB 的安全范围内）
- SparseTensor 的 batch concat 需要正确设置 coords 的 batch index（第 0 列）
- SDE 的随机噪声生成需要每个 sample 独立的 generator

**预期收益**：
- shape + tex 推理从 96 次 forward 降到 24 次，理论加速 ~4x
- 每 group 从 ~300s 降到估计 ~100-120s
- 每 epoch 采样从 ~42 分钟降到估计 ~15-18 分钟

## 当前训练配置摘要（`trellis2_shape.yaml`）

| 参数 | 值 | 说明 |
|------|-----|------|
| GPU | 7 × H20 (96GB each) | `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6` |
| target_flow_model | `shape_slat_1024` | 训练 shape stage |
| group_size | 4 | GRPO 每组 4 个 sample |
| per_device_batch_size | 2 | 每设备 batch size |
| unique_sample_num_per_epoch | 56 | 7 GPU × 8 unique prompts/GPU |
| num_batches_per_epoch | 16 | 56/7×2 = 16 batches per device per epoch |
| num_groups | 8 | 16/2 = 8 groups |
| num_inference_steps | 12 | 每 stage 12 步采样 |
| guidance_scale | 7.5 | CFG guidance |
| decode_output | true | 采样后 decode mesh 并渲染 multiview |
| render_num_frames | 24 | 渲染 24 帧 |
| render_resolution | 512 | 渲染分辨率 |
| shape_sde | Flow-SDE, noise_level=0.7 | 训练 stage 用随机 SDE |
| dense_sde / tex_sde | ODE | 非训练 stage 用确定性 ODE |
| gradient_checkpointing | true | 显存优化 |
| lora_rank | 64, alpha=128 | LoRA 配置 |
| learning_rate | 1e-4 | 学习率 |
| reward | PickScore (mean=0.7835) | 图像质量奖励 |

## 环境相关

- 启动命令：
  ```bash
  bash -c 'source /home/zhiyuan_ma/anaconda3/etc/profile.d/conda.sh && \
    conda activate grpo3d_trellis2 && \
    export PATH=/home/zhiyuan_ma/anaconda3/envs/grpo3d_trellis2/bin:$PATH && \
    cd /home/zhiyuan_ma/code/Flow-Factory && \
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 ff-train examples/grpo/lora/trellis2_shape.yaml 2>&1'
  ```
- Profiling 代码仍在文件中（带 `# #region agent log` / `# #endregion` 标记），下个 session 完成 batch 化后可清理
