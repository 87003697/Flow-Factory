# Session Handoff: Trellis2 GRPO 采样阶段性能 Profiling

## 任务目的

在 7×H20 GPU 上进行 Trellis2 GRPO shape-stage 训练，目标 200 epoch。当前所有 DDP / device 错误已修复并验证通过，但采样阶段存在两个问题需要解决：
1. **CUDA OOM**：Epoch 1 采样中 `decode_latents` → `mesh.fill_holes()` → `cumesh.get_edges()` 触发 GPU 显存溢出
2. **性能瓶颈未知**：采样每个 group 约 3-6 分钟，8 个 group 总计约 30-50 分钟/epoch，需要 profiling 定位哪个子阶段耗时最大

## 执行历程（跨 4 个 session，按时间顺序）

### Session 1: 全链路验证 + 7 卡启动 (`2026-04-20-trellis2-grpo-7gpu-launch.md`)
1. 在 8 条 debug 数据上完成 2 epoch 单卡验证
2. 修复了 9 个 bug（详见该 handoff 文件的 Bug 表格）
3. 重构 `Trellis2Sample` 字段：分离 `*_final_latent`（decode 用，始终有值）和 `*_all_latents`（optimize 用，仅训练时有值）
4. 切换到正式数据集（2296 train / 100 test），启动 7 卡训练

### Session 2: NCCL Timeout + Device Mismatch 复现 (`2026-04-20-trellis2-grpo-device-mismatch.md`)
1. **修复 NCCL timeout**：在 `trellis2_grpo.py` 的 `sample()` 循环内添加 per-group barrier（`self.accelerator.wait_for_everyone()`），防止 rank 间时间差累积超过 600s
2. **发现 device mismatch 复现**：之前修复的 `self.transformer` + `cond.to(device)` 因用户手动编辑被覆盖丢失

### Session 3: DDP "ready twice" + OOM（本 session 前半段）
1. **恢复 device mismatch 修复**：在 `_forward_sparse` 和 `_forward_dense` 中重新设置 `flow_model = self.transformer`（训练 stage）和 `cond.to(device=device)`
2. **修复 DDP "ready twice"**：CFG 双次 forward（positive + negative）导致 DDP 的 reduction hook 被触发两次。根因是 gradient checkpointing (`use_reentrant=False`) 与 DDP 的交互。修复方案：在 `_forward_sparse` / `_forward_dense` 中添加 `bypass_ddp: bool` 参数，negative forward 时传 `bypass_ddp=True`，内部使用 `self._unwrap(self.transformer)` 绕过 DDP wrapper
3. **运行验证**：DDP 和 device 问题均解决。Epoch 0 eval + sampling + optimize 全部通过。Epoch 1 sampling 在第 7 个 group（75%进度）遇到 CUDA OOM

## 调试经验（重要，下一个 session 避坑）

### DDP 相关
- `self.transformer` 是 `accelerator.prepare()` 包装后的 `PeftModel`（DDP wrapped），**optimize 阶段必须用它**才能正确同步梯度
- 非训练 stage（如 dense 在训 shape 时）使用 `pipeline.get_flow_model()` 返回原始模型
- CFG negative forward 不需要梯度同步，可以安全用 `_unwrap` 绕过 DDP
- gradient checkpointing 的 `use_reentrant=False` 是 DDP "ready twice" 的根源之一

### 环境相关
- `accelerate` 版本冲突：系统 Python 3.13 有一个 `accelerate`，会与 conda 环境的冲突。必须在启动前设 `export PATH=/home/zhiyuan_ma/anaconda3/envs/grpo3d_trellis2/bin:$PATH`
- 启动命令：
  ```bash
  bash -c 'source /home/zhiyuan_ma/anaconda3/etc/profile.d/conda.sh && \
    conda activate grpo3d_trellis2 && \
    export PATH=/home/zhiyuan_ma/anaconda3/envs/grpo3d_trellis2/bin:$PATH && \
    cd /home/zhiyuan_ma/code/Flow-Factory && \
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 ff-train examples/grpo/lora/trellis2_shape.yaml 2>&1'
  ```

### 采样流程耗时观察（来自运行日志）
- Epoch 0 sampling 8 groups：总计约 42 分钟，每 group 约 3-6 分钟不等
- Epoch 1 sampling 在第 7 个 group 崩溃前已运行约 31 分钟
- 各 rank 的 coords 数量差异巨大：最少 279 coords，最多 30693 coords，说明 mesh 复杂度差异是耗时和显存的主要变量

### OOM 具体信息
- **崩溃位置**：Epoch 1 Sampling 第 7 个 group（75% 进度）
- **调用链**：`sample()` → `_inference_fn()` → `inference_with_shared_dense()` L1557-1563 decode 循环 → `render_latents()` L2383 → `decode_latents()` L2312 → `mesh.fill_holes()` → `cumesh.get_edges()` → CUDA OOM
- **报错 rank**：rank 0 和其他 rank 同时报错（因为分布式通信）
- **错误信息**：
  ```
  RuntimeError: [CuMesh] CUDA error:
      File:       /tmp/extensions/CuMesh/src/utils.h
      Line:       42
      Error code: 2
      Error text: out of memory
  ```

## 参考代码

### 采样主流程（每个 epoch 执行一次）

```
GRPOTrainer.start()                        # grpo.py L60-91
  └─ samples = self.sample()               # grpo.py L85
       └─ Trellis2GRPOTrainer.sample()     # trellis2_grpo.py L81-135
            │   K=4 (group_size), bs=2, batches_per_group=2, num_groups=8
            │   每 group 合并 2 个 batch → 4 个 sample
            └─ for group_idx in range(8):
                 merged_batch = _merge_batches(...)
                 sample_batch = self._inference_fn(**sample_kwargs)
                 │   _inference_fn = adapter.inference_with_shared_dense (因为 train shape)
                 │
                 └─ inference_with_shared_dense()    # trellis2.py L1437-1565
                      ├─ _inference_dense(pilot)     # L1489-1503  B=1, 12 步 ODE
                      │   └─ for i in range(12): _forward_dense() × 2 (pos+neg CFG)
                      │   └─ _decode_dense_to_coords()
                      │
                      ├─ 复制 pilot 输出到 K=4 个 sample  # L1505-1519
                      │
                      ├─ _inference_shape(samples)   # L1524-1537  B=1×4（逐 sample 循环）
                      │   └─ for b in range(4):
                      │       for step in range(12): _forward_sparse() × 2 (pos+neg CFG)
                      │
                      ├─ _inference_tex(samples)     # L1542-1554  B=1×4（逐 sample 循环）
                      │   └─ for b in range(4):
                      │       for step in range(12): _forward_sparse() × 2 (pos+neg CFG)
                      │
                      └─ if decode_output:            # L1557-1563  ← OOM 在此
                           envmap = _build_envmap()
                           for s in samples:           # 4 个 sample 逐个 decode+render
                             render_latents(s)
                             ├─ decode_latents(s)      # trellis2.py L2268-2325
                             │   ├─ decode_shape()     # L2300 → shape_decoder(slat)
                             │   ├─ decode_texture()   # L2306 → tex_decoder(slat, subs)
                             │   ├─ mesh.fill_holes()  # L2312 ← OOM 触发点
                             │   └─ MeshWithVoxel(...)  # L2314-2323
                             │
                             ├─ mesh.simplify(16M)     # L2387
                             └─ render_frames(mesh)    # L2401-2407 → multiview 渲染

                 self.accelerator.wait_for_everyone()   # per-group barrier
```

### 关键文件一览

| 文件 | 关键函数/行号 | 说明 |
|------|-------------|------|
| `src/flow_factory/trainers/grpo.py` | `start()` L60-91 | 主循环：`sample()` → `prepare_feedback()` → `optimize()` |
| `src/flow_factory/trainers/trellis2_grpo.py` | `sample()` L81-135 | 采样实现：合并 batch → 调用 `_inference_fn` → per-group barrier |
| `src/flow_factory/models/trellis2/trellis2.py` | `inference_with_shared_dense()` L1437-1565 | shape 训练的采样入口（dense pilot → shape K → tex K → decode/render） |
| 同上 | `_inference_dense()` L1744-1870 | dense stage ODE 采样，12 步，B=1 |
| 同上 | `_inference_shape()` L1871-1994 | shape stage Flow-SDE 采样，12 步，逐 sample 循环 |
| 同上 | `_inference_tex()` L1996-2140 | tex stage ODE 采样，12 步，逐 sample 循环 |
| 同上 | `decode_latents()` L2268-2325 | 解码 shape+tex latents → `MeshWithVoxel` |
| 同上 | `decode_shape()` L2179-2225 | shape latent → mesh（调用 `shape_decoder`） |
| 同上 | `decode_texture()` L2227-2266 | tex latent → tex voxels（调用 `tex_decoder`） |
| 同上 | `render_latents()` L2351-2415 | mesh → multiview frames（24 帧 512×512） |
| 同上 | `forward()` L1020-1098 | optimize 阶段的 forward，CFG + `bypass_ddp` |
| 同上 | `_forward_sparse()` L1210-1234 | sparse stage forward，含 `bypass_ddp` + device 修复 |
| 同上 | `_forward_dense()` L1102-1119 | dense stage forward，含 `bypass_ddp` + device 修复 |
| `third_party/TRELLIS.2/trellis2/representations/mesh/base.py` | `fill_holes()` L41 | 调用 `cumesh.get_edges()` ← OOM 触发点 |
| `examples/grpo/lora/trellis2_shape.yaml` | 全文件 | 训练配置：7卡, group_size=4, bs=2, 12 步, LoRA rank 64, PickScore reward |

### 当前代码中已修复项的确切位置

```python
# _forward_sparse() L1225-1228
if stage == self._training_stage:
    flow_model = self._unwrap(self.transformer) if bypass_ddp else self.transformer
else:
    flow_model = self.pipeline.get_flow_model(stage, stage_resolution)

# _forward_sparse() L1230
cond = cond.to(device=device, dtype=torch.float32)

# forward() L1043 — dense negative CFG
pred_neg = self._forward_dense(t_val, latents, neg_image_cond, bypass_ddp=True)

# forward() L1074-1078 — sparse negative CFG
pred_neg = self._forward_sparse(
    t_val, x_t, neg_image_cond, concat_cond=concat_cond,
    stage=stage, stage_resolution=stage_resolution,
    bypass_ddp=True,
)

# trellis2_grpo.py sample() L133 — per-group barrier
self.accelerator.wait_for_everyone()
```

## 当前训练配置摘要（`trellis2_shape.yaml`）

| 参数 | 值 | 说明 |
|------|-----|------|
| GPU | 7 × H20 (96GB each) | `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6` |
| target_flow_model | `shape_slat_1024` | 训练 shape stage |
| group_size | 4 | GRPO 每组 4 个 sample |
| per_device_batch_size | 2 | 每设备 batch size |
| unique_sample_num_per_epoch | 56 | 7 GPU × 8 unique prompts/GPU |
| num_batches_per_epoch | 16 | 56/7×2 = 16 batches per device per epoch |
| num_groups | 8 | 16/2 = 8 groups (因 batches_per_group=2) |
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
| reward | PickScore | 图像质量奖励 |
| Wandb | `Flow-Factory` project | 日志记录 |

## 下一步任务

### 任务 1: Profiling 采样各子阶段耗时

在以下 8 个节点插入 `torch.cuda.synchronize()` + `time.time()` 计时，只在 rank 0 打印：

1. **整个 group 耗时** — `trellis2_grpo.py` 的 `sample()` 循环内
2. **`_inference_dense`** — 1 次 pilot 采样 (12 步 ODE × 2 CFG forward)
3. **`_inference_shape`** — 4 个 sample 的 shape 采样 (4×12 步 SDE × 2 CFG forward)
4. **`_inference_tex`** — 4 个 sample 的 tex 采样 (4×12 步 ODE × 2 CFG forward)
5. **`decode_shape`** — shape decoder 推理
6. **`decode_texture`** — tex decoder 推理
7. **`mesh.fill_holes()`** — mesh 后处理（OOM 点）
8. **`render_frames`** — 24 帧 multiview 渲染

### 任务 2: 显存监控

在 `decode_latents` 前后记录 `torch.cuda.max_memory_allocated()` 和 `torch.cuda.memory_allocated()`，特别关注 `fill_holes()` 前后的显存变化。

### 任务 3: 修复 OOM（基于 profiling 结果）

可能的方向：
- 在 `decode_latents` 前调用 `torch.cuda.empty_cache()` 释放中间激活
- 限制 mesh 顶点数上限（`mesh.simplify()` 在 `fill_holes` 之前执行）
- 将 `fill_holes` 移到 CPU（如果 cumesh 支持）
- 降低 `render_resolution`（当前 512）
- 对 `decode_latents` 做 try-except 保护，OOM 时跳过该 sample 的 decode/render

## 初步方案

1. 在 `inference_with_shared_dense()` L1489/L1524/L1542/L1557 前后各插入 `torch.cuda.synchronize(); t = time.time()` 计时点
2. 在 `decode_latents()` L2300/L2306/L2312/L2314 前后插入细粒度计时 + `torch.cuda.memory_allocated()` 记录
3. 在 `render_latents()` L2383/L2387/L2401 前后插入计时
4. 所有计时结果用 `logger.info` 打印（仅 rank 0），格式：`[PROFILE] {stage}: {elapsed:.2f}s, GPU mem: {mem_mb:.0f}MB`
5. 运行 1-2 个 epoch 收集数据，根据结果决定优化策略
6. **同时尝试在 `decode_latents` 入口添加 `torch.cuda.empty_cache()` 作为 OOM 缓解的第一步尝试**
