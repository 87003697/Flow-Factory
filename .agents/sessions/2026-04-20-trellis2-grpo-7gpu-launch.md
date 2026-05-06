# Session Handoff: Trellis2 GRPO 7卡正式训练

## 任务目的

在 debug 数据集上验证全链路跑通后，切换到正式数据集（2296 train / 100 test），7 × H20 GPU 启动 200 epoch 训练。

## 执行内容

### 1. Debug 训练全链路验证（单 GPU）

在 `dataset/trellis2_debug`（8 train + 2 test）上完成 2 epoch 训练，验证 sample → decode → render → reward → optimize 全链路无错误。

### 2. 修复的 Bug（本 session）

| Bug | 根因 | 修复 |
|-----|------|------|
| `pipeline.py` 模型加载 404 | `sparse_structure_decoder` 在 `pipeline.json` 中用 HF repo 路径，代码错误拼接本地 base path | 检查本地 `.json` 是否存在，不存在则直接用原始 HF 路径 |
| optimize 阶段 device 不匹配 | `_forward_sparse` 通过 `pipeline.get_flow_model()` 取到未被 `accelerator.prepare` 管理的原始模型 | 训练 stage 使用 `self.transformer`（prepared PeftModel）；`cond` 显式 `.to(device)` |
| `accelerate` 版本冲突 | `cli.py` 硬编码 `"accelerate"` 解析到系统 Python 3.13 | 不改 cli，通过 `conda activate` 保证 PATH 正确（启动前 `export PATH=/home/zhiyuan_ma/anaconda3/envs/grpo3d_trellis2/bin:$PATH`） |
| eval 缺少 stages 参数 | `evaluate()` 调用 `self.adapter.inference()` 时 `stages=None`，默认只跑 shape stage（无 dense → 无 sparse_coords） | `_extra_eval_inference_kwargs()` 注入 `stages=['dense', 'shape', 'tex']` |
| eval dense stage `torch.stack` 报错 | dataloader 把 `image_cond_512` collate 成 tensor，但 `_inference_dense` 用 `torch.stack()` 期望 list | 兼容处理：`isinstance(image_cond, torch.Tensor)` 判断 |
| eval `torch.randn(generator=list)` | `create_generator_by_prompt` 返回 list of generators，`torch.randn` 只接受单个 | dense 用 `generator[0]`，shape/tex 用 `generator[b]`（per-sample 循环内） |
| eval tex stage `shape_all_latents=None` | `trajectory_indices=None` 禁用 collector → `shape_all_latents=None` → tex decode 断链 | **重构**：新增 `*_final_latent` 字段，与 collector 解耦 |
| eval render `envmap` 参数错误 | `decode_latents` 在 tex latent 缺失时返回纯几何 `Mesh`，`MeshRenderer` 不接受 `envmap` | 根因同上，`*_final_latent` 修复后 tex latent 始终有值，decode 返回 `MeshWithVoxel` |
| 磁盘空间满 | 预处理缓存默认存 `~/.cache`（根分区 454G），2296 条 × 43.6MB/条 = 98G，加 shard 峰值 196G | `cache_dir` 改到 `/data`（2.8T 剩余）；清理 pip 缓存释放 44G |

### 3. 架构重构：`Trellis2Sample` 字段分离

**修改前**：`shape_all_latents` / `tex_all_latents` 同时承担训练 trajectory 存储和 stage 间传递两个职责，由 `TrajectoryCollector` 统一管理。eval 时 collector 禁用导致下游断链。

**修改后**：
- `*_final_latent`（新增）：循环结束后直接从 `x_t.feats` 赋值，**始终有值**，供 decode 和下游 stage 使用
- `*_all_latents`（保留）：仅由 collector 管理，训练时有值、eval 时 None，供 optimize 使用

## 修改文件清单

### 修改的文件（已跟踪）

| 文件 | 改动类型 | 说明 |
|------|---------|------|
| `src/flow_factory/models/trellis2/trellis2.py` | 大幅修改 (+666 行) | `Trellis2Sample` 字段重构、`_forward_sparse` device fix、`enable_gradient_checkpointing`、`_inference_*` generator 兼容、eval 路径修复 |
| `src/flow_factory/models/trellis2/pipeline.py` | 重构 | `from_pretrained` 简化、`_instantiate` 抽取、HF/本地路径兼容 |
| `src/flow_factory/trainers/grpo.py` | 小改 (+5 行) | 清理 debug probe |
| `src/flow_factory/trainers/abc.py` | 小改 | tokenizer 安全获取 |
| `src/flow_factory/trainers/registry.py` | +1 行 | 注册 `trellis2_grpo` |
| `src/flow_factory/rewards/pick_score.py` | -2 行 | 清理 debug probe |
| `src/flow_factory/hparams/training_args.py` | +1 行 | 微调 |
| `src/flow_factory/models/abc.py` | -1 行 | 移除无用行 |

### 新增文件（未跟踪）

| 文件 | 说明 |
|------|------|
| `src/flow_factory/trainers/trellis2_grpo.py` | Trellis2 GRPO Trainer，upstream stage sharing |
| `examples/grpo/lora/trellis2_shape.yaml` | 7 卡训练配置 |

### 删除的文件

| 文件 | 说明 |
|------|------|
| `scripts/test_trellis2_inference.py` | 临时测试脚本 |
| `scripts/test_trellis2_single_step.py` | 临时测试脚本 |

---

## 后续 Session 1: NCCL Timeout + Device Mismatch 复现

来源：`2026-04-20-trellis2-grpo-device-mismatch.md`

### 问题 1: NCCL ALLREDUCE Timeout

- **现象**：7 卡训练 sampling 阶段超过 600s NCCL timeout，进程挂死
- **根因**：`sample()` 中 7 卡各自独立跑 dense→shape→tex→decode→render，不同 prompt 的 mesh 复杂度差异巨大（顶点数从 90 万到 1200 万不等），导致各 rank 完成时间差最大数分钟。所有 group 的时间差累积后超过 NCCL 默认 600s timeout。
- **修复**：在 `Trellis2GRPOTrainer.sample()` 的 `for group_idx` 循环内添加 per-group barrier：

```python
# trellis2_grpo.py sample() L131-133
                samples.extend(sample_batch)
                self.reward_buffer.add_samples(sample_batch)
                self.accelerator.wait_for_everyone()  # per-group barrier
```

- **效果**：每组耗时约 5 分钟，单组时间差远小于 600s，成功消除 timeout

### 问题 2: Optimize 阶段 device mismatch 再次出现

- **现象**：eval + sampling 全部通过后，optimize 阶段所有 7 个 rank 报 `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:X and cpu!`
- **调用链**：`optimize()` → `adapter.forward(**forward_inputs)` → `forward()` L1070 → `_forward_sparse()` L1234 → `flow_model(x=x_t, t=t_tensor, cond=cond, ...)` → `cross_attn.to_kv(context)` → `F.linear(input, self.weight, self.bias)` → `mat1(cuda:X) and mat2(cpu)`
- **根因**：本 session 修复的两个关键改动在用户手动编辑 `trellis2.py` 时被覆盖（丢失）：
  1. `flow_model = self.transformer` → 回退到 `pipeline.get_flow_model()` 返回的原始模型（未被 accelerator.prepare 管理，参数可能在 CPU）
  2. `cond.to(device=device, dtype=torch.float32)` → 只剩 `cond.to(dtype=torch.float32)`（缺少 device 移动）
- **为什么 sampling 阶段不报错**：sampling 全程 `torch.no_grad()`，模型在 `_load_inference_components` 中被 `.to(device)` 加载到 GPU；而 optimize 阶段需要梯度计算，必须通过 accelerator-prepared 的 PeftModel 才能正确处理

---

## 后续 Session 2: DDP "ready twice" + OOM

来源：`2026-04-21-trellis2-grpo-sampling-profile.md`

### 问题 3: CFG 双次 forward 导致 DDP "ready twice" 错误

- **现象**：optimize 阶段 `forward()` 中 CFG 路径做了两次 `_forward_sparse` 调用（positive + negative），DDP 期望每个 forward 只调用一次，报错 `RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one`
- **根因**：gradient checkpointing (`use_reentrant=False`) + DDP wrapper，第二次 forward（negative）触发了 DDP 的重复 reduction 检测
- **修复**：在 `_forward_sparse` 和 `_forward_dense` 中添加 `bypass_ddp: bool` 参数。negative forward 传 `bypass_ddp=True`，内部使用 `self._unwrap(self.transformer)` 绕过 DDP wrapper：

```python
# _forward_sparse() L1226-1229（当前代码）
        if stage == self._training_stage:
            flow_model = self._unwrap(self.transformer) if bypass_ddp else self.transformer
        else:
            flow_model = self.pipeline.get_flow_model(stage, stage_resolution)
```

```python
# forward() L1074-1078 — negative forward 传 bypass_ddp=True
            if apply_cfg:
                pred_neg = self._forward_sparse(
                    t_val, x_t, neg_image_cond, concat_cond=concat_cond,
                    stage=stage, stage_resolution=stage_resolution,
                    bypass_ddp=True,
                )
```

### 问题 4: Sampling 阶段 CUDA OOM

- **现象**：Epoch 1 sampling 阶段 `decode_latents()` → `mesh.fill_holes()` → `cumesh.get_edges()` 触发 CUDA OOM
- **根因**：某些 prompt 生成的 mesh 顶点数极大（>1200 万），`fill_holes()` 的边表计算需要 O(V) 额外显存
- **当前状态**：尚未修复，需要 profiling 定位性能瓶颈和显存峰值

---

## 当前代码状态（截至最新 session）

| 修复项 | 位置 | 当前状态 |
|--------|------|---------|
| `flow_model = self.transformer` (+ `bypass_ddp`) | `_forward_sparse` L1226-1229 | **已修复** |
| `cond.to(device=device, dtype=...)` | `_forward_sparse` L1231 | **已修复** |
| `bypass_ddp` 在 negative forward | `forward()` L1044, L1078 | **已修复** |
| `torch.stack` tensor/list 兼容 | `_inference_dense` L1764 | **已修复** |
| generator list 解包 | `_inference_dense/shape/tex` | **已修复** |
| `*_final_latent` 字段 | `Trellis2Sample` + 三个 `_inference_*` | **已修复** |
| `stages=['dense','shape','tex']` | `_extra_eval_inference_kwargs` L137-138 | **已修复** |
| per-group barrier | `trellis2_grpo.py sample()` L133 | **已修复** |
| OOM (`fill_holes`) | `decode_latents` | **未修复** — 需要 profiling |

## 当前训练状态

- **运行命令**：`conda activate grpo3d_trellis2 && export PATH=/home/zhiyuan_ma/anaconda3/envs/grpo3d_trellis2/bin:$PATH && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 ff-train examples/grpo/lora/trellis2_shape.yaml`
- **Wandb**：`https://wandb.ai/zm2354-ma-the-hong-kong-polytechnic-university/Flow-Factory`
- **配置**：7 × H20, 200 epoch, 56 unique samples/epoch, LoRA rank 64, lr 1e-4, PickScore reward
- **进度**：Epoch 0 eval + sampling + optimize 已通过，Epoch 1 sampling 遇到 OOM

## 下一步

1. 在采样流程关键节点插入 profiling（`torch.cuda.synchronize()` + `time.time()`），定位各子阶段耗时
2. 监控 `decode_latents` 前后的 `torch.cuda.max_memory_allocated()` 定位 OOM 显存峰值
3. 可能的 OOM 缓解：限制 mesh 最大顶点数、在 decode 前 `torch.cuda.empty_cache()`、降低 `render_resolution`
4. 根据 Wandb 训练曲线调参（lr, clip_range, noise_level, group_size 等）
5. 考虑 `gradient_accumulation_steps: auto` 的修复（当前手动设 4）
