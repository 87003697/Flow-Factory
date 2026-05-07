# Session Handoff: Trellis2 上游共享采样 + 单 Stage 训练选择

## 任务目的

为 Trellis2 实现单 stage 训练选择（dense/shape/tex）和上游共享采样——同 prompt 的 K 个 sample 共享上游 stage 的输出，避免重复计算。

## 执行内容

- 讨论并解决了 `per_device_batch_size` 同时控制 sampling 和 optimize batch 大小的耦合问题（`per_device_batch_size=2, group_size=16` 时 `inference()` 只看到 2 个 sample，无法实现全 K 共享）。
- 最终选定方案：创建 `Trellis2GRPOTrainer` 继承 `GRPOTrainer`，在 `sample()` 中累积 `K//bs` 个连续 batch 后一次调用共享推理方法；在 adapter 上新增穷举方法 `inference_with_shared_dense()` / `inference_with_shared_dense_shape()` 而非修改 `inference()`。
- `pipeline.py` 添加 `'dense': 'sparse_structure_flow_model'` 到 `TARGET_MODEL_MAP`。
- `trellis2.py` 的 `load_scheduler()` 改为根据 `target_flow_model` 动态返回对应 stage 的 scheduler。
- `trellis2.py` 新增 `inference_with_shared_dense()` 和 `inference_with_shared_dense_shape()` 两个穷举方法。
- 新建 `trainers/trellis2_grpo.py`：`Trellis2GRPOTrainer` 继承 `GRPOTrainer`，重写 `sample()`，`__init__` 中根据 `train_stage` 选定 `_inference_fn`。
- 在 `registry.py` 和 `training_args.py` 注册 `trellis2_grpo`。
- 回归测试（forward_bs、train_smoke、rollout_replay、inference_bs）全部 PASS。
- 新建 `test_trellis2_upstream_share.py`：3 个 gate 全 PASS（shape 共享 dense bit-identical；tex 共享 dense+shape bit-identical；dense 无共享全异）。
- 扩展 `test_trellis2_train_smoke.py` 新增 Gate C 验证 `target_flow_model=dense` 梯度流健康（300 LoRA params, max |grad| = 1.89e-01）。
- 新建 `test_trellis2_grpo_dryrun.py`：trainer plumbing 正确调度 + rollout-replay ratio = 1.000000。

## 调试经验

- `inference_with_shared_dense()` 直接调用 `_inference_dense()` 时，flow model 可能在 CPU 上——需确保 test 脚本中显式 `.to(adapter.device)` 所有 flow model 和 decoder。
- `preprocess_func` 的 `images` 参数是 `List[List[Image]]`（外层是 batch，内层是同一 sample 的多视角），不是 `List[Image]`。
- `all_latents` 只存储 `trajectory_indices` 选中的步骤，需通过 `latent_index_map` 映射 step_idx → 存储位置，不能直接用 step_idx 索引。

## 参考代码

| 文件 | 关键位置 | 说明 |
|------|---------|------|
| `src/flow_factory/trainers/trellis2_grpo.py` | `Trellis2GRPOTrainer` | 新 trainer：`__init__` 选定 `_inference_fn`，`sample()` 累积 batch + 调用共享推理 |
| `src/flow_factory/models/trellis2/trellis2.py` L1367-1600 | `inference_with_shared_dense()` / `inference_with_shared_dense_shape()` | 穷举的两个共享推理方法 |
| `src/flow_factory/models/trellis2/trellis2.py` L405-410 | `load_scheduler()` | 根据 train_stage 动态返回 scheduler |
| `src/flow_factory/models/trellis2/pipeline.py` L90-96 | `TARGET_MODEL_MAP` | 包含 dense 映射 |
| `src/flow_factory/trainers/grpo.py` L141-170 | `GRPOTrainer.sample()` | 原始 sample() 逻辑，新 trainer 的 base |
| `src/flow_factory/trainers/grpo.py` L182-340 | `GRPOTrainer.optimize()` | 继承不改，继续用 `per_device_batch_size` 切批 |
| `src/flow_factory/data_utils/sampler.py` L81-145 | `GroupContiguousSampler` | 保证组内 K 条连续，batch 累积的前提 |
| `src/flow_factory/data_utils/image_3D_dataset.py` | `Image3DDataset` | Trellis2 用的 dataset，RGBA 保留 |
| `src/flow_factory/rewards/abc.py` | `BaseRewardModel` | 3D reward 需要继承的基类 |

## 最终方案

**穷举 + 新 Trainer** 方案：

- 不修改 `inference()`, `grpo.py`, `training_args.py`（字段）, `args.py`, `sampler_loader.py` 等共享基础设施。
- 在 adapter 上新增两个穷举方法（`inference_with_shared_dense` / `inference_with_shared_dense_shape`），直接调用已有的 `_inference_dense/shape/tex` 私有方法，pilot B=1 → repeat K → 训练 stage B=K。
- 新建 `Trellis2GRPOTrainer` 继承 `GRPOTrainer`，只重写 `sample()`：累积 `K//bs` 个 batch 后调用对应的共享推理方法。`optimize()` 完全继承。
- YAML 中 `trainer_type: trellis2_grpo` 即可启用。

选择理由：零侵入共享代码；穷举比泛化（`_STAGE_ORDER` + index 比较）简单直观；新 trainer 隔离了 batch 累积逻辑，不影响其他模型。

## 下一步任务

正式开始 Trellis2 GRPO 训练。

## 初步方案

1. **准备训练 YAML**：参考 `examples/grpo/lora/flux1.yaml` 的结构，创建 `examples/grpo/lora/trellis2_shape.yaml`，关键字段：
   - `trainer_type: trellis2_grpo`
   - `model_type: trellis2`
   - `target_flow_model: shape_slat_1024`
   - `sampler_type: group_contiguous`（上游共享的前提）
   - `dataset_type: image_3d`
   - SDE 配置：shape_sde 用 Flow-SDE，dense/tex 用 ODE

2. **准备数据集**：确认 `Image3DDataset` 格式要求（图片路径 + prompt），准备或复用已有的 image-to-3D 数据。

3. **准备 3D Reward Model**：目前 `src/flow_factory/rewards/` 中没有 3D reward。需要实现至少一个，可能的选项：
   - VLM-based reward（用 VLM 对渲染的多视角图打分）
   - CLIP-based reward（对渲染图 vs 条件图算 CLIP 相似度）
   - 简单的 dummy reward 先跑通流程

4. **DeepSpeed 配置**：Trellis2 模型较大（4B），需要 DeepSpeed ZeRO-2 或 ZeRO-3 + gradient checkpointing。确认 `config/deepspeed/` 下有合适的配置。

5. **首次小规模试跑**：`max_epochs=2, unique_sample_num_per_epoch=4, group_size=4, per_device_batch_size=2`，确认端到端流程无报错，loss 数值合理，然后逐步放大。
