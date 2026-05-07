# Session Handoff: Trellis2 GRPO 训练配置 + PickScore Reward 对接

## 任务目的

为 Trellis2 GRPO 训练创建完整的 YAML 配置，并打通 decode→render→PickScore reward 的数据流，使训练流程端到端可运行。

## 执行内容

- 修改 `Trellis2GRPOTrainer.__init__`：从 `model_args.extra_kwargs` 读取 `decode_output`、`render_num_frames`、`render_resolution`、`envmap_path` 存为 `self._render_kwargs`。
- 修改 `Trellis2GRPOTrainer.sample()`：在 `sample_kwargs` 中注入 `**self._render_kwargs`，使推理时 `decode_output=True` 传入 inference，触发 `render_latents()` 填充 `sample.video`。
- 在 `GRPOTrainer` 中新增 hook `_extra_eval_inference_kwargs()` 返回 `{}`，`evaluate()` 的 `inference_kwargs` 构建处调用此 hook。
- `Trellis2GRPOTrainer` 覆盖 `_extra_eval_inference_kwargs()` 返回 `self._render_kwargs`，使 eval 也能 decode+render。
- 新建 `examples/grpo/lora/trellis2_shape.yaml`：完整训练配置（trellis2_grpo + shape_slat_1024 + PickScore reward + per-stage SDE/ODE）。
- 验证了 PickScore 已原生支持 video 输入（`_compute_video_scores` 展平多帧 → 批量计算 CLIP 相似度 → mean pool），无需新建 reward model。
- 验证了 `RewardProcessor._convert_media_format` 可将 `List[Tensor(T,C,H,W)]` → `List[List[PIL.Image]]` 供 PickScore 消费。

## 调试经验

- `GRPOTrainingArguments` / `EvaluationArguments` 都没有 `decode_output` 字段，所以 `**self.training_args` / `**self.eval_args` 不会自动传递渲染参数。必须从 `model_args.extra_kwargs`（YAML `model:` 段的非标准字段）单独读取。
- per-stage SDE 覆盖（`dense_sde`/`shape_sde`/`tex_sde`）放在 YAML 的 `model:` 段而非 `scheduler:` 段，因为 `_stage_sde_kwargs()` 从 `model_args.extra_kwargs` 读取。
- `PickScoreRewardModel.__call__` 在 `image` 和 `video` 同时非 None 时会 raise。`Trellis2Sample.image` 为 None（3D 不产出 2D image），`_compute_pointwise_batch` 的 `all(getattr(s, k) is not None ...)` 过滤会自动排除 `image` 键，只传 `video`，不会触发冲突。

## 参考代码

| 文件 | 关键位置 | 说明 |
|------|---------|------|
| `src/flow_factory/trainers/trellis2_grpo.py` | `__init__` L67-73, `sample()` L113, `_extra_eval_inference_kwargs` L123 | render_kwargs 读取、注入、eval hook |
| `src/flow_factory/trainers/grpo.py` | `_extra_eval_inference_kwargs` L94-96, `evaluate()` L120 | 新增 hook 定义与调用 |
| `src/flow_factory/models/trellis2/trellis2.py` | `render_latents()` L2296-2358 | decode latents → mesh → multiview frames → sample.video |
| `src/flow_factory/models/trellis2/trellis2.py` | `_stage_sde_kwargs()` L357-364 | per-stage SDE 配置消费入口 |
| `src/flow_factory/rewards/pick_score.py` | `_compute_video_scores()` L69-99 | 多帧 PickScore 打分逻辑 |
| `src/flow_factory/rewards/reward_processor.py` | `_compute_pointwise_batch` L146-160, `_convert_media_format` L116-143 | sample → reward 字段提取与格式转换 |
| `examples/grpo/lora/trellis2_shape.yaml` | 全文 | 完整训练 YAML |

## 最终方案

- 渲染参数通过 `model_args.extra_kwargs` 传递（YAML `model:` 段），在 trainer `__init__` 中一次读取，sample / eval 两处复用。
- eval 流程用 `_extra_eval_inference_kwargs()` hook 扩展，避免在子类中复制整个 `evaluate()` 方法。
- 直接复用现有 PickScore 对 multiview video 打分，不新建 reward model。

## 数据准备（已完成）

- 数据集 `ZhiyuanthePony/AlphaImages_v2` 已上传 caption metadata（`train/metadata.csv` 2296 条，`test/metadata.csv` 100 条），`load_dataset` 时自动带 `caption` 字段。
- 下载脚本 `scripts/download/alpha_images_v2.py` 可将 HF 数据集转换为 Flow-Factory 格式：
  ```
  dataset/trellis2/
  ├── images/          (2396 张 RGBA PNG)
  ├── train.jsonl      (2296 行，{"prompt": "...", "image": "xxx.png"})
  └── test.jsonl       (100 行)
  ```
- 数据已下载到 `dataset/trellis2/`，与 YAML 中 `data.dataset_dir: "dataset/trellis2"` 对应。

## 运行前必须修复

### 1. YAML 缺少 `dataset_type: image_3d`（BUG）

`trellis2_shape.yaml` 的 `data:` 段未设置 `dataset_type`，默认值为 `"general"`，会使用 `GeneralDataset` 以 **RGB** 模式加载图片。Trellis2 需要 **RGBA**（alpha 通道作为前景 mask），必须使用 `Image3DDataset`。

修复：在 `data:` 段添加 `dataset_type: "image_3d"`。

相关代码：`src/flow_factory/data_utils/loader.py` L222-224 根据 `data_args.dataset_type == 'image_3d'` 决定使用 `Image3DDataset`；`src/flow_factory/hparams/data_args.py` L57-66 定义该字段，默认 `"general"`。

### 2. YAML 缺少 `max_epochs`

当前 YAML 无 `max_epochs`，默认 `None`（无限训练）。debug 运行建议设 `max_epochs: 2`。

## 下一步任务

在单 GPU debug 模式下运行 Trellis2 GRPO 完整训练流程，验证关键变量的数值、渲染结果可视化、reward 计算是否符合预期。

## 环境与启动

- **conda 环境**：`conda activate grpo3d_trellis2`
- **启动命令**：`ff-train examples/grpo/lora/trellis2_shape.yaml`
  - `num_processes: 1` 时 CLI 走直接启动路径（`src/flow_factory/cli.py` L164-166），不经 accelerate launch。
  - 实际执行：`python -m flow_factory.train examples/grpo/lora/trellis2_shape.yaml`

## 执行方案

### Step 1: 修复 YAML

在 `examples/grpo/lora/trellis2_shape.yaml` 中：
- `data:` 段添加 `dataset_type: "image_3d"`
- `train:` 段添加 `max_epochs: 2`
- 调整 debug 参数：`num_processes: 1`、`config_file: null`、`logging_backend: none`
- 可选减小：`unique_sample_num_per_epoch: 4`

### Step 2: 插入 debug 探针代码

在以下位置插入临时 debug 日志，覆盖 **数据 → 调度器 → 采样 → 渲染 → reward → 优势 → 优化** 全链路。运行后逐一确认数据流是否正确。

#### 阶段 A: 数据与预处理

**探针 A1: RGBA 图片加载**
- **文件**: `src/flow_factory/data_utils/image_3D_dataset.py` `_load_image()`
- **检查**: 打印 `image.mode`（应 `"RGBA"`）、`image.size`
- **目的**: 确认 `dataset_type: "image_3d"` 生效

**探针 A2: 预处理输出**
- **文件**: `src/flow_factory/data_utils/dataset.py` `_preprocess_batch()` 返回前
- **检查**: 打印返回 dict 的 keys、`images` 的数量与 tensor shape、`prompt` 数量
- **目的**: 确认 RGBA 图片正确编码为 condition tensor，prompt 非空

**探针 A3: DataLoader batch 内容**
- **文件**: `src/flow_factory/trainers/trellis2_grpo.py` `sample()` 的 `group_batches = [next(data_iter)...]` 之后
- **检查**: 打印 `merged_batch` 的 keys、`prompt` 列表（确认同组 prompt 相同）、`images` tensor shape
- **目的**: 确认 `GroupContiguousSampler` 正确分组（同 prompt 连续排列），batch merge 无误

#### 阶段 B: 调度器与推理配置

**探针 B1: per-stage SDE 配置**
- **文件**: `src/flow_factory/models/trellis2/trellis2.py` `_stage_sde_kwargs()` 调用处（`__init__` 中创建 scheduler 时，约 L366-400）
- **检查**: 打印 dense/shape/tex 三个 stage 的 `dynamics_type`、`noise_level`、`num_sde_steps`
- **目的**: 确认 shape 为 `Flow-SDE`（有随机性可探索），dense/tex 为 `ODE`

**探针 B2: scheduler.train_timesteps**
- **文件**: `src/flow_factory/trainers/trellis2_grpo.py` `sample()` 的 `trajectory_indices = ...` 之后
- **检查**: 打印 `self.adapter.scheduler.train_timesteps`（shape stage scheduler 的 SDE step indices）、`trajectory_indices`
- **目的**: 确认训练的时间步索引非空且在合理范围内（shape stage 有 `num_sde_steps` 个 SDE 步）

**探针 B3: LoRA 参数确认**
- **文件**: `src/flow_factory/trainers/trellis2_grpo.py` `__init__()` 末尾
- **检查**: 打印 `self.adapter.trainable_parameters_count()`；遍历 trainable params 打印 name、shape、`requires_grad`
- **目的**: 确认 LoRA 注入到了 shape model 的 `self_attn.to_qkv` 等目标模块，非零数量

#### 阶段 C: 采样与渲染

**探针 C1: inference_with_shared_dense 输出**
- **文件**: `src/flow_factory/trainers/trellis2_grpo.py` L117-118（`sample_batch = self._inference_fn(...)` 之后）
- **检查**: 对每个 sample 打印:
  - `sample.video`: shape（应 `(T, C, H, W)`）、min/max（应 [0,1]）
  - `sample.log_prob`: shape、值域（应负数，非 NaN）
  - `sample.latent_index_map` / `sample.log_prob_index_map`: shape
  - `sample.prompt`: 值
- **目的**: 确认推理产出完整，video 由 `render_latents()` 填充，log_prob 被正确记录

**探针 C2: render_latents 内部 mesh decode**
- **文件**: `src/flow_factory/models/trellis2/trellis2.py` `render_latents()` 内部（mesh decode 之后，L2326 `mesh = self.decode_latents(sample)` 之后）
- **检查**: mesh 是否非 None、vertices 数量（应 >100）、faces 数量
- **目的**: 确认 latent → mesh decode 成功，mesh 非退化空壳

**探针 C3: 渲染结果落盘（可视化验证）**
- **文件**: `src/flow_factory/trainers/trellis2_grpo.py` `sample()` 内，`samples.extend(sample_batch)` 之后
- **落盘内容与路径**（写入 `scripts/debug/render_dump/epoch_{epoch}/`）：
  1. **条件图片**：`sample.condition_images[0]`（PIL Image），保存为 `group{g}_sample{i}_cond.png`
  2. **渲染多视角帧**：`sample.video` tensor `(T,C,H,W)` → 每帧转 PIL 保存为 `group{g}_sample{i}_frame{t:02d}.png`，或拼成一张 grid 保存为 `group{g}_sample{i}_grid.png`
  3. **（可选）mesh**：如果 `render_latents` 中顺便保存了 mesh，导出为 `group{g}_sample{i}.glb`。否则可在 C2 探针中添加 `mesh.export(path)` 临时保存。
- **代码示例**:
  ```python
  import os
  from torchvision.utils import save_image, make_grid
  dump_dir = f"scripts/debug/render_dump/epoch_{self.epoch}"
  os.makedirs(dump_dir, exist_ok=True)
  for i, s in enumerate(sample_batch):
      if s.video is not None:
          grid = make_grid(s.video, nrow=6, padding=2)  # (C, H', W')
          save_image(grid, f"{dump_dir}/group{group_idx}_sample{i}_grid.png")
  ```
- **目的**: 肉眼验证渲染的多视角是否合理（物体可辨识、视角均匀旋转、无全黑/全白/撕裂），条件图片与渲染物体是否对应

#### 阶段 D: Reward 计算

**探针 D1: RewardProcessor 格式转换**
- **文件**: `src/flow_factory/rewards/reward_processor.py` `_compute_pointwise_batch()` 调用 reward model 之前
- **检查**: 传给 PickScore 的 kwargs 的 keys（应有 `video` 无 `image`）、`video` 类型（应 `List[List[PIL.Image]]`）、每个 sample 帧数、单帧 size
- **目的**: 确认 `_convert_media_format` 正确将 `Tensor(T,C,H,W)` → `List[PIL.Image]`

**探针 D2: PickScore 打分**
- **文件**: `src/flow_factory/rewards/pick_score.py` `_compute_video_scores()` 返回前，以及 `__call__` L121 归一化前后
- **检查**:
  - `self.model.logit_scale.exp()` 的值（CLIP 学到的温度参数）
  - per-frame 原始分（`logit_scale * cosine_sim`）的 min/max/mean
  - mean-pooled raw score（归一化前）
  - 归一化后的 score（`/ 26`）
- **注意**: 原始分范围不确定。PickScore 在自然图片上的典型值约 17-22，但 3D 渲染的多视角帧分布未知——黑色背景、非自然光照、物体视角变化大都可能导致分数偏移。归一化除数 `26` 是硬编码的经验值，debug 时需观察实际数值，必要时调整或去掉归一化。
- **目的**: 建立 3D 渲染场景下 PickScore 的基线分布，判断是否需要调整归一化策略

**探针 D3: reward_buffer.finalize 输出**
- **文件**: `src/flow_factory/trainers/grpo.py` `prepare_feedback()` 内，`rewards = self.reward_buffer.finalize(...)` 之后
- **检查**: `rewards` dict 的 keys、每个 reward tensor 的 shape 和值域
- **目的**: 确认 reward 正确从 buffer 汇总到所有 samples

#### 阶段 E: 优势计算

**探针 E1: advantage 数值**
- **文件**: `src/flow_factory/trainers/grpo.py` `prepare_feedback()` 内，`self.compute_advantages(...)` 之后
- **检查**: 从 samples 中取 `sample.advantage`，打印 shape、mean、std、min、max
- **目的**: 确认 advantage 在组内有正有负（zero-centered），非全零/NaN。`advantage_aggregation: gdpo` 下应为 per-timestep advantage。

#### 阶段 F: 优化

**探针 F1: loss 值**
- **文件**: `src/flow_factory/trainers/grpo.py` `optimize()` 内，`accelerator.backward(loss)` 之后
- **检查**: `loss.item()`
- **目的**: 确认 loss 非 NaN/Inf，数值在合理范围

**探针 F2: 梯度流通**
- **文件**: `src/flow_factory/trainers/grpo.py` `optimize()` 内，backward 之后、optimizer.step 之前
- **检查**: 遍历 trainable params，打印 `.grad` 是否非 None、grad norm
- **目的**: 确认梯度从 loss 回传到了 LoRA 参数

**探针 F3: 参数更新**
- **文件**: `src/flow_factory/trainers/grpo.py` `optimize()` 内，optimizer.step 之后
- **检查**: 打印一个 LoRA 参数的值的 mean，与 step 之前对比
- **目的**: 确认参数确实被更新了（delta 非零）

### Step 3: 运行并观察

```bash
conda activate grpo3d_trellis2
cd /home/zhiyuan_ma/code/Flow-Factory
ff-train examples/grpo/lora/trellis2_shape.yaml
```

### Step 4: 清理

确认全部通过后，移除所有 debug 探针代码。

## 潜在风险

- decode+render 在 GPU 上可能 OOM（mesh simplify + rasterization），需要关注显存；如遇 OOM 可降 `render_resolution` 到 256 或 `render_num_frames` 到 8。
- PickScore 模型额外占约 2GB 显存。
- `sampler_type: group_contiguous` 在 `num_processes=1` 时要求 `unique_sample_num_per_epoch % 1 == 0` 且 `unique_sample_num * group_size % per_device_batch_size == 0`，当前参数 (4*4/2=8) 满足。
