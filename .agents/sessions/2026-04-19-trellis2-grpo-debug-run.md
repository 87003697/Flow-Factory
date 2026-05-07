# Session Handoff: Trellis2 GRPO Debug Run — Decode/Render/Reward 贯通

## 任务目的
在 debug 数据集（8 train + 2 test）上跑通 Trellis2 GRPO 训练的端到端流程（sampling → decode → render → reward → optimize），逐个修复运行时错误。

## 执行内容
- 创建 `dataset/trellis2_debug/`：从 AlphaImages_v2 取前 8 条 train + 2 条 test，图片通过 symlink 引用
- 修改 `trellis2_shape.yaml` 中 `dataset_dir` 指向 debug 数据集
- **[已修复] Decoder dtype 不匹配**：`shape_decoder` / `tex_decoder` 内部 `self.dtype = fp16` 导致 triton sparse conv3d 报 `tl.dot operands fp16 vs fp32`；通过 `decoder.to(dtype=torch.float32)` + `decoder.dtype = torch.float32` 解决
- **[已修复] nvdiffrast rasterize 要求 fp32**：在 `render_latents` 上加 `@torch.autocast('cuda', enabled=False)` 装饰器
- **[已修复] PickScore 缺少 prompt**：三个 inference 方法已有 `prompt` 显式参数但未传递给 `Trellis2Sample`；已在 `Trellis2Sample(prompt=prompts[b] ...)` 中补上
- **[已修复] gradient_accumulation_steps = 0**：`auto` 计算出 0，临时改为显式 `4`
- **[未修复] forward_one_step 中 flow model input_layer 维度不匹配**：`(37318×64 vs 32×1536)`——x_t 有 64 channels（shape latent），但 input_layer 期望 32 channels（dense model）。推测是 `_build_sparse_inputs` 或 `_forward_sparse` 中 `get_flow_model()` 拿到了错误的模型

## 调试经验
- **Trellis2 decoder `self.dtype` 属性**：`SparseUnetVaeDecoder.__init__` 中 `use_fp16=True` 会设置 `self.dtype = torch.float16`；其 `forward()` 在进入 blocks 前执行 `h = h.type(self.dtype)` 做内部 cast，离开 blocks 后 `h = h.type(x.dtype)` cast 回。**`.to(dtype=fp32)` 只改参数不改 `self.dtype` 属性**，必须手动 `decoder.dtype = torch.float32`
- **`torch.autocast` 与 triton sparse conv**：即使 decoder 参数和输入都是 fp32，活跃的 autocast context 仍会导致 triton kernel 编译时出 dtype 不匹配。需在 decode + render 全链路上显式 `autocast(enabled=False)`
- **`@torch.no_grad()` 包装导致 `filter_kwargs` 丢失 `**kwargs`**：`filter_kwargs` 用 `inspect.signature()` 检测函数签名；`@torch.no_grad()` 装饰后签名可能丢失 `VAR_KEYWORD` 参数，导致 `prompt` 等非显式参数被过滤掉。**必须用显式命名参数**
- **Trellis2 原始训练代码不跑 decoder**：`sparse_flow_matching.py` 的 `training_losses` 只调用 denoiser forward，不做 decode/render。decode dtype 边界问题**仅存在于 Flow-Factory 的 GRPO 集成中**（reward 需要 render multi-view RGB）
- **Trellis2 的 `manual_cast` 设计**：`autocast 关闭时强制 cast，开启时不 cast`——这意味着在 bf16 autocast 下，模块内部的 `manual_cast` 会被跳过，完全依赖 autocast 来管理精度。但 sparse conv（triton kernel）不支持 autocast 隐式 cast

## 修改函数清单

### `src/flow_factory/models/trellis2/trellis2.py` (主改动，+493 -80)

| 函数 | 改动类型 | 说明 |
|------|---------|------|
| `load_pipeline` | 修改 | 适配 pipeline 加载逻辑 |
| `_make_sparse_scheduler` | 修改 | Scheduler 构造适配 |
| `enable_gradient_checkpointing` | **新增** | Trellis2 自定义 per-block gradient checkpointing |
| `_as_scalar_resolution` | **新增** | 将 `(H,W)` tuple 统一转为 `int`，解决 FF 与 Trellis2 的 resolution 类型不匹配 |
| `_resolve_conditioning` | 修改 | 修复 `if tensor_var` 对多元素 tensor 的歧义判断 → `tensor_var is not None` |
| `_reduce_sparse_log_prob` | 修改 | 修复 tensor boolean 歧义 + 逻辑调整 |
| `_resolve_component_names` | 修改 | 扩展为 `hasattr(x, 'to')` duck-typing，使 `DinoV3FeatureExtractor` 等非 `nn.Module` 也能被移动到 GPU |
| `inference_modules` (property) | **重写** | 动态推导上下游依赖（decoder、sparse structure、upstream transformer），替代硬编码 |
| `forward` / `forward_one_step` | 修改 | 添加 timestep 1D tensor assert，确保 `t.eq(t[0]).all()`；加入 FWD DEBUG 日志 |
| `_apply_cfg_sparse` | 修改 | CFG 逻辑调整适配 |
| `inference` | 修改 | 入口加 `_as_scalar_resolution`；`prompt` 参数显式化并传递给 `Trellis2Sample` |
| `inference_with_shared_dense` | **新增** | 共享 dense stage 的推理路径，含 prompt 传递 |
| `inference_with_shared_dense_shape` | **新增** | 共享 dense+shape stage 的推理路径，含 prompt 传递 |
| `_inference_shape` | 修改 | 入口加 `_as_scalar_resolution` |
| `_inference_tex` | 修改 | 入口加 `_as_scalar_resolution` |
| `decode_shape` | 修改 | `decoder.dtype = torch.float32` + features cast fp32，修复 triton sparse conv dtype 不匹配 |
| `decode_texture` | 修改 | 同 `decode_shape` |
| `_build_envmap` | 修改 | 微调 |
| `render_latents` | 修改 | 加 `@torch.autocast('cuda', enabled=False)` 装饰器，修复 nvdiffrast fp32 要求 |

### `src/flow_factory/models/trellis2/pipeline.py` (+47 行改动)

| 函数 | 改动类型 | 说明 |
|------|---------|------|
| `Trellis2PseudoPipeline` (class) | 修改 | `OPTIONAL_MODELS` 增加 `tex_slat_flow_model_1024` / `tex_slat_decoder` |
| `from_pretrained` | 修改 | 加载逻辑适配，处理旧 pipeline.json 兼容 |
| `get_flow_model` | 修改 | 防御性处理 resolution 为 tuple 的情况 `resolution[0]` |

### `src/flow_factory/trainers/abc.py` (+7 行改动)

| 函数 | 改动类型 | 说明 |
|------|---------|------|
| `_init_reward_model` | 修改 | 安全获取 tokenizer：`self.adapter.tokenizers[0] if self.adapter.tokenizers else None` |
| `_init_dataloader` | 修改 | 微调 |

### `src/flow_factory/trainers/grpo.py` (+25 行改动)

| 函数 | 改动类型 | 说明 |
|------|---------|------|
| `_extra_eval_inference_kwargs` | **新增** | 为 eval 推理提供额外 kwargs |
| `evaluate` | 修改 | 微调 |
| `prepare_feedback` | 修改 | DEBUG PROBE 打印 |
| `optimize` | 修改 | DEBUG PROBE 打印 |

### `src/flow_factory/trainers/registry.py` (+1 行)

| 改动 | 说明 |
|------|------|
| 模块级注册 | 添加 `'trellis2_grpo'` trainer 注册 |

### `src/flow_factory/rewards/reward_processor.py` (+10 行)

| 函数 | 改动类型 | 说明 |
|------|---------|------|
| `_compute_pointwise_batch` | 修改 | DEBUG PROBE 打印 |

### `src/flow_factory/rewards/pick_score.py` (+6 行改动)

| 函数 | 改动类型 | 说明 |
|------|---------|------|
| `_compute_video_scores` | 修改 | DEBUG PROBE 打印 |
| `__call__` | 修改 | DEBUG PROBE 打印 |

### `src/flow_factory/data_utils/image_3D_dataset.py` (+8 行改动)

| 函数 | 改动类型 | 说明 |
|------|---------|------|
| `Image3DDataset` (class) | 修改 | 类属性调整 |
| `_load_image` | 修改 | 图片加载逻辑调整 |

### `src/flow_factory/hparams/training_args.py` (+1 行)

| 函数 | 改动类型 | 说明 |
|------|---------|------|
| `get_num_train_timesteps` | 修改 | 微调 |

### `src/flow_factory/models/abc.py` (-1 行)

| 改动 | 说明 |
|------|------|
| 模块级 | 移除 1 行 |

### 已删除文件

| 文件 | 说明 |
|------|------|
| `scripts/test_trellis2_inference.py` | -316 行，临时测试脚本 |
| `scripts/test_trellis2_single_step.py` | -407 行，临时测试脚本 |

## 参考代码
| 文件 | 关键位置 | 说明 |
|------|---------|------|
| `src/flow_factory/models/trellis2/trellis2.py` | `decode_shape()` L2141, `decode_texture()` L2192, `render_latents()` L2319 | Decoder 调用和 autocast 禁用 |
| `src/flow_factory/models/trellis2/trellis2.py` | `_forward_sparse()` L1158, `forward_one_step()` L952 | Training forward path，当前报错位置 |
| `src/flow_factory/models/trellis2/trellis2.py` | `inference_with_shared_dense()` L1386 | Shape 训练的 sampling 入口，prompt 传递 |
| `src/flow_factory/models/trellis2/pipeline.py` | `_STAGE_MODEL_MAP` L433, `get_flow_model()` L441 | Stage → flow model 映射 |
| `third_party/TRELLIS.2/trellis2/models/structured_latent_flow.py` | `forward()` L170-195 | `manual_cast` 进出主干、input_layer 维度 |
| `third_party/TRELLIS.2/trellis2/models/sc_vaes/sparse_unet_vae.py` | `forward()` L478-507 | Decoder `h.type(self.dtype)` / `h.type(x.dtype)` 设计 |
| `third_party/TRELLIS.2/trellis2/modules/utils.py` | `manual_cast()` | autocast 感知 cast |
| `third_party/TRELLIS.2/trellis2/trainers/basic.py` | `run_step()` | 原始训练 autocast 模式 |
| `examples/grpo/lora/trellis2_shape.yaml` | 全文件 | Debug 配置（已临时修改 dataset_dir, gradient_accumulation_steps, mixed_precision 等） |

## 最终方案
Sampling → decode → render → reward 全链路已贯通（PickScore 正常输出分数 0.61–0.73），还差 optimize 阶段的最后一个 bug。当前方案是"暴力 fp32"——把 decoder 强制 fp32、禁用 autocast。下一步应参考 Trellis2 原生的 `manual_cast` + `use_fp16` 设计，做更精细的 dtype 管理。

## 下一步任务
1. **修复 `forward_one_step` 中 flow model 维度不匹配 bug**：`(37318×64 vs 32×1536)` —— 需确认 `_build_sparse_inputs` 中为什么构建的 `x_t` 与 `_forward_sparse` 中选到的 flow model 不匹配
2. **参考 Trellis2 原生 dtype 设计重构 adapter 精度管理**：
   - Flow model 内部有 `manual_cast`，只对 blocks 做半精度；input_layer/out_layer 保持 fp32
   - Decoder 同理：blocks 内 `self.dtype`，进出用 `x.dtype` 对齐
   - 在 Flow-Factory 的 bf16 autocast 下，`manual_cast` 会被跳过（`torch.is_autocast_enabled()` 为 True）
   - 需要在 decode/render 路径显式关闭 autocast，让 `manual_cast` 生效
3. **清理 DEBUG PROBE 代码**（6 个文件）
4. **恢复 `gradient_accumulation_steps: auto` 并修复计算逻辑**

## 初步方案
- **Bug 1（维度不匹配）**：在 `_build_sparse_inputs` 和 `forward_one_step` 入口加 assert 打印 `stage`/`stage_resolution`/`flow_model.input_layer.in_features`，确认是不是 trainer 传入的 `stage=None` 导致 fallback 到了错误的 `self.transformer`（可能 LoRA 包装后不再是 `transformer_shape_1024` 的同一对象）
- **Bug 2（dtype 设计）**：改用"在 `render_latents` 入口关闭 autocast + 不强制 decoder fp32"的方案——让 decoder 保持原始的 `use_fp16` 布局，只通过关闭 autocast 让 `manual_cast` + `h.type(self.dtype)` / `h.type(x.dtype)` 正常工作。输入 `slat.feats` 传 fp32，decoder 内部自动 cast 到 fp16 跑 blocks，出来再 cast 回 fp32
- **入口文件**：`trellis2.py` 的 `decode_shape`/`decode_texture`（去掉 `decoder.to(dtype=fp32)` + `decoder.dtype = fp32`），`forward_one_step`（增加 stage 断言）
- **潜在风险**：LoRA 包装可能改变 `self.transformer` 与 `self.transformer_shape_1024` 的对象 identity，导致 `get_flow_model` 返回未经 LoRA 包装的原始模型
