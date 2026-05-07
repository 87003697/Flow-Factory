# Session Handoff: Trellis2 GRPO 首次运行

## 任务目的

完成代码整理后，在 debug 数据集上首次运行 `ff-train examples/grpo/lora/trellis2_shape.yaml`，验证全链路（sample → decode → render → reward → optimize）可以跑通。

## 执行内容

- **验证 forward_one_step 修复已就位**：`stage=None` fallback 通过 `_training_stage` 属性从 `pipeline._target_flow_model` 推导；`effective_concat_cond = tex_concat_cond if stage == 'tex' else None` 防止 tex conditioning 污染 shape forward，通道数不再错误拼接。
- **`Trellis2Sample` 添加 per-stage 字段**：每个 stage 增加了 `{dense,shape,tex}_latent_index_map` / `log_prob_index_map` / `timesteps`；新增 `activate_stage(stage)` 替代 `_inference_*` 里手动逐字段复制。
- **去掉旧版 `image_cond` / `neg_image_cond` 参数**：三个 `inference*` 方法和 `_resolve_conditioning` 均移除，`_resolve_conditioning` 的 4 行嵌套三元改为直接赋值。
- **`pipeline.py` 重构**：`tex_slat_flow_model_1024` / `tex_slat_decoder` 移入 `REQUIRED_MODELS`；新增 `_instantiate()` 静态方法统一处理 `{name, args}` 格式实例化；`from_pretrained` 去掉双重 try-except 和 `low_cpu_mem_usage` 参数，配置读取改为 `if not os.path.exists(config_path)` 内联。

## 调试经验

- **`tex_concat_cond` 泄漏**：rollout 时 `_inference_tex` 写入 `sample.tex_concat_cond`，经 `**batch` 传入 `forward()` 后被无条件传给 `_build_sparse_inputs`，导致 shape forward 输入通道从 32 变成 64。修复关键在 `effective_concat_cond` 这一行守卫。
- **旧版 `image_cond` 别名不再走 Trainer 路径**：`preprocess_func` 已直接返回 `image_cond_512` / `image_cond_1024`，旧别名在正常训练中永远用不到，去掉安全。

## 参考代码

| 文件 | 关键位置 | 说明 |
|------|---------|------|
| `src/flow_factory/models/trellis2/trellis2.py` | `forward()` L1025-1064 | stage fallback + effective_concat_cond 守卫（核心 bug 修复） |
| `src/flow_factory/models/trellis2/trellis2.py` | `_training_stage` L984-987 | 从 pipeline 配置推导训练 stage |
| `src/flow_factory/models/trellis2/trellis2.py` | `Trellis2Sample.activate_stage` L194-206 | per-stage 字段投影到标准字段 |
| `src/flow_factory/models/trellis2/pipeline.py` | `_instantiate()` L178-191, `from_pretrained()` L193-260 | 重构后的模型加载 |
| `examples/grpo/lora/trellis2_shape.yaml` | 全文 | Debug 训练配置（dataset_dir=trellis2_debug, max_epochs=2, 4 unique samples） |

## 最终方案

所有代码改动均为未提交状态（on top of commit `f2ee782`）。训练配置当前处于 debug 模式：

- `dataset_dir: "dataset/trellis2_debug"`（8 train + 2 test 图片）
- `max_epochs: 2`、`unique_sample_num_per_epoch: 4`
- `gradient_accumulation_steps: 4`（临时显式值，避免 auto 计算出 0）
- DEBUG PROBE 代码仍残留在 6 个文件（`grpo.py`×20、`trellis2_grpo.py`×51、`reward_processor.py`×10、`pick_score.py`×5、`trellis2.py`×2、`image_3D_dataset.py`×4），用于诊断，等训练跑通再清理。

## 下一步任务

在单 GPU debug 模式下运行 Trellis2 GRPO 完整训练，验证全链路可以无错误完成 2 epoch。

## 代码精炼方案

`trellis2.py` 当前 2415 行，`pipeline.py` 537 行。以下按优先级排列可精炼项。

### P0: 违反项目 Rule

1. **删除所有 section-divider 注释** — `.cursor/rules/no-section-divider-comments.mdc` 明确禁止 `# ====...` / `# ----...` 风格的分割线注释。当前 `trellis2.py` 有 16 处、`pipeline.py` 有 7 处。应全部删除，依靠 class/method docstring 自述结构。
2. **`encode_image` docstring 有拼写残留** — L866 行 `c` 孤字，需删除。

### P1: 消除重复代码（~400 行可省）

3. **合并 `_inference_shape` 和 `_inference_tex`** — 两者结构高度同构（123 vs 144 行），唯一区别是 `tex` 多了 `concat_cond` 的构造和 `noise_channels` 计算。可提取为 `_inference_sparse_stage(stage: str, ...)` 单一方法，内部 if/else 处理差异，预计合并后不超过 160 行。
4. **合并 `inference_with_shared_dense` 和 `inference_with_shared_dense_shape`** — 两者分别是 129 和 134 行，核心区别仅在于 pilot 跑 1 个 stage 还是 2 个。可重构为统一的 `_inference_shared_upstream(pilot_stages: List[str], train_stage: str, ...)` 方法，由 `Trellis2GRPOTrainer.__init__` 传入 `pilot_stages` 参数。
5. **`encode_image` 是死代码** — `preprocess_func` 已完全取代了 `encode_image` 的功能（双分辨率编码 + 预处理）。`encode_image` 仍保留了 74 行旧逻辑（单分辨率、不同的 dummy embed_dim 获取路径），但当前训练/推理路径中无调用方。如果 `BaseAdapter` 接口不要求实现 `encode_image`，可以直接删除或改为 `raise NotImplementedError`。

### P2: 结构优化

6. **`Trellis2Sample` 字段爆炸** — 当前 dataclass 有 ~30 个字段（dense_×8 + shape_×8 + tex_×9 + 全局×5）。建议：
   - 引入 `@dataclass StageData` 子结构体：`final_latent`, `all_latents`, `log_probs`, `image_cond`, `neg_image_cond`, `latent_index_map`, `log_prob_index_map`, `timesteps`。
   - `Trellis2Sample` 改为 `dense: Optional[StageData]`, `shape: Optional[StageData]`, `tex: Optional[StageData]`，加上 `sparse_coords`, `resolution`, `mesh` 等全局字段。
   - `activate_stage` 变成 `self.all_latents = self.{stage}.all_latents` 的投影，逻辑不变但字段数从 30 降到 ~10。
   - **注意**：需要同步更新 `_SPARSE_LATENT_FIELDS`, `_stack_values`, `_STAGE_FIELD_MAP`。
   - **风险**：如果现有 Trainer 代码通过 `getattr(sample, f'{stage}_all_latents')` 动态访问，需要一并修改。先 grep 确认引用范围。

7. **`_get_stage_guidance` 参数透传链冗长** — `guidance_scale/interval/rescale` 从 `inference()` → `_get_stage_guidance()` → 又传给 `_inference_*`。而 `_get_stage_guidance` 本身就是读 pipeline.json 的 per-stage config。建议 `_inference_*` 直接调用 `_get_stage_guidance`，不再从外部接收这三个参数，减少 `inference()` 参数列表。

8. **`_find_local_model_path` (44 行) 应移至 `pipeline.py`** — 这是路径发现逻辑，与 adapter 的模型适配职责无关。放到 `Trellis2PseudoPipeline.from_pretrained` 里更内聚。

### P3: 命名 & 风格

9. **`_STAGE_MODEL_MAP` / `_STAGE_SCHEDULER_ATTR` / `_STAGE_SAMPLER_PARAMS_ATTR`** — 三个 dict 分别在 `pipeline.py` 和 `trellis2.py` 定义了 stage → attribute 的映射。考虑统一到 pipeline 层的一个 `StageConfig` 注册表：
   ```python
   _STAGES = {
       'dense': StageConfig(model_attr='sparse_structure_flow_model', scheduler_attr='scheduler_dense', ...),
       'shape': StageConfig(...),
       'tex': StageConfig(...),
   }
   ```
   这样 `get_flow_model`, `_get_stage_scheduler`, `_get_stage_guidance` 都可以统一查表。

10. **`pipeline.py` 的 `__init__` 同时暴露 `self.sparse_structure_flow_model` 和 `self.transformer_dense`**，两者指向同一对象。`transformer_dense` 是为了被 `transformer_names` 发现，但造成了名字冗余。如果 dense stage 永远不需要 LoRA，可以只保留 `sparse_structure_flow_model` 不挂 `transformer_` 前缀。

11. **`_apply_cfg_sparse` 中的 for loop** — L1186 的 `for b in range(batch_size)` scatter-then-gather 逻辑可以用 `scatter_add` + `scatter_add(count)` 的向量化实现替换（与 `_reduce_sparse_log_prob` 的模式一致），避免 Python 循环。

### P4: Misc

12. **`pipeline.py` 的 `preprocess_image` (55 行)** — 这是从 Trellis2 官方代码复制的图片预处理。如果 `trellis2.pipelines` 已经暴露了同名方法，可以直接委托调用，避免维护副本。
13. **`pipeline.py` 的 `apply_cfg_rescale` (18 行)** — 与 `Trellis2Adapter._apply_cfg_dense` 功能重叠，但签名不同。如果 pipeline 层不再被直接使用（adapter 层已接管所有 inference），可以删除 pipeline 层的版本。
14. **`low_cpu_mem_usage=False` 参数** — `load_pipeline` L366 传给 `from_pretrained`，但 `from_pretrained` 签名中没有这个参数（被 `**kwargs` 吞掉后无人使用）。应删除。

### 精炼优先级建议

| 阶段 | 预计减少行数 | 风险 |
|------|------------|------|
| P0 (rule compliance) | ~25 行 | 零 |
| P1-3 (合并 inference_shape/tex) | ~100 行 | 低（纯内部重构） |
| P1-4 (合并 shared_dense 变体) | ~120 行 | 低 |
| P1-5 (删除 encode_image) | ~74 行 | 需确认无外部调用 |
| P2-6 (StageData 子结构) | 净减 ~50 行 + 可读性大幅提升 | 中（影响 _stack_values） |
| P3 (命名统一) | 净减 ~30 行 | 低 |

合计可从 2415 行降到 ~2000 行左右，同时提升可维护性。

## 初步方案

**启动命令**：
```bash
conda activate grpo3d_trellis2
cd /home/zhiyuan_ma/code/Flow-Factory
ff-train examples/grpo/lora/trellis2_shape.yaml
```

**观察顺序（对应 reward-config session 中的 debug 探针）**：

1. **阶段 A（数据）**：确认 `image.mode == "RGBA"`、`image_cond_512` / `image_cond_1024` 非空、prompt 非 None
2. **阶段 B（调度器）**：确认 shape stage 的 `dynamics_type = Flow-SDE`、`train_timesteps` 非空、LoRA 参数 count 非零
3. **阶段 C（推理）**：确认 `sample.video` shape 为 `(24, 3, 512, 512)`、`sample.log_prob` 为负数非 NaN
4. **阶段 D（reward）**：确认 PickScore 收到 `video`（不是 `image`）、分数在 0.5–0.8 范围
5. **阶段 E（优势）**：确认 advantage mean ≈ 0、std > 0、非全零
6. **阶段 F（优化）**：确认 loss 非 NaN、梯度非零、参数 delta 非零

**潜在风险**：
- `render_latents` 的 mesh decode 可能 OOM（建议先降 `render_resolution: 256` 或 `render_num_frames: 8` 观察显存）
- `gradient_accumulation_steps: 4` 是手动计算值，如果 batch 参数变了需要重算
- PickScore 分数的归一化除数 `26` 是硬编码经验值，3D 渲染分布可能偏移，需观察实际值域
