# Session Handoff: trellis2 multi-stage alt-train plan

## 任务目的

把 Trellis2 适配从单 transformer 训练扩展到 shape+tex 多 transformer **epoch 级交替训练**（DDP only, v1）。本次 session 的产出是一份完整、经过三轮 audit 的实施 plan：[trellis2-multi-stage-alt-train_41eba9d6.plan.md](/home/zhiyuan_ma/.cursor/plans/trellis2-multi-stage-alt-train_41eba9d6.plan.md)。下一 session 直接按 plan 中 todos 顺序实施。

## 执行内容

- 调研 [trellis2.py](src/flow_factory/models/trellis2/trellis2.py) / [trellis2_grpo.py](src/flow_factory/trainers/trellis2_grpo.py) / [grpo.py](src/flow_factory/trainers/grpo.py) / [pipeline.py](src/flow_factory/models/trellis2/pipeline.py)，确认现有单 stage 训练路径和约束。
- 经过三轮迭代收敛到 "DDP-only + base trainer 不感知 multi-optimizer + Trellis2GRPOTrainer 自管 multi-optimizer + by-name checkpoint" 架构。
- 第 1 轮 audit：暴露 `stage_context` exit 必须 restore（critical）/ `save_checkpoint` 按 `model_only` 分两路径 / non-primary opt 不能走 `accelerator.prepare` / evaluate 必须显式 full pipeline / EMA 必须 fail-fast。
- 第 2 轮 audit：暴露 `save_checkpoint` final_dir 重算（base mutate 局部变量）/ `_init_optimizer` 参数源 / `_upstream_stages` 初始 init / resume 时序 / grad isolation sanity check。
- 第 3 轮 audit：误判一个"grad 累积 bug"（实际 sample 全 no_grad + optimize 单 stage forward 是结构性安全），收回。审计 `_STAGE_BROADCAST_FIELDS` / `is_training_stage` / `use_ref_parameters` 三处确认结构性正确，无需追加修订。
- 写入 plan：13 个修订点全部落地，3 个 audit 项确认结构性安全。

## 调试经验

- **base `save_checkpoint` 行为有双路径**：[trainers/abc.py L391-399](src/flow_factory/trainers/abc.py) delegate 给 [models/abc.py L1392-1469](src/flow_factory/models/abc.py)；`model_only=True` 只 save model 不 save optimizer，`model_only=False` 才走 `accelerator.save_state`。子类 multi-opt 的 by-name save 必须分这两路径。
- **base `save_checkpoint` mutate 局部变量**：`save_directory = os.path.join(save_directory, f"checkpoint-{epoch}")` 是 base 内部 join，super() 返回后子类拿不到 final dir，必须按相同规则**重新算一次**。
- **`accelerator.prepare(opt)` 在 DDP 下虽是 identity，但仍 push 到 `self._optimizers` list**，导致 `accelerator.save_state` 自动按位置 save 成 `optimizer_1.bin`，与 by-name save 冲突。non-primary opt 必须用 raw `torch.optim.AdamW`、跳过 prepare。
- **`stage_context` 必须 try/finally exit 回滚**：base `start()` 主循环里 `save_checkpoint` 和 `evaluate` 在 `with stage_context()` **外面**。如果 exit 不回滚，下一 epoch 顶部 `save_checkpoint` 会用上一 stage 残留的 optimizer 状态污染 `optimizer.bin`。
- **multi-stage grad isolation 是结构性保证**：sample 阶段 [grpo.py L128](src/flow_factory/trainers/grpo.py) 全 `with torch.no_grad()`；optimize 阶段调 `adapter.forward(...)` 只 forward active stage 单个 transformer。inactive stage params 永远不参与 forward 不接 grad，不需要修 trellis2 inference。
- **`use_ref_parameters()` LoRA 路径**：遍历 `target_module_map` 全部 disable，multi-stage 下 shape_lora/tex_lora 同时 disable，但 active forward 只走 active stage transformer，inactive 的 disable 无副作用，结构性正确。
- **`_STAGE_BROADCAST_FIELDS` 是 stage-keyed 字典**（'dense' / 'shape' 各自的 broadcast 字段），按 `self._upstream_stages` 迭代，自动随 active stage 切换，**plan 不动是对的**。

## 参考代码

| 文件 | 关键位置 | 说明 |
|------|---------|------|
| [src/flow_factory/models/trellis2/trellis2.py](src/flow_factory/models/trellis2/trellis2.py) | `_forward_dense` L1190-1211 / `_forward_sparse` L1298-1324 / `forward` L1104-1105 / `inference` L1400+ / `_run_stage_inference` L1333+ / `inference_modules` L829 / `load_scheduler` L531 | 改动点 1 / 2 主战场（adapter prepared-component fix + stage runtime context + alias 迁移） |
| [src/flow_factory/models/trellis2/pipeline.py](src/flow_factory/models/trellis2/pipeline.py) | `__init__` L97-176 / `from_pretrained` L193-270 | 改动点 3 主战场（target_flow_model 接受 list） |
| [src/flow_factory/trainers/grpo.py](src/flow_factory/trainers/grpo.py) | `start` L60-91 / `evaluate` L98-143 / `_extra_eval_inference_kwargs` L94-96 / optimize accumulate L231 / clip_grad_norm L336 / KL no_grad L287 | 改动点 4 主战场（base trainer 加 3 个 default no-op hook） |
| [src/flow_factory/trainers/trellis2_grpo.py](src/flow_factory/trainers/trellis2_grpo.py) | `__init__` / `_set_static_graph` L101-107 / `sample` L109 / `_rollout_group` L143-164 / `_distributed_upstream_stages` L166 | 改动点 5 主战场（trellis2 子类 override hook + 自管 multi-opt） |
| [src/flow_factory/trainers/abc.py](src/flow_factory/trainers/abc.py) | `save_checkpoint` L391-399 / `load_checkpoint` L403 / `self.epoch=0` L64 | base trainer 不动，但实施时核对 epoch resume 来源 |
| [src/flow_factory/models/abc.py](src/flow_factory/models/abc.py) | `save_checkpoint` L1392-1469 / `use_ref_parameters` L552-575 / `ema_step` L510 | adapter 层不动，但实施时核对 model_only 双路径 + ref_parameters 多 stage 行为 |
| [examples/grpo/lora/trellis2/shape.yaml](examples/grpo/lora/trellis2/shape.yaml) / [tex_unified_reward.yaml](examples/grpo/lora/trellis2/tex_unified_reward.yaml) | 完整 yaml | 单 stage 回归 smoke 用 |

## 最终方案

**架构核心**：

1. **base GRPOTrainer 仅加 3 个 default no-op hook**（`stage_context` / `_active_components` / `_active_parameters`），`_init_optimizer` / `_initialization` / `save_checkpoint` / `load_checkpoint` / `optimize` / `sample` / `prepare_feedback` 内部代码 0 改动。
2. **base `start()` 把 sample/prepare_feedback/optimize 三句包进 `with self.stage_context():`**，stage-neutral 的 save_checkpoint / evaluate / ema_step 保持在 with 外。
3. **Trellis2GRPOTrainer 自管 multi-optimizer**：override `_init_optimizer` 返回 primary 的 AdamW；`__init__` super 之后用 raw `torch.optim.AdamW`（**不调 `accelerator.prepare`**）建非 primary opt；EMA fail-fast。
4. **`stage_context` reentrant + try/finally exit restore**：进入时记录 prev state（current_stage / optimizer / num_train_timesteps / upstream / inference），exit 时恢复，保证 stage-neutral 路径永远看到 primary。
5. **by-name checkpoint**：`optimizer_stage_<name>.pt`，model_only=True/False 双路径都覆盖。
6. **evaluate 显式走全 pipeline**：override `_extra_eval_inference_kwargs` 注入 `{stages: [dense,shape,tex], training_stage: tex}`，evaluate 不依赖 current_stage。

**为什么这个方案**：
- DDP only 让"non-primary opt 不 prepare" 这个简化合法（DDP 下 prepare 是 identity）→ 完全避开 accelerate 多 optimizer 自动 save/load 的位置寻址 footgun
- base 不感知 multi-opt → 单 stage trainer 行为 0 变化，回归风险极低
- by-name ckpt + model_only 双路径覆盖 → 跨配置可读 / 缺文件 fail-fast

**v1 边界（必须守住）**：
- DDP only（不兼容 DeepSpeed，留 v2）
- `ema_decay=0` 强制（multi-stage EMA 留 v2，`__init__` fail-fast 校验）
- `finetune_type='lora'` 强制（multi-stage full finetune 的 ref_ema 行为已 audit 但未 smoke）

## 下一步任务

按 plan 中 todos 顺序实施，每完成一组改动跑一次 smoke test 早期暴露问题。

## 初步方案

**实施顺序**（优先保证单 stage 退化等价，再上多 stage）：

### 阶段 1：adapter + pipeline 层（改动点 1 / 2 / 3）
- todo `pipeline_target_list`：[pipeline.py](src/flow_factory/models/trellis2/pipeline.py) `__init__` / `from_pretrained` 接受 `target_flow_model: Union[str, List[str]]`，新增 `_target_flow_models` list，**移除** `_target_flow_model` 单值 alias，保留 `self.transformer`（BaseAdapter 契约）
- todo `adapter_target_flow_model_audit`：迁移 [trellis2.py](src/flow_factory/models/trellis2/trellis2.py) L837 `inference_modules` / L1479 `sample()` 默认 stages / L1097/L1465 docstring 等所有 `_target_flow_model` 直访
- todo `adapter_resolve_helper`：抽 `_resolve_flow_model` + `_stage_component_name` helper，重写 `_forward_dense` / `_forward_sparse`
- todo `adapter_current_stage`：加 `_current_stage` + `current_stage` property/setter（rebind `pipeline.scheduler`），`load_scheduler` 默认返回 `_training_stages[0]`
- todo `adapter_get_stage_parameters`：新增 `get_stage_parameters(stage)` helper（前缀过滤）
- **冲烟**：跑 [shape.yaml](examples/grpo/lora/trellis2/shape.yaml) 单 stage 验证 prepared-component fix + alias 迁移退化等价（ratio_mean ≈ 1.0）

### 阶段 2：base trainer hook（改动点 4）
- todo `base_trainer_hooks`：[grpo.py](src/flow_factory/trainers/grpo.py) 加 3 个 default no-op hook + `optimize` 内 `accumulate(*trainable_components)` 与 `clip_grad_norm_(get_trainable_parameters())` 改走 hook + `start()` 主循环包 `with stage_context():`
- **冲烟**：再跑一次 shape.yaml，验证 base hook 加上后单 stage 行为 0 变化

### 阶段 3：trellis2 子类（改动点 5）
- todo `trellis2_trainer_init_primary_opt`：override `_init_optimizer` + `__init__` post-super 用 raw `torch.optim.AdamW` 建非 primary（**不调 `accelerator.prepare`**） + EMA fail-fast + `_validate_stage_schedulers` + `_upstream_stages / _inference_stages` 初始 init
- todo `trellis2_trainer_stage_context`：override `stage_context(stage=None)` reentrant + **try/finally exit restore**
- todo `trellis2_trainer_multi_opt_ckpt`：override `save_checkpoint` / `load_checkpoint`，分 `save_model_only` 双路径，by-name save/load `optimizer_stage_<name>.pt`，缺文件 fail-fast，**注意 final_dir 重算**
- todo `trellis2_trainer_eval_full_pipeline`：override `_extra_eval_inference_kwargs` 返回 full pipeline kwargs
- todo `trellis2_trainer_remove_sample_setup`：sample() 入口移除 `set_current_stage / set_seed`（改由 stage_context 统一管），`_rollout_group` 用 `self.adapter.current_stage`

### 阶段 4：yaml + 多 stage smoke
- todo `yaml_shape_tex`：新增 [examples/grpo/lora/trellis2/shape_tex.yaml](examples/grpo/lora/trellis2/shape_tex.yaml)（target list + 三 stage SDE + ema_decay=0 + adam_weight_decay 1e-4），register 进 examples 列表
- todo `smoke_single_stage_regression`：shape.yaml + tex_unified_reward.yaml 各自跑 5 epoch，确认 `_stage_optimizers` 只有 primary 一项 / stage_context 退化等价 / ratio_mean ≈ 1.0
- todo `smoke_multi_stage`：shape_tex.yaml 跑 10 epoch，log epoch 0/1 reward_mean、active stage、id(self.optimizer)、num_train_timesteps；ckpt 同时产出 LoRA 权重 + `optimizer.bin` + `optimizer_stage_tex.pt`
- todo `smoke_stage_context_restore`：epoch 0 末记录 id(self.optimizer)/current_stage/num_train_timesteps；epoch 1 顶部三字段必须等于 `__init__` 后的 primary 值（验证 exit restore 正确性）
- todo `smoke_resume`：从 epoch N 中断的 ckpt 恢复，确认 epoch N+1 切到正确 stage / primary + non-primary opt state bit-exact / 故意删除 `optimizer_stage_tex.pt` 后抛 FileNotFoundError / `save_model_only=True` 下 resume optimizer 全 reset

### 实施风险点（必须守住）
- 阶段 1 完成后必须先跑单 stage 回归 smoke，再进阶段 2，否则 bug 会层层传染
- 阶段 3 的 `stage_context` 必须 `try/finally` exit restore（critical），否则 multi-stage save_checkpoint 会静默污染 optimizer.bin
- 阶段 3 的 `_init_optimizer` 必须用 `get_stage_parameters(primary)` 而非 `get_trainable_parameters()`（union），否则 primary opt 会带上所有 stage params
- 阶段 3 的 non-primary opt **不能** 调 `self.accelerator.prepare(opt)`，否则与 by-name ckpt 冲突
- 阶段 3 的 `save_checkpoint` 子类 override 必须**重新算 final_dir**（base mutate 局部变量），不能依赖 super 的 save_directory 参数

### 实施前的 final check
- 重新通读 [plan 文档](/home/zhiyuan_ma/.cursor/plans/trellis2-multi-stage-alt-train_41eba9d6.plan.md) 完整 5.1-5.9 章节
- 确认 [shape_unified_reward.yaml](examples/grpo/lora/trellis2/shape_unified_reward.yaml) 的 reward 配置可移植到 shape_tex.yaml（shared reward 假设）
