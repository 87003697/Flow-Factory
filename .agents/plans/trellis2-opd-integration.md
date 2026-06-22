# Plan: Trellis2 OPD 集成（Phase 1 — Dense LoRA self-distill, base-as-teacher）

## 目标

为 Trellis2 dense stage 实现一个**独立的 OPD 蒸馏 trainer**，让 LoRA student 在多视角一致性约束下被训练：
- **Teacher** = pretrained base（LoRA 禁用，通过 `adapter.use_ref_parameters()` 进入）
- **Student** = pretrained base + LoRA（训练中，默认上下文）
- **Loss** = `0.5 * ||mu_S(x_j, c_ref) - mu_T(x_j, c_tgt)||^2 / σ̄²`，其中 c_tgt 是 student rollout 多视角渲染随机挑一帧（**B 方案：student-rollout 来源**）

**不复用 SD3.5 的 `DiffusionOPDTrainer`** — SD3.5 是多 teacher LoRA-checkpoint 蒸馏，trellis2 是 base-as-teacher cross-condition 蒸馏，两个算法形态不同。强行子类化会引入大量 mode dispatch / TeacherConfig 包袱。Trellis2 自己写一个干净的 trainer，把 SD3.5 OPDTrainer 留给 SD3.5。

**Phase 1 范围**：
- dense stage only（最简单、风险最低）
- LoRA-only（full-FT 路径走 `EMA snapshot` 逻辑相同但本期不验证）
- c_tgt = student rollout 渲染随机帧（B 方案）
- 不修 sparse scheduler B>1 bug，不动 `_distill` 的 dense 假设（Phase 2）

## 关键发现（Explore 阶段）

### 适配器原语已就绪

| 原语 | 行为 | Phase 1 用途 |
|------|------|-------------|
| `adapter.use_ref_parameters()` | LoRA → `disable_adapter()`；full-FT → EMA 恢复 init | **进入 teacher 上下文（PASS 1）** |
| 默认上下文 | LoRA enabled（base+LoRA） | **student 上下文（PASS 2）** |
| `adapter.forward(t, t_next, latents, next_latents, image_cond, ..., return_kwargs)` | 返回 `SDESchedulerOutput.next_latents_mean / std_dev_t / dt` | trainer 用它取 mu |
| `adapter.preprocess_func(images=List[List[PIL]])` | 返回 `{image_cond_512, image_cond_1024, ...}` | 把 c_tgt 视图编码成 image_cond |
| `adapter.render_latents(sample, num_frames, resolution, ...)` | sample.video ← (T,C,H,W) ∈ [0,1] | rollout 已自动调（trellis2_grpo 模板） |

### 已审计接口（A 路径）

- **A1（forward 接口）✅** — Trellis2Adapter.forward 通过 `**kwargs` 吞 `return_kwargs`；dense scheduler 默认 return_kwargs 已含 `next_latents_mean / std_dev_t / dt`
- **A2（disable_adapters 封装）✅** — `use_ref_parameters` 是 LoRA-disabled 上下文的官方原语，**正是 teacher 入口**

### Trellis2 mixin 复用点

- `Trellis2TrainerMixin._init_trellis2()` — stage detection、render_kwargs、DDP `_set_static_graph()`
- `Trellis2TrainerMixin._rollout_group()` — 跨 GPU 上游共享 + `adapter.inference()`，OOM 安全
- `Trellis2TrainerMixin._merge_batches() / _batches_to_merge` — group_contiguous 合并
- `Trellis2GRPOTrainer.sample()` — 一份 reference 实现，可复制改写（去掉 reward_buffer 即可）

### 算法决策记录

| 决策 | 选项 | 取值 | 备注 |
|------|------|------|------|
| Teacher 来源 | base / external_LoRA / EMA-snapshot | **base**（LoRA disabled） | 不需要 teacher 文件 |
| c_tgt 来源 | (A) base-rollout / (B) student-rollout / (C) dataset multi-view | **B** | on-policy；早期 mesh 差是已知风险 |
| Frame 选取策略 | 固定第 0 帧 / 随机帧 / 多帧 batch | **随机帧（每 epoch 重抽）** | 最简；多帧拼 batch 等 Phase 2 |
| Stage | dense / shape / tex | **dense** | 不踩 sparse `_distill` 坑 |
| Dynamics | ODE / Flow-SDE / CPS | **Flow-SDE 起步** | dense scheduler 全模式都返回三字段；Flow-SDE 跟 GRPO 配置一致 |
| Warmup | 早期用 dataset 多视角 → 切到 self-rollout | **不做** | 朴素版本先跑通看曲线 |

### Phase 1 风险（明确接受）

1. **早期训练 c_tgt 是垃圾视图** — student 还没训好，render 出的 mesh 质量低，teacher 在垃圾 c_tgt 上预测的 mu_T 会噪声大。决策：先跑朴素版本观察曲线，必要时 Phase 1.5 加 warmup
2. **dense stage 没有 mesh-level rendering** — `render_latents` 需要 shape + tex 全跑完才能渲染。**这是 trellis2_grpo 已经在用的路径**：训练 dense stage 时 `_rollout_group` 仍会调下游 shape/tex 拿到 mesh + render（grpo shape.yaml 里 `decode_output: true`）。Phase 1 沿用这个配置
3. **Sparse scheduler B>1 bug** 不阻塞 dense，留 Phase 2

## 主分支 vs worktree 现状（重要前置）

> ⚠️ Reviewer C1 已确认：本期工作的 base 是 `trellis2` 分支（主目录），**不是** `feat/diffusion-opd`。两条分支结构差异如下：

| 资产 | 主分支 `trellis2` | Worktree `feat/diffusion-opd` |
|------|------------------|------------------------------|
| `hparams/training_args` | **单文件** `training_args.py`（含 `_standardize_timestep_range` @ L393、`TrainingArguments` @ L104） | 已拆包：`training_args/{__init__.py, _base.py, opd.py}` |
| `SDESchedulerMixin.get_kl_divergence_denominator` | **不存在**（`scheduler/abc.py` 共 153 行） | 存在（`scheduler/abc.py:157-212`） |
| `resolve_distill_step_band` helper | **不存在** | 存在（`hparams/training_args/opd.py:32-`） |
| `trainers/opd/` SD3.5 OPDTrainer | **不存在** | 存在（515 LOC） |
| `trellis2/` 模型 + mixin | 存在 | **不存在** |

**结论**：本期需要从 worktree 移植 2 个 helper 到主分支（Step 0.5），然后在主分支落地新 trainer。**不要拆包 `training_args.py`**（YAGNI），新 args 直接放主分支单文件 `training_args.py` 里跟 GRPO/NFT/AWM/DPO/DGPO args 同级即可。SD3.5 OPDTrainer 不在本期范围（独立 trainer 路线，不依赖父类）。

## 相关代码

| 文件 | 函数/类 | 作用 / 改动 |
|------|---------|------------|
| `src/flow_factory/models/abc.py` | `BaseAdapter.use_ref_parameters` (L552-583) | **不动**，直接当 teacher ctx 用 |
| `src/flow_factory/models/trellis2/trellis2.py` | `Trellis2Adapter.forward` (L1139), `render_latents` (L2791), `preprocess_func` (L925) | **不动** |
| `src/flow_factory/trainers/trellis2_mixin.py` | `Trellis2TrainerMixin._init_trellis2`, `_rollout_group`, `_merge_batches`, `prepare_feedback` | **复用** |
| `src/flow_factory/trainers/trellis2_grpo.py` | `Trellis2GRPOTrainer.sample()` | **作为 sample() 的参考模板**（去 reward；改 `compute_log_prob=False`；trajectory_indices 用 `_select_train_step_indices` 而非 `scheduler.train_timesteps`） |
| `src/flow_factory/trainers/abc.py` | `BaseTrainer` | **父类**（不经过 SD3.5 OPDTrainer） |
| `src/flow_factory/utils/trajectory_collector.py` | `compute_trajectory_indices` | 用于 sample() 决定保存哪些 trajectory 位置 |
| `src/flow_factory/utils/base.py` | `filter_kwargs`, `create_generator` | utility |
| `src/flow_factory/trainers/registry.py` | `_TRAINER_REGISTRY` | 加一行 |
| `src/flow_factory/hparams/training_args.py` | `_standardize_timestep_range` (L393)、`TrainingArguments` (L104) | **复用**；新 `Trellis2OPDTrainingArguments` 直接加进同一文件 |
| `src/flow_factory/scheduler/abc.py` | `SDESchedulerMixin` (L43) | **Step 0.5** 需补 `get_kl_divergence_denominator` 方法（从 worktree 移植）|

新文件（3 个，**不再拆 training_args 包**）：
| 文件 | 操作 |
|------|------|
| `src/flow_factory/trainers/trellis2_opd.py` | 新建：独立 Trellis2OPDTrainer |
| `examples/opd/lora/trellis2/dense_self_distill.yaml` | 新建：最小可跑 yaml |
| `.agents/plans/trellis2-opd-integration.md` | 本文件 |

> Note：`Trellis2OPDTrainingArguments` 加进既有 `hparams/training_args.py`（不是新文件），跟 `GRPOTrainingArguments` 等并列。同时 `resolve_distill_step_band` helper 也加进同一文件。

## 实现步骤

### Step 0 — 分支准备
- [ ] 在主目录从 `trellis2` 分支切出 `feat/trellis2-opd`
- [ ] worktree `opd-baseline-test` 留作 SD3.5 OPD 参考代码源（不删）

### Step 1 — Args 数据类（~40 LOC）
- [ ] 新建 `src/flow_factory/hparams/training_args/trellis2_opd.py`
- [ ] 加入 `Trellis2OPDTrainingArguments` 数据类，含三字段：
  - `timestep_range: Union[float, Tuple[float, float]] = 0.99`
  - `num_inner_epochs: int = 1`
  - `teacher_frame_strategy: Literal["random"] = "random"`（Phase 1 锁死，留扩展位）
- [ ] `__post_init__`: 调 `_standardize_timestep_range` 把 float 标准化成 tuple
- [ ] 加 `get_num_train_timesteps(args)` override，返回 `hi - lo`（让 gradient_accumulation 数学正确，模仿 SD3.5 OPD args）
- [ ] 在 `hparams/training_args/__init__.py` export
- [ ] 在 `hparams/__init__.py` export

### Step 2 — Trainer 主体（~280 LOC）

新建 `src/flow_factory/trainers/trellis2_opd.py`：

- [ ] `__init__(**kwargs)`: super 后调 `_init_trellis2()`；初始化 `self._is_sde = scheduler.dynamics_type != "ODE"`、`self._student_noise_level = scheduler.noise_level if SDE else 0.0`、`self._mu_store_device`
- [ ] `start()`: 标准 epoch loop（save → eval → sample → optimize → ema_step）
- [ ] `sample()`: 复制 `Trellis2GRPOTrainer.sample()` 移除 reward_buffer 部分；trajectory_indices 来自 `_select_train_step_indices`
- [ ] `_select_train_step_indices(num_inference_steps, timestep_range)`: 解析 `timestep_range` 成 `[lo, hi)`，返回 `torch.arange(lo, hi)`
- [ ] `prepare_feedback(samples)`: 走父类 mixin 的空 sample guard（OOM 全跳兜底）；不算 reward
- [ ] `optimize(samples)`:
  1. 空 samples guard
  2. `_compute_c_tgt(samples)` — 从每个 sample 的 video 随机抽帧 → encode → `sample.extra_kwargs["image_cond_tgt"]`
  3. `_precompute_mu_T(samples, train_timesteps)` — PASS 1
  4. `_distill(samples, train_timesteps)` — PASS 2
- [ ] `_compute_c_tgt(samples)` (`@torch.no_grad()`):
  - per-epoch + per-sample 决定性 RNG（用 `create_generator(seed, epoch, sample.unique_id)`）
  - 取 `s.video[frame_idx]` (C,H,W) ∈ [0,1] → `to_pil_image(frame)`
  - `adapter.preprocess_func(images=[[pil]])` → `image_cond_512[0]`
  - 写入 `s.extra_kwargs["image_cond_tgt"]`（dense stage 用 512）
  - sample 端要在 `Trellis2Sample._extra_kwargs_keys` 已有 `image_cond_tgt`？— 确认下，可能需要加
- [ ] `_precompute_mu_T(samples, train_timesteps)` (`@torch.no_grad()`):
  - 进入 `with self.adapter.use_ref_parameters():` → 进入 `with self.autocast():`
  - 对每个 batch（per_device_batch_size 切分），构造 `cond_override = {"image_cond": batch["image_cond_tgt"]}`
  - 对每个 train_timestep idx，调 `_forward_step(batch, ts_idx, cond_override)` 拿 `mu_T`，stack 后 `s.extra_kwargs["mu_teacher"] = ...`（搬到 `_mu_store_device`）
  - 退出 `use_ref_parameters` 后 `torch.clear_autocast_cache()`
- [ ] `_distill(samples, train_timesteps)`:
  - 标准 inner-epoch + shuffle + per_device_batch_size 切批
  - LoRA enabled 默认 → student
  - 对每个 train_timestep idx，调 `_forward_step(batch, ts_idx, cond_override=None)`（用原 batch["image_cond"]）
  - 取 `mu_S, std_dev_t, dt`；从 batch 取 cached `mu_teacher[:, idx]` → MSE
  - `denom = scheduler.get_kl_divergence_denominator(std_dev_t, dt)` → KL
  - `accelerator.backward(loss)`；sync_gradients 时 clip + step + log
- [ ] `_forward_step(batch, timestep_index, cond_override=None)`:
  - 抄 SD3.5 OPDTrainer 的实现（trainer.py:465-515）；`cond_override` 直接 update 进 `forward_inputs`（不用碰父类）
  - 调 `self.adapter.forward(...)` 拿 output；validate `next_latents_mean is not None`
- [ ] `_log_metrics(kl_sum, kl_count, grad_norm)`: 单 teacher，无需多 teacher 路由；log `train/kl_div / grad_norm`
- [ ] 顶部 `@register_trainer("trellis2_opd")` 装饰

### Step 3 — Sample 字段（如需）
- [ ] 检查 `Trellis2Sample._extra_kwargs_keys` 是否含 `image_cond_tgt` 和 `mu_teacher`
- [ ] 若无，加进去（让 BaseSample.stack / .to(device) / extra_kwargs 序列化能正确处理）

### Step 4 — Registry 注册（1 LOC）
- [ ] `src/flow_factory/trainers/registry.py:_TRAINER_REGISTRY` 加 `"trellis2_opd": "flow_factory.trainers.trellis2_opd.Trellis2OPDTrainer"`

### Step 5 — YAML 模板（~80 LOC）
新建 `examples/opd/lora/trellis2/dense_self_distill.yaml`：
- [ ] 起点：拷 `examples/grpo/lora/trellis2/shape.yaml`
- [ ] 改 `target_flow_model: dense_*`
- [ ] 改 `trainer_type: trellis2_opd`
- [ ] 删 GRPO-only 字段（`advantage_aggregation` / `clip_range` / `adv_clip_range` / `kl_type` / `kl_beta` / `ref_param_device`）
- [ ] 加 OPD 字段：`timestep_range: 0.99` / `num_inner_epochs: 1` / `teacher_frame_strategy: random`
- [ ] `scheduler.dynamics_type: Flow-SDE`，`noise_level: 1.0`，`num_sde_steps: 1`
- [ ] `rewards: []` 留空（OPD 不需要 reward）
- [ ] `eval_rewards`: 保留 PickScore 一个用于监控

### Step 6 — Smoke test
- [ ] 不实际启动训练，只跑 `python -c "from flow_factory.trainers.trellis2_opd import Trellis2OPDTrainer"` 验证 import OK
- [ ] 跑 `flow-factory --config dense_self_distill.yaml` dry-run，确认 trainer 能初始化、`sample()` 第一个 batch 出 video，`optimize()` 第一个 micro-batch backward 不崩
- [ ] 验证 `mu_T` 和 `image_cond_tgt` 出现在 sample.extra_kwargs

## 代码变更预览

### 新文件 1：`hparams/training_args/trellis2_opd.py`

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal, Tuple, Union

from ._base import TrainingArguments, _standardize_timestep_range
from .opd import resolve_distill_step_band  # reuse helper from SD3.5 OPD args


@dataclass
class Trellis2OPDTrainingArguments(TrainingArguments):
    """OPD distillation for Trellis2: base (teacher) → LoRA (student).

    Teacher = pretrained base (LoRA disabled via `adapter.use_ref_parameters`).
    Student = base + LoRA (default ctx).
    c_tgt = student rollout multi-view render → random frame → re-encoded.

    Loss: `0.5 * ||mu_S(x_j, c_ref) - mu_T(x_j, c_tgt)||^2 / sigma_bar^2`.
    """
    timestep_range: Union[float, Tuple[float, float]] = field(default=0.99,
        metadata={"help": "Fraction band of denoising steps to distill (1000->0)."})
    num_inner_epochs: int = field(default=1,
        metadata={"help": "Reuse epochs over the on-policy trajectories."})
    teacher_frame_strategy: Literal["random"] = field(default="random",
        metadata={"help": "Phase 1: 'random' — pick one frame per sample per epoch."})

    def __post_init__(self):
        super().__post_init__()
        self.timestep_range = _standardize_timestep_range(self.timestep_range)

    def get_num_train_timesteps(self, args: Any) -> int:
        lo, hi = resolve_distill_step_band(self.num_inference_steps, self.timestep_range)
        return hi - lo
```

### 新文件 2：`trainers/trellis2_opd.py`（骨架）

```python
@register_trainer("trellis2_opd")
class Trellis2OPDTrainer(Trellis2TrainerMixin, BaseTrainer):
    """OPD self-distillation: pretrained base (teacher) → LoRA (student).

    PASS 1 (no_grad, base via use_ref_parameters):
        mu_T_j = base.forward(x_j, t_j, image_cond=c_tgt)
    PASS 2 (gradient, LoRA enabled):
        mu_S_j = (base+LoRA).forward(x_j, t_j, image_cond=c_ref)
        loss   = 0.5 * ||mu_S_j - mu_T_j||^2 / denom_j

    c_tgt = student rollout's multi-view render → random frame → re-encoded.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.training_args: Trellis2OPDTrainingArguments
        self._init_trellis2()
        scheduler = self.adapter.scheduler
        self._is_sde = scheduler.dynamics_type != "ODE"
        self._student_noise_level = float(scheduler.noise_level) if self._is_sde else 0.0
        self._mu_store_device = "cpu" if self.training_args.offload_samples_to_cpu else self.accelerator.device
        logger.info(f"Trellis2OPDTrainer init: stage={self._training_stage}, "
                    f"dynamics={scheduler.dynamics_type}, frame_strategy={self.training_args.teacher_frame_strategy}")

    # === lifecycle ===
    def start(self): ...                                      # 标准 epoch loop
    def sample(self) -> List[BaseSample]: ...                 # 复制 Trellis2GRPOTrainer.sample(), 删 reward
    def prepare_feedback(self, samples): super().prepare_feedback(samples)
    def optimize(self, samples) -> None:
        if not samples:
            return
        self.adapter.train()
        train_ts = self._select_train_step_indices(...)
        self._compute_c_tgt(samples)
        self._precompute_mu_T(samples, train_ts)
        self._distill(samples, train_ts)

    # === passes ===
    @torch.no_grad()
    def _compute_c_tgt(self, samples): ...                    # video[idx] → preprocess_func → image_cond_tgt

    @torch.no_grad()
    def _precompute_mu_T(self, samples, train_timesteps):
        with self.adapter.use_ref_parameters():               # ← LoRA OFF
            with self.autocast():
                for batch in batches:
                    cond_override = {"image_cond": batch["image_cond_tgt"]}
                    mu_T_steps = [self._forward_step(batch, t, cond_override=cond_override)[0]
                                  for t in train_timesteps]
                    # cache mu_T to sample.extra_kwargs
        torch.clear_autocast_cache()

    def _distill(self, samples, train_timesteps):
        # LoRA enabled (default) → student
        with self.autocast():
            for batch in shuffled_batches:
                for ts_idx in train_timesteps:
                    mu_S, std_dev_t, dt = self._forward_step(batch, ts_idx)  # uses batch["image_cond"] = c_ref
                    mu_T = batch["mu_teacher"][:, ts_idx]
                    mse  = (mu_S.float() - mu_T.float()).pow(2).flatten(1).mean(1)
                    denom = self.adapter.scheduler.get_kl_divergence_denominator(std_dev_t, dt)
                    loss = 0.5 * (mse / denom).mean()
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        ...   # clip + step + log

    # === forward step (SD3.5 OPD 抄过来) ===
    def _forward_step(self, batch, timestep_index, cond_override=None):
        forward_inputs = {**self.training_args, **batch, "t": ..., "t_next": ..., ...}
        if cond_override:
            forward_inputs.update(cond_override)
        forward_inputs = filter_kwargs(self.adapter.forward, **forward_inputs)
        forward_inputs["return_kwargs"] = ["next_latents_mean", "std_dev_t", "dt"]
        out = self.adapter.forward(**forward_inputs)
        return out.next_latents_mean, out.std_dev_t, out.dt
```

### 新文件 3：`examples/opd/lora/trellis2/dense_self_distill.yaml`（关键 diff）

```yaml
# 拷 examples/grpo/lora/trellis2/shape.yaml 改：
model:
  target_flow_model: "dense"
  decode_output: true        # 必须，c_tgt 需要渲染回视图
  render_num_frames: 24
  render_resolution: 512
  render_mode: "shaded"      # 用 PBR shaded（teacher 看完整外观）
  ...

train:
  trainer_type: 'trellis2_opd'
  # OPD-specific
  timestep_range: 0.99
  num_inner_epochs: 1
  teacher_frame_strategy: 'random'
  # 删除: advantage_aggregation, clip_range, adv_clip_range, kl_type, kl_beta, ref_param_device

scheduler:
  dynamics_type: "Flow-SDE"
  noise_level: 1.0
  num_sde_steps: 1
  ...

rewards: []  # OPD 不需要 reward (loss 不依赖)
eval_rewards:
  - name: "pickscore"  # 仅监控
    ...
```

### 修改文件：`trainers/registry.py`

```diff
 _TRAINER_REGISTRY: Dict[str, str] = {
     ...
     "trellis2_grpo": "flow_factory.trainers.trellis2_grpo.Trellis2GRPOTrainer",
     "trellis2_nft": "flow_factory.trainers.trellis2_nft.Trellis2NFTTrainer",
+    "trellis2_opd": "flow_factory.trainers.trellis2_opd.Trellis2OPDTrainer",
     ...
 }
```

## 方案对比

### 已选定（无需再讨论）

| 维度 | 选项 | 取值 | 取胜理由 |
|------|------|------|----------|
| Trainer 继承 | A. 子类 SD3.5 OPDTrainer / B. **独立 trainer** | **B** | SD3.5 是多 teacher LoRA-checkpoint 蒸馏，trellis2 是 base-as-teacher cross-condition；强行子类化要写大量 mode dispatch + TeacherConfig 改动；独立 trainer 净代码更少更清晰 |
| Teacher 来源 | base / external_LoRA / EMA-snapshot | **base** | self-distill 研究目标，无需任何 teacher 文件 |
| c_tgt 来源 | A. base-rollout / **B. student-rollout** / C. dataset multi-view | **B**（用户决策） | on-policy 信号，最贴近研究目标 |
| Frame 策略 | 固定 0 帧 / **随机帧** / 多帧 batch | **随机帧** | 最简实现；多帧拼 batch 等 Phase 2 |
| Stage | dense / shape / tex | **dense** | 不踩 sparse `_distill` 坑（L349/L319-324/L265） |
| Dynamics | ODE / **Flow-SDE** / CPS | **Flow-SDE** | 跟 GRPO 配置一致，dense scheduler 全 mode 都返回三字段 |

### Phase 1 vs Phase 2

| 工作项 | Phase 1 | Phase 2 |
|--------|---------|---------|
| dense LoRA self-distill | ✅ | — |
| sparse scheduler scatter-mean (修 B>1 bug) | — | ✅ |
| `_distill` dense 假设 redesign（如复制到 trellis2_opd 的等效逻辑要适配 sparse SparseTensor） | — | ✅ |
| shape / tex stage 训练支持 | — | ✅ |
| c_tgt 多视角拼 batch | — | ✅ |
| Warmup 切换（dataset multi-view → student-rollout） | — | 可选 |
| full-FT teacher（EMA-snapshot 路径） | — | 可选 |

## 状态

**当前阶段**: Planning（等用户确认）

确认后进入 Code 阶段，按 Step 0 → Step 6 顺序实现，每个 Step 完成后小 commit + 勾选 checkbox。

## Review

> **Steelman**: 方案最强之处在于：基于对 Trellis2 adapter / mixin / sample 体系的深度审计，做出了"独立 trainer 而非子类化 SD3.5 OPDTrainer"的正确架构决策，并给出了可直接编码的代码预览（含完整的 PASS 1 / PASS 2 分离、cond_override 机制、forward_step 签名）。每个设计选项都有决策记录和取舍说明，这比大多数 plan 做得好。
>
> **Pre-mortem** (6 个月后方案彻底失败的 3 个最可能原因):
> 1. OPD 基础设施（`get_kl_divergence_denominator`、`resolve_distill_step_band`）一直没从 worktree 合并到主分支，trainer 在主分支跑不起来，变成孤岛代码
> 2. `mu_T` 全 timestep 缓存的 CPU 内存开销在 8 卡 × 64 step × dense 5D latent 下爆内存，逼出一轮紧急重构
> 3. student rollout 早期渲染质量太低（方案"风险 1"），`mu_T` 在垃圾 c_tgt 上发散，没留 warmup fallback 导致多轮无效实验

### 🔴 Critical

#### C1. OPD 基础设施在主分支不存在——方案未标注前置合并步骤

- **位置**：§代码变更预览 · `hparams/training_args/trellis2_opd.py`（L169 `from .opd import resolve_distill_step_band`）、§Step 2 · `_distill`（L128 `scheduler.get_kl_divergence_denominator(std_dev_t, dt)`）
- **为什么**：经源码验证，`resolve_distill_step_band` 和 `get_kl_divergence_denominator` 仅存在于 `.claude/worktrees/opd-baseline-test/`，**主分支 `src/` 无此代码**。主分支的 `SDESchedulerMixin`（abc.py:154 行结束）没有 `get_kl_divergence_denominator`；主分支的 `hparams/training_args.py` 是单体文件，无 `opd.py` 子模块。方案 Step 0 只写了"切分支"，没有"合并 OPD 基础设施"或"cherry-pick 到 trellis2 分支"这一前置步骤。如果直接按 Step 1→Step 2 实现，import 就会报错。
- **修正**：在 Step 0 和 Step 1 之间加 **Step 0.5 — 合并 OPD 基础设施**，明确列出需要从 worktree 移植的文件/方法：(a) `SDESchedulerMixin.get_kl_divergence_denominator`、(b) `resolve_distill_step_band` helper、(c) `hparams/training_args/` 拆分为包结构（如果尚未完成）。或者退而求其次：在 `trellis2_opd.py` 内联一个极简版 `_kl_denom()` + `_resolve_band()`，标注 `# TODO: unify with opd.py after merge`。

#### C2. `_extra_kwargs_keys` 机制不存在——Step 3 基于错误假设

- **位置**：§Step 3 — Sample 字段（L137–138），"检查 `Trellis2Sample._extra_kwargs_keys` 是否含 `image_cond_tgt` 和 `mu_teacher`"
- **为什么**：全代码库搜索 `_extra_kwargs_keys`，**无任何定义或使用**。`BaseSample.extra_kwargs` 是一个自由 dict（`samples.py:115`），写入即可用，`__getattr__` 会自动代理查找（`samples.py:198-207`）。`to_dict()` 会把 `extra_kwargs` 展平到顶层（`samples.py:176-177`），所以 batch dict 里 `batch["mu_teacher"]` 能直接取到。但 **`BaseSample.to(device)` 不会自动迁移 `extra_kwargs` 内的 tensor**——SD3.5 OPD worktree 里有显式 `.to(device)` 逻辑（trainer.py:317-325）。方案既没提这个迁移需求，Step 3 还在找一个不存在的接口。
- **修正**：删除 Step 3 对 `_extra_kwargs_keys` 的检查。改为：(a) 在 `_compute_c_tgt` 和 `_precompute_mu_T` 的代码预览中标注 `extra_kwargs` tensor 的 device 管理策略；(b) 在 `_distill` 开头加显式 `.to(device)` 步骤，把 `mu_teacher` 从 `_mu_store_device` 搬回 GPU。

### 🟡 Warning

#### W1. `return_kwargs` 被 `**kwargs` 吞掉——代码预览有误导

- **位置**：§已审计接口 A1（L32），"通过 `**kwargs` 吞 `return_kwargs`"；§Step 2 `_forward_step`（L273），`forward_inputs["return_kwargs"] = ["next_latents_mean", "std_dev_t", "dt"]`
- **为什么**：`Trellis2Adapter.forward()` 签名有 `**kwargs`（L1153），但 dense 分支的 `scheduler.step()` 调用（L1227-1235）**从未转发 kwargs**。`return_kwargs` 被吞且不生效。Dense scheduler 之所以能返回三字段是因为 `FlowMatchEulerDiscreteSDEScheduler.step()` 的默认参数值已包含它们（scheduler L254）。方案的"审计"声称 A1 通过，但实际上 `return_kwargs` 是死代码。在 `_forward_step` 里显式 set 它会给读者造成"这行起作用"的假象。
- **修正**：(a) 在"已审计接口"表更正 A1 为"kwargs 被吞但不转发，dense 默认全返回所以无影响"；(b) `_forward_step` 代码预览中删除 `forward_inputs["return_kwargs"] = ...` 行，改为注释说明 dense scheduler 默认返回三字段，或如果想显式控制，需先在 `Trellis2Adapter.forward()` dense 分支的 `scheduler.step()` 调用中加 `**kwargs` 转发。

#### W2. `mu_T` 全 timestep 缓存的内存估算缺失

- **位置**：§Step 2 `_precompute_mu_T`（L118-122），"stack 后 `s.extra_kwargs["mu_teacher"]`（搬到 `_mu_store_device`）"
- **为什么**：Dense stage 默认 64 推理步。`timestep_range: 0.99` → 约 63 个训练步（`resolve_distill_step_band(64, 0.99) = (0, 63)`）。每个 mu_T 形状同 dense latent `(C, D, H, W)`——假设 `(32, 16, 16, 16)` fp32 ≈ 512KB。63 步 × 8 samples/device × 512KB ≈ 252MB/GPU。看起来不大，但如果 group_size 更大或 resolution 更高，CPU 侧也会线性增长。方案提了 `_mu_store_device` 但没给估算，无法判断是否安全。
- **修正**：在 §Phase 1 风险 或 §Step 2 加一段内存估算（插入 dense latent shape × timestep 数 × batch_size 的公式），给出"OK / 需要分段计算"的结论。如果 per-step forward + 即时 loss（而非全量缓存）可行，标注为 Phase 1.5 的优化选项。

#### W3. `_compute_c_tgt` 的 `preprocess_func` 逐 sample 调用——性能陷阱

- **位置**：§Step 2 `_compute_c_tgt`（L112-116），"per-epoch + per-sample 决定性 RNG … `adapter.preprocess_func(images=[[pil]])` → `image_cond_512[0]`"
- **为什么**：`preprocess_func`（trellis2.py:925-988）内部跑 DINOv2 + cross-attention projection，是 GPU 密集操作。逐 sample 调用 = batch_size=1 的 DINOv2 forward，效率极低（实际训练中 batch 内帧可以拼一起编码）。而且 `preprocess_func` 返回 512 + 1024 两个 resolution 的 cond，dense 只用 512，另一半白算。
- **修正**：在 `_compute_c_tgt` 中先收集所有 sample 的目标帧到一个 batch list `[[pil_1], [pil_2], ...]`，一次调用 `preprocess_func(images=batch_list)` 得到 batch 结果，再按 sample 切分。这是 O(1) GPU forward 对比 O(N) 的差距。标注到代码预览中。

#### W4. `sample.extra_kwargs` 被 inference 整体覆盖的风险

- **位置**：§Step 2 `optimize()` 流程（L107-111），"1. 空 samples guard → 2. `_compute_c_tgt` → 3. `_precompute_mu_T` → 4. `_distill`"
- **为什么**：`_inference_dense`（trellis2.py:2170）对训练 stage 执行 `sample.extra_kwargs = {callback_index_map: ..., **extra_callback_res}`——这是 **赋值而非合并**。当前 optimize 流程中 `_compute_c_tgt` 在 rollout 之后执行（写 `image_cond_tgt` 到已初始化的 `extra_kwargs`），所以不受影响。但如果未来有人把 `_compute_c_tgt` 提到 `sample()` 里（例如为了 warmup），或者在 optimize 中再次调 rollout，`extra_kwargs` 会被 wipe。方案没提及这个 ordering invariant。
- **修正**：在 `_compute_c_tgt` 代码预览上方加注释 `# Must run AFTER sample() — inference overwrites extra_kwargs`。或者更稳妥：让 `_compute_c_tgt` 写入 Trellis2Sample 的显式字段而非 `extra_kwargs`。

### 🔵 Suggestion

| 位置 | 问题 | 建议 |
|------|------|------|
| §Step 1 L93 `__post_init__` | `timestep_range` 标准化直接调 `_standardize_timestep_range`，但主分支路径是 `hparams/training_args.py` 非 `_base.py` | 修正 import 路径或标注"如果 hparams 已拆包则从 `_base` 导入" |
| §Step 2 L104 `sample()` | "复制 Trellis2GRPOTrainer.sample() 移除 reward_buffer"——实际还需改 `compute_log_prob=False`、替换 `trajectory_indices` 逻辑（用 `resolve_distill_step_band` 而非 `scheduler.train_timesteps`） | 在 Step 2 `sample()` 条目下列出与 GRPO 的 3 个具体差异 |
| §Step 5 YAML L151 | `rewards: []` 但 BaseTrainer 可能要求至少一个 reward 才能初始化 | 验证空 rewards 不会触发 BaseTrainer init 报错（`(需 verify)`）|
| §代码预览 `_distill` L260 | `mse = (mu_S.float() - mu_T.float()).pow(2).flatten(1).mean(1)` | dense latent 是 5D `(C,D,H,W)` 不是 2D，`flatten(1)` 对 4 维 tensor 需确认语义正确（应该没问题但值得标注） |
| §Step 6 smoke test L156 | "跑 `flow-factory --config` dry-run" | CLI 入口是 `ff-train` 不是 `flow-factory` |

### ✅ 做得好的

1. **独立 trainer 的架构决策充分论证** — 没有贪图代码复用而强行子类化 SD3.5 OPDTrainer，方案对比表（§方案对比）清晰列出了取胜理由。这避免了 mode dispatch 的复杂度负担。
2. **适配器原语审计表** — §适配器原语已就绪 的四行表比"我看过代码了"有说服力得多，直接映射原语→Phase 1 用途。
3. **Phase 1 vs Phase 2 切割明确** — 不修 sparse scheduler B>1 bug、不做 warmup、不动 full-FT 路径，每条都标了归属，避免 scope creep。
4. **代码预览详细到可编码** — `_forward_step` 的参数名、`cond_override` 机制、`denom = scheduler.get_kl_divergence_denominator(...)` 调用链，节省了实现者 80% 的"怎么拼"的思考时间。
5. **风险决策记录** — §算法决策记录 + §Phase 1 风险 把"已知风险 + 决策 + fallback"一起写了，这是多数 plan 跳过的部分。

### TL;DR

方案的架构方向和算法设计是对的，但**缺了一步关键前置工作**：OPD 基础设施（`get_kl_divergence_denominator` + `resolve_distill_step_band`）在主分支不存在，Step 0 和 Step 1 之间需要插入合并/移植步骤。最该补的一条是 **C1 — 标注 OPD 基础设施的合并路径**，否则第一个 import 就会失败。

---

### Review · 2026-06-02 · 二轮（C1 真因更正）

> **触发**：用户问"flowfactory 是不是跟上游 main 没同步"。`git log origin/main..trellis2 / trellis2..origin/main` 验证后发现：**trellis2 是 fork 上的功能分支，落后 `origin/main` 18 个 commit**，包括 PR #170 (整个 DiffusionOPD trainer)、PR #168 (多数据集 + per-source reward)、PR #163 (`hparams/training_args.py` 拆包)。

#### 修正：C1 真因不是"基础设施未合并"，而是"trellis2 分支没 sync 到 main"

- **原 C1 假设**："基础设施仅在 worktree 上游 `feat/diffusion-opd` 存在，需要从 worktree 移植到主分支"
- **实际真相**：所有 OPD 基础设施已经在 `origin/main`（PR #170 已合）。worktree 的 `feat/diffusion-opd` 跟 `origin/main` 内容基本一致。**问题在 trellis2 没同步 main**
- **影响范围扩大**：不只是 `get_kl_divergence_denominator` + `resolve_distill_step_band` 两个 helper，而是 **18 个 commit 整体** 包括：
  - `hparams/training_args.py` 已拆成 `training_args/{__init__.py, _base.py, _registry.py, opd.py, ...}` 包
  - 多数据集架构 (`dataset_args.py`, `data_utils/multi_source.py`)
  - GenEval reward 集成
  - 统一 trainer sampling pipeline
  - HF checkpoint resume

#### 新 Step 0（替代旧 Step 0 + 旧 Step 0.5）：sync trellis2 ← origin/main

- [ ] 评估 merge 冲突范围：`git merge-tree $(git merge-base trellis2 origin/main) trellis2 origin/main | head -200`
- [ ] 在 worktree 里跑 sync 试错（`git worktree add .claude/worktrees/sync-test -b sync-test trellis2`，merge 完看冲突），不污染 trellis2 主线
- [ ] 主要冲突预测点：
  1. `hparams/` 单文件→包结构（trellis2 的 NFT trainer 可能 import 单文件路径）
  2. `trainers/abc.py` 统一 sampling pipeline 跟 trellis2_mixin 的 `_rollout_group` 设计
  3. `examples/` 目录结构调整
  4. trellis2 的 FlowEdit / OOM-safe / clay/mask reward 这些 trellis2 本地 commit 跟 main 的 reward 重构怎么对接
- [ ] 解冲突 + smoke test（NFT/GRPO trainer 在 sync 后还能跑）
- [ ] 提 PR `feat/sync-trellis2-from-main`，merge 完后再开始 OPD 集成

完成后：
- ✅ OPD 基础设施直接可用（不需要任何手动移植）
- ✅ `hparams/training_args/` 拆包后跟 worktree 一致，`Trellis2OPDTrainingArguments` 该放 `hparams/training_args/trellis2_opd.py`（**回到原 plan 路径**，不是单文件）
- ✅ 多数据集架构可用（Phase 2 多视角 batch 拼接的潜在基础设施）
- ✅ 之后 trellis2 跟 main 不再分叉

#### 修正：相关代码表 import 路径

> sync 完成后，主分支 = `origin/main` 结构，hparams 是包结构

| 文件 | 状态变化 |
|------|---------|
| `src/flow_factory/hparams/training_args.py` | sync 后**不存在**（已拆包） |
| `src/flow_factory/hparams/training_args/_base.py` | sync 后存在；含 `_standardize_timestep_range` + `TrainingArguments` |
| `src/flow_factory/hparams/training_args/opd.py` | sync 后存在；含 `resolve_distill_step_band` + SD3.5 OPD args |
| `src/flow_factory/scheduler/abc.py:get_kl_divergence_denominator` | sync 后存在 |
| `src/flow_factory/trainers/opd/{__init__,common,trainer}.py` | sync 后存在（SD3.5 OPDTrainer），**不依赖**它（独立 trainer 路线） |

所以新文件清单回到原始版本（4 个）：
1. `src/flow_factory/hparams/training_args/trellis2_opd.py` — 极简 args（**重新启用**）
2. `src/flow_factory/trainers/trellis2_opd.py` — 独立 trainer
3. `examples/opd/lora/trellis2/dense_self_distill.yaml` — yaml
4. `.agents/plans/trellis2-opd-integration.md` — 本文件

#### 替代方案（如果 sync 风险太高）

| 选项 | 说明 | 决策 |
|------|------|------|
| **A. 先 sync trellis2 ← origin/main**（推荐） | 从根本解决分叉，OPD 全套基础设施直接获得 | 默认走这条 |
| B. 不 sync，按当前 trellis2 单文件结构做 OPD（原 C1 修正路径：手动移植 2 个 helper 到单文件） | 最小动作 | trellis2 跟 main 越来越分叉，是技术债加重 |
| C. cherry-pick 必要 commits（PR #170 + 它的依赖）到 trellis2 | 中等动作，定向修复 | 容易遗漏依赖（PR #170 依赖 PR #168 的多数据集架构） |

#### 其他 review 项不受影响（C2 / W1-W4 / Suggestions 仍有效）

C2（`_extra_kwargs_keys` 不存在）、W1（`return_kwargs` 被吞）、W2（mu_T 内存估算）、W3（`preprocess_func` batch 化）、W4（`extra_kwargs` 覆盖风险）这些 review 项跟 sync 状态无关，sync 完成后仍要在 trainer 实现时处理。

#### 新增风险（sync 引入）

1. **Merge 冲突修复时间不可估** — 18 commit 跨 hparams 拆包 + trainer 重构，trellis2 本地 commit 改过的文件（NFT trainer / FlowEdit / clay reward）大概率冲突
2. **trellis2 现有 GRPO/NFT 训练在 sync 后的回归风险** — 既有训练曲线/yaml 可能受 hparams 拆包 / trainer pipeline 重构影响。需要 sync PR 自带 smoke test
3. **Sync PR 的审核周期** — 如果走 fork 上提 PR 流程，可能阻塞 OPD 落地。本期 OPD 实现可以在 sync 分支上并行写代码，等 sync merge 完再 rebase 到 trellis2

### 二轮 TL;DR

C1 的真因是 **trellis2 分支 18 commit 落后 origin/main**，不是基础设施未合并。**最该补的是 Step 0 改成 "sync trellis2 ← origin/main"**，sync 后 OPD 基础设施 + hparams 拆包结构直接获得，本期工作回到 4 个新文件的原始路径。

---

### Review · 2026-06-08 · 三轮（Error #6: gradient checkpointing + inplace FFT）

> **触发**：Koala job `ericzyma-job-normal-20260608-151217` 在 `_distill` backward 崩溃。训练已跑过 eval（pickscore=0.7242）、sampling（4 windows）、`_compute_c_tgt`、`_precompute_mu_T`（100% teacher targets in 8s），在 `_distill` 的 `accelerator.backward(loss)` 处失败。

#### 根因

Dense transformer (`sparse_structure_flow_model`) 的 TRELLIS.2 后端包含 **inplace FFT 操作**。当 `enable_gradient_checkpointing: true` 时，backward pass 尝试 **recompute** forward，但 inplace 操作已改变原始 tensor，recompute 看到被污染的输入后崩溃。

这是 gradient checkpointing 的已知限制：checkpoint 机制释放中间 activation → backward 时重新 forward → 如果 forward 中有 inplace 操作，重新 forward 的输入已被修改 → 结果不一致 → 报错。

#### 各 stage 情况

| Stage | Model | Block 类型 | 有 inplace FFT？ | Grad ckpt 冲突？ |
|-------|-------|-----------|-----------------|-----------------|
| Dense | `sparse_structure_flow_model` | Standard transformer | **是** | **是** — backward recompute 时崩溃 |
| Shape | `shape_slat_flow_model_{512,1024}` | `ModulatedSparseTransformerCrossBlock` | **否** | 否 — sparse attention, 无 FFT |
| Tex | `tex_slat_flow_model_{512,1024}` | `ModulatedSparseTransformerCrossBlock` | **否** | 否 — sparse attention, 无 FFT |

#### 设计：YAML config flag + `_set_transformer_checkpoint()` in mixin

**思路**：是否在 distill 时 disable gradient checkpointing 由 **config 控制**。Dense stage 需要 disable（inplace FFT 冲突），shape/tex 不需要。各 stage 的 YAML 自行决定。

**1. 新增 training arg 字段**（`Trellis2OPDTrainingArguments`）：

```python
disable_grad_checkpoint_for_distill: bool = field(
    default=False,
    metadata={"help": "Disable gradient checkpointing during _distill backward. "
              "Required for dense stage (inplace FFT ops conflict with recompute)."},
)
```

**2. Mixin helper**（`Trellis2TrainerMixin`）：

```python
def _set_transformer_checkpoint(self, enabled: bool) -> None:
    """Toggle gradient checkpointing on the training-stage transformer."""
    model = self.adapter.pipeline.transformer
    if not hasattr(model, "blocks"):
        return
    for block in model.blocks:
        if hasattr(block, "use_checkpoint"):
            block.use_checkpoint = enabled
```

**3. 调用点**（`trellis2_opd.py:optimize()`）：

```python
if self.training_args.disable_grad_checkpoint_for_distill:
    self._set_transformer_checkpoint(False)
self._distill(samples, train_timesteps)
if self.training_args.disable_grad_checkpoint_for_distill:
    self._set_transformer_checkpoint(True)
```

**4. YAML 配置**：
- `dense_self_distill.yaml`: `disable_grad_checkpoint_for_distill: true`
- 未来 shape/tex yaml: 不设或 `false`（默认值）

**为什么这样设计：**
- Dense 有 inplace FFT 冲突，必须 disable → config 设 true
- Shape/tex 无此问题，保持 checkpointing 节省显存 → 默认 false
- 未来如果其他 stage 也出现类似问题，yaml 里开一个 flag 即可，不用改代码
- `_set_transformer_checkpoint` 放 mixin 供所有 Trellis2 trainer 复用

#### 修改文件

1. **`src/flow_factory/hparams/training_args/trellis2_opd.py`** — 新增 `disable_grad_checkpoint_for_distill: bool = False` 字段
2. **`src/flow_factory/trainers/trellis2_mixin.py`** — 新增 `_set_transformer_checkpoint(self, enabled: bool)` 方法（~8 LOC）
3. **`src/flow_factory/trainers/trellis2_opd.py`** — `optimize()` 中用 `if self.training_args.disable_grad_checkpoint_for_distill:` 包裹 toggle 调用
4. **`examples/opd/lora/trellis2/dense_self_distill.yaml`** — 加 `disable_grad_checkpoint_for_distill: true`

#### 验证

1. 重新提交 Koala → `_distill` backward 应通过（不再有 inplace FFT recompute 错误）
2. 检查 `_precompute_mu_T` 仍正常工作（`torch.no_grad()` 下 checkpointing 不生效，不受影响）
3. 确认 `train/kl_div` 有实际值输出（证明 backward + optimizer step 完成）

---

### Review · 2026-06-08 · 四轮（三轮 fix 无效 → 真因：rope_phases complex buffer version tracking）

> **触发**：三轮的 gradient checkpointing toggle fix 提交后（job `ericzyma-job-normal-20260608-165233`），`_distill` backward **仍然崩溃**，报完全相同的错误：
> ```
> RuntimeError: one of the variables needed for gradient computation has been modified
> by an inplace operation: [CUDAComplexFloatType [4096, 1, 64]] is at version 3; expected version 2
> ```

#### 三轮诊断为什么错了

三轮假设"inplace FFT 操作在 gradient checkpointing recompute 时导致 tensor 被污染"——但出错的 tensor 类型是 **`CUDAComplexFloatType [4096, 1, 64]`**，这不是任何 FFT 中间结果。下载 TRELLIS.2 后端源码后发现：

- `[4096, 1, 64]` complex = `self.rope_phases.unsqueeze(-2)`（`rope.py:31`）
- `self.rope_phases` 是 `SparseStructureFlowModel` 的 registered buffer（`sparse_structure_flow.py:113`），形状 `[4096, 64]` complex64
- 4096 = 16³（voxel 位置数），64 = head_dim / 2
- 关闭 gradient checkpointing 并不解决这个问题——version tracking 冲突发生在 autograd 图内部

#### 真正的根因

**`self.rope_phases`（complex64, [4096, 64]）作为 registered buffer 被 30 个 transformer block 共享使用。**

在 `sparse_structure_flow.py:240` 的 forward 循环中：
```python
for block in self.blocks:  # 30 blocks
    h = block(h, t_emb, cond, self.rope_phases)  # 同一个 buffer
```

每个 block 的 `apply_rotary_embedding`（`rope.py:29-33`）执行：
```python
x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
x_rotated = x_complex * phases.unsqueeze(-2)  # autograd 保存 phases view
x_embed = torch.view_as_real(x_rotated).reshape(...).to(x.dtype)
```

`phases.unsqueeze(-2)` 创建 `self.rope_phases` 的 view（shape [4096, 1, 64]），autograd 为 `*` 运算的 backward 保存这个 view 及其 version counter。

**关键交互**：当 `use_reentrant=False` 的 gradient checkpointing 在 backward 时重计算 forward：
1. checkpoint 保存所有 tensor 输入（包括 `phases`）及其 version
2. 30 个 block 的 checkpoint boundary 都保存同一个 `self.rope_phases` buffer
3. `torch.view_as_complex` / `view_as_real` 在 complex tensor 上的 version tracking 与 checkpoint 的 version 验证产生冲突
4. backward recompute block N 时，检测到 `phases` 的 version 与 checkpoint 时不一致 → crash

**但即使关闭 checkpointing**（三轮 fix），普通 autograd 在 backward 处理多个 block 时也可能遇到类似问题——因为 PyTorch 对 complex tensor view 的 version tracking 存在微妙的 edge case（同一 storage 的多个 view 在 autograd 图中的 version 可能被非预期地递增）。

#### 修复方案：每次 forward 前 clone rope_phases

**核心思路**：在 `SparseStructureFlowModel.forward` 执行 block 循环之前，**clone `self.rope_phases` 一次**。克隆出的副本有独立的 version counter，与原始 buffer 的 storage 完全隔离，autograd 不会再检测到 version mismatch。

**开销**：每次 forward 多一个 4096 × 64 × 8 bytes = **2 MB** 的 clone，可以忽略不计。

**实现方式**：在 `Trellis2TrainerMixin` 中 monkey-patch 模型的 forward 方法（因为 TRELLIS.2 后端代码在 S3 tar 中，不宜直接修改）：

```python
def _patch_rope_phases_clone(self) -> None:
    """Monkey-patch transformer forward to clone rope_phases per call.

    The dense transformer's rope_phases buffer (complex64, [4096, 64])
    is shared across all 30 blocks. Autograd saves views of it for
    backward, and version-counter conflicts arise between checkpoint
    boundaries. Cloning once per forward gives autograd a fresh tensor
    with its own version counter.
    """
    model = self.adapter.pipeline.transformer
    if not hasattr(model, "rope_phases") or model.rope_phases is None:
        return

    _orig_forward = model.forward

    def _forward_with_cloned_phases(x, t, cond):
        orig_buf = model.rope_phases
        model.rope_phases = orig_buf.clone()
        try:
            return _orig_forward(x, t, cond)
        finally:
            model.rope_phases = orig_buf

    model.forward = _forward_with_cloned_phases
    logger.info("Patched %s.forward to clone rope_phases per call",
                type(model).__name__)
```

**调用时机**：在 `_init_trellis2()` 末尾调用 `self._patch_rope_phases_clone()`，对所有 Trellis2 trainer 生效。

#### 与三轮 fix 的关系

| Fix | 作用 | 是否保留 |
|-----|------|---------|
| `_patch_rope_phases_clone()` | **根因修复**——消除 version tracking 冲突 | ✅ 保留 |
| `disable_grad_checkpoint_for_distill` flag + `_set_transformer_checkpoint()` | **辅助防御**——减少 checkpoint 带来的额外 version 检查 | ✅ 保留（作为 config 选项） |

两个 fix 可以独立工作。clone fix 是根本修复；checkpointing toggle 是额外的安全网。当 clone fix 验证成功后，可以在 YAML 中设 `disable_grad_checkpoint_for_distill: false` 恢复 checkpointing 以节省显存。

#### 修改文件

1. **`src/flow_factory/trainers/trellis2_mixin.py`** — 新增 `_patch_rope_phases_clone()` 方法（~20 LOC），在 `_init_trellis2()` 末尾调用。同时给 `_set_transformer_checkpoint()` 加日志确认 toggle 生效
2. **`src/flow_factory/trainers/trellis2_opd.py`** — 无需额外修改（patch 通过 mixin init 自动应用）

#### 验证

1. 重新提交 Koala → `_distill` backward 应通过（rope_phases clone 消除 version mismatch）
2. 日志中应出现 `"Patched SparseStructureFlowModel.forward to clone rope_phases per call"` + `"Set use_checkpoint=False on 30 blocks"`
3. 确认 `train/kl_div` 有实际值输出（证明 backward + optimizer step 完成）
4. 后续验证：设 `disable_grad_checkpoint_for_distill: false`，仅靠 clone fix 是否足够

---

### Review · 2026-06-22 · 五轮（group_size 约束过严）

> **触发**：shape self-distillation 训练提交时因 `group_size (1) must be divisible by per_device_batch_size (2)` 报错。临时改 group_size=2 绕过，但发现该约束对非 `group_contiguous` sampler 无意义。

#### 问题

`trellis2_mixin.py` L95-101 的 `K % bs != 0` 检查是**无条件执行**的，但 `_batches_to_merge` 只在 `sampler_type == "group_contiguous"` 时 > 1。对于 `distributed_group_aligned`（当前默认 sampler），`_batches_to_merge = 1`，这个整除约束完全不生效——它计算出的值没有被使用。

```python
# 当前代码（L95-105）
K = self.training_args.group_size
bs = self.training_args.per_device_batch_size
if K % bs != 0:  # ← 无条件检查，但只在 group_contiguous 下有意义
    raise ValueError(...)
if self.config.data_args.sampler_type == "group_contiguous":
    self._batches_to_merge = K // bs
else:
    self._batches_to_merge = 1
```

**对比 dgpo**：`(num_processes * per_device_batch_size) % group_size == 0`（方向相反，允许 group_size < batch_size）。

#### 修复

将整除检查移入 `group_contiguous` 分支：

```python
K = self.training_args.group_size
bs = self.training_args.per_device_batch_size
if self.config.data_args.sampler_type == "group_contiguous":
    if K % bs != 0:
        raise ValueError(
            f"group_size ({K}) must be divisible by "
            f"per_device_batch_size ({bs}) for batch accumulation."
        )
    self._batches_to_merge = K // bs
else:
    self._batches_to_merge = 1
```

#### 影响

- `shape_self_distill.yaml` 可以恢复 `group_size: 1`（OPD self-distill 不需要 group 语义）
- 现有 `group_contiguous` 用户不受影响（约束仍在该分支内生效）
- 这是 mixin 改动，影响所有 Trellis2 trainer（GRPO/NFT/OPD）

#### 修改文件

1. **`src/flow_factory/trainers/trellis2_mixin.py`** L95-105 — 重构约束位置
2. **`examples/opd/lora/trellis2/shape_self_distill.yaml`** — 可选：`group_size` 改回 1

