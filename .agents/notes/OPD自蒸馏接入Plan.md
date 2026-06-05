# Flow-Factory OPD 自蒸馏接入方案

> 目标：把 `flow_grpo_custom` 的 OPD 自蒸馏（student=c_ref, teacher=c_tgt，同权重）思路移植到 Flow-Factory 的 Trellis2 训练栈。
> 范围：仅讨论 Flow-Factory 侧的改造，不动 `flow_grpo_custom`。

## 2026/06/02

---

### 1. 现状盘点

| 组件 | flow_grpo_custom | Flow-Factory |
|---|---|---|
| SDE Scheduler（dense） | ✅ | ✅ `FlowMatchEulerDiscreteSDEScheduler.step()` 暴露 `next_latents_mean / std_dev_t / dt / log_prob` 六字段（`scheduler/flow_match_euler_discrete.py:243-438`） |
| SDE Scheduler（sparse） | n/a | ✅ `SparseFlowMatchEulerSDEScheduler` 同样产出 `next_latents_mean / std_dev_t / dt`，但 **签名不一致**：`std_dev_t/dt` 是 0-d 标量、`next_latents_mean` 是 SparseTensor（`models/trellis2/flow_match_euler_discrete.py:222-249, 326-404`） |
| `dynamics_type` ODE/SDE 切换 | ✅ | ✅ `scheduler.eval()` 切 ODE（`scheduler/abc.py:73`） |
| OPD KL 损失（μ-matching） | ✅ | ✅ `feat/diffusion-opd` 分支 `DiffusionOPDTrainer._distill` 已实现 `0.5·‖μ_S−μ_T‖²/σ̄²`（`trainers/opd/trainer.py:289-315`） |
| 两段式（precompute teacher → distill） | ✅ | ✅ `_precompute_teacher_targets` + `_distill`（`trainers/opd/trainer.py:196-243, 244-331`） |
| Teacher = "同权重 + 不同 cond" | ✅ | ❌ `load_teachers` 硬性要求 `finetune_type=="lora"` 且 `path` 必填（`trainers/opd/common.py:97-109`） |
| 每 teacher 条件覆盖 hook | ✅ 自蒸馏关键 | ❌ `_forward_step` 只暴露 `guidance_scale`，conditioning 一律走 `**batch`（`trainers/opd/trainer.py:497-510`） |
| 多 teacher 同 dataset | ✅ | ❌ Trainer 主动拒绝（`trainers/opd/trainer.py:107-116`） |
| Trellis2 GRPO/NFT trainer | n/a | ✅ `Trellis2GRPOTrainer / Trellis2NFTTrainer` + `Trellis2TrainerMixin`（window-merge rollout） |
| 渲染 z₀ → 2D | ✅ | ✅ `Trellis2Adapter.render_latents`（`models/trellis2/trellis2.py:2789-2857`） |
| FlowEdit 指引 | ✅ | ✅ `_inference_flowedit`（`models/trellis2/trellis2.py:1722-1896, 1822-1864`），但用相同 cond + 不同 cfg_scale，**不是不同的 cond** |
| 配对 (c_ref, c_tgt) 数据通路 | ✅ | ❌ dataset 单一 `images` 列，无配对 |
| 同模型双 forward 是否被禁 | n/a | ✅ 不被禁。`_forward_sparse(cond=...)` stateless |
| `next_latents_mean / std_dev_t` 预存 trajectory | ✅ | ❌ Trellis2 rollout 不存，但训练态 `forward()` 调 `scheduler.step` 直接返回，可现算 |
| Per-sample teacher routing | ✅ | ✅ `_teacher_index_for_sample`（`trainers/opd/trainer.py:432-456`） |

---

### 2. 关键架构判断 — 推荐方案 (A)

**扩展 `feat/diffusion-opd` 的 `DiffusionOPDTrainer`，新增 `Trellis2OPDTrainer`，并把 `TeacherConfig` 推广为支持 "self-cond" 模式。**

理由：
- OPD 损失和两段式 cache 已经写好，从零写只是复制
- PASS 2 对 `mu_T` 来源不可知 —— 只读 `sample.extra_kwargs["mu_teacher"]`，怎么得到的不影响梯度流
- 改 GRPO 替换损失代价大（reward/advantage/KL-to-ref 整套要重写）
- 真正缺的部分有限：(a) "self-cond teacher" 模式 (b) cond 覆盖 hook (c) sparse σ̄² broadcast 适配 (d) c_tgt 数据通路

---

### 3. 改造文件清单

#### 3.1 Trainer 侧（核心，~340 LOC）

| # | 文件 | 动作 | LOC |
|---|---|---|---|
| **T1** | `src/flow_factory/trainers/trellis2_opd.py` | **新建** `Trellis2OPDTrainer(Trellis2TrainerMixin, DiffusionOPDTrainer)`：复用 mixin rollout + OPD distill；override `_forward_step` 支持 sparse 输出（取 `.feats`、广播 σ̄²） | ~250 |
| **T2** | `trainers/opd/trainer.py` | **改** `_forward_step` L497-505：加 `cond_override: Optional[Dict]` 参数，`forward_inputs.update(cond_override or {})` | ~10 |
| **T3** | `trainers/opd/common.py` | **改** `load_teachers`：当 `teacher.path == "self"` 跳过 `_load_lora`，直接 snapshot 当前 student | ~40 |
| **T4** | `trainers/opd/trainer.py` | **改** `__init__` L107-116：放宽 `_source_to_teacher` 唯一性检查；收集 `_teacher_cond_override` | ~25 |
| **T5** | `trainers/opd/trainer.py` | **改** `_precompute_teacher_targets` L213-240：从 `_teacher_cond_override` 取 cond 透传给 `_forward_step` | ~15 |

#### 3.2 TeacherConfig 扩展

| # | 文件 | 动作 | LOC |
|---|---|---|---|
| **C1** | `hparams/training_args/opd.py` | `TeacherConfig` 加 `mode: Literal["lora","self"] = "lora"`、`cond_source: Optional[str]`、`image_cond_keys: Optional[List[str]]`；放宽 `path` 必填 | ~30 |

#### 3.3 Conditioning 数据通路

| # | 文件 | 动作 | LOC |
|---|---|---|---|
| **D1** | `data_utils/image_3D_dataset.py` 或新建 `paired_image_3D_dataset.py` | 支持 JSONL schema 携带 `tgt_image` 列 | ~40 |
| **D2** | `models/trellis2/trellis2.py:925-988` `preprocess_func` | batch 含 `tgt_images` 时额外 encode → `image_cond_tgt_512 / 1024 / neg_*` | ~30 |
| **D3** | `models/trellis2/trellis2.py:247-262` `Trellis2Sample` | 加 `tgt_image_cond_512 / 1024` 字段 | ~10 |

#### 3.4 离线 FlowEdit 脚本

| # | 文件 | 动作 | LOC |
|---|---|---|---|
| **R2** | `scripts/offline_flowedit_to_dataset.py`（新建） | 离线脚本：(image, prompt) → `flowedit_inference` → 落盘 `tgt_image` | ~120 |

> **不需要 port 渲染 / FlowEdit pipeline**：Flow-Factory 已有 `render_latents` 和 `_inference_flowedit`。

#### 3.5 Loss & Rollout

| # | 文件 | 动作 | LOC |
|---|---|---|---|
| **L1** | `trainers/trellis2_opd.py` | `_forward_step` override：从 batch 选 `image_cond_512`（student）或 `image_cond_tgt_512`（teacher），同一份 `adapter.forward`，返回 `(mu, std_dev_t_broadcast, dt)` | ~60 |
| **L2** | `trainers/trellis2_opd.py` | sparse 适配：`mu_S − mu_T` 取 `.feats`；`std_dev_t` 0-d 展为 `(B,)`；`per_sample_mse` 用 sparse_coords scatter | ~50 |

#### 3.6 Hparams 新增字段

| 字段 | 位置 | 默认 | 说明 |
|---|---|---|---|
| `train.teachers[i].mode` | `TeacherConfig` | `"lora"` | `"self"` 表示 teacher = student weights |
| `train.teachers[i].cond_source` | `TeacherConfig` | `None` | 如 `"c_tgt"` |
| `train.teachers[i].image_cond_keys` | `TeacherConfig` | `None` | `{"image_cond_512": "image_cond_tgt_512", ...}` |
| `data.datasets[i].tgt_image_dir` | dataset args | `None` | c_tgt 图像目录 |

---

### 4. 推荐分阶段实施

#### Phase 1 — 最小可行 OPD（~2-3 天，~600 LOC）
**c_tgt 离线生成 + 同权重 self-distill**

1. D1 + D2 + D3：dataset 双图列 + preprocess 双 encode
2. R2：离线 FlowEdit 脚本，小数据集跑出 `(ref.png, tgt.png)` 对
3. C1：`TeacherConfig` 加新字段
4. T2 + T3 + T4 + T5：`DiffusionOPDTrainer` "self" 分支
5. T1 + L1 + L2：`Trellis2OPDTrainer` + sparse μ-matching

**验证策略**：
- 先固定 `c_tgt = c_ref` → loss 应该很快趋零（sanity check）
- 再切换 `c_tgt` 到 FlowEdit 输出 → loss 非零且单调下降

#### Phase 2 — 在线 FlowEdit（~2 天，~150 LOC）

1. `Trellis2OPDTrainer.sample()` 复用 `_flowedit_group`（`trainers/trellis2_mixin.py:199-229`）
2. 每个 rollout window 先跑 FlowEdit 得 `(latents_src, latents_tgt)`，编码 `c_tgt` 塞回 batch
3. 跳过 R2 脚本，但训练 step 翻倍

#### Phase 3 — 渲染态指引（可选，研究性，~3-5 天）

1. 训练步内 `render_latents` 把 student z₀ 渲染成视频
2. 用 qwen-image-edit-plus 把每帧编辑成 c_tgt 视图
3. 编码视图回 DINOv3 → 新 c_tgt 进入 OPD KL

---

### 5. 需要拍板的开放问题

| Q | 问题 | 影响 |
|---|---|---|
| **Q1** | c_tgt 来源：Phase 1 离线 FlowEdit 落盘 / Phase 2 在线 FlowEdit | D1/D2/R2 工作量分配 |
| **Q2** | LoRA-only 还是支持 full-FT student？现 `load_teachers` 强制 `finetune_type=="lora"` | T3 改动复杂度 |
| **Q3** | sparse `std_dev_t/dt` 是 0-d 标量（全 batch 共享 σ̄²），接受这个近似还是改 scheduler？ | sparse scheduler 跨项目影响 |
| **Q4** | 先在哪个 stage 做 OPD？dense (512) / shape (1024 sparse) / tex (1024 sparse)？ | 建议先 dense，σ̄² 最规整 |
| **Q5** | OPD 中是否启用非零 negative（现 `pipeline.get_cond` neg 是 zeros）？ | 不启用则无需特殊处理 |

---

### 6. 关键代码定位（开发时速查）

| 想做什么 | 看这里 |
|---|---|
| OPD KL loss 公式 | `trainers/opd/trainer.py:289-315` |
| Teacher cache PASS 1 | `trainers/opd/trainer.py:196-243` |
| Forward step（要加 cond_override） | `trainers/opd/trainer.py:497-510` |
| Teacher 加载 | `trainers/opd/common.py:97-109` |
| TeacherConfig schema | `hparams/training_args/opd.py:77-101, 168-186` |
| Trellis2 sparse scheduler | `models/trellis2/flow_match_euler_discrete.py:222-249, 326-404` |
| Trellis2 dense scheduler | `scheduler/flow_match_euler_discrete.py:243-438` |
| Trellis2 渲染 | `models/trellis2/trellis2.py:2789-2857` |
| Trellis2 FlowEdit 推理 | `models/trellis2/trellis2.py:1722-1896, 1822-1864` |
| Trellis2 preprocess（要加 tgt encode） | `models/trellis2/trellis2.py:925-988` |
| Trellis2 Sample 数据类 | `models/trellis2/trellis2.py:247-262` |
| Trellis2 mixin rollout | `trainers/trellis2_mixin.py:151-197` |
| Trellis2 NFT sparse loss 参考 | `trainers/trellis2_nft.py:182-212` |

---

### 7. 一句话总结

> **不需要新造任何核心组件**。Flow-Factory 已经有 SDE scheduler + Trellis2 trainer + OPD trainer + 渲染 + FlowEdit；现在要做的只是写一个 `Trellis2OPDTrainer` 把它们串起来，并给 `TeacherConfig` 加个 `mode="self"` 让 teacher 可以等于"student 权重 + 不同 cond"。

---

## 2026/06/02 v2 — Day-1 可动手版本

> 在 worktree `feat/diffusion-opd` 上做了代码级核对。Q1-Q5 用户回答收齐：
> Q1=render-of-rollout / Q2=LoRA-only-extensible / Q3=升级 sparse scheduler / Q4=dense 优先 / Q5=cond only

### v2.1 假设核对结果 — LOC 大幅下调

| 任务 | 原估 | 修正 | 说明 |
|---|---|---|---|
| T1 Trellis2OPDTrainer | ~250 | **150-200** | mixin 是纯 behavioral；Q1 加了 base-rollout 分支 |
| T2 `_forward_step += cond_override` | ~10 | **6-10** ✅ |  |
| T3 load_teachers self mode | ~40 | **15-25** | adapter 已有 `add_named_parameters` 内存快照 |
| T4 trainer init 放宽 routing | ~25 | **15-20** | Phase 1 不需彻底放宽 |
| T5 _precompute cond override | ~15 | **8-12** |  |
| C1 TeacherConfig 新字段 | ~30 | **20-30** ✅ |  |
| D1+D2+D3 paired dataset | ~80 | **❌ 删除** | Q1 改在线生成 c_tgt |
| Sparse scheduler 升级 | 未估 | **20-35** | 修真 bug（B>1 当前会崩） |

**总 Phase 1: 470 → 280-360 LOC**

### v2.2 6 个关键发现

🟢 **好消息**

1. **Adapter 已有内存快照原语** — `add_named_parameters` (`models/abc.py:617-662`) 直接对 live trainable LoRA 做 EMAModuleWrapper(decay=0.0) 快照，无磁盘 IO；T3 self-mode **实质是删代码**
2. **`_flowedit_group` (`trellis2_mixin.py:199-229`) 是天然 "render side-channel" 模板** — 把 `flowedit_inference` 替换成 `decode_and_render` 就是 c_tgt 骨架
3. **`_render_kwargs` 默认 `decode_output=False`** — c_tgt 路径只需局部覆盖

🔴 **坏消息（必须修）**

4. **Sparse scheduler `B>1` 当前直接崩溃（真 bug）** — `opd/trainer.py:357` reshape `(B,-1)` 在 sparse 拿到 0-d 标量 → RuntimeError；**Q3 升级是必须项**
5. **`_distill` 对 dense tensor 有 3 处隐藏耦合** — L349 flatten/L319-324 isinstance/L265 stack；Phase 1 dense 全部回避，Phase 2 sparse 必撞
6. **rollout 不存 `next_latents_mean / std_dev_t`** — OPD 训练时通过重跑 `adapter.forward(..., return_kwargs=[...])` 重算；**Trellis2 adapter.forward 必须支持这个调用约定，Day-1 第一个 smoke 测就是它**

### v2.3 Phase 1 文件清单 (~320 LOC)

| # | 路径 | 改动 | LOC |
|---|---|---|---|
| **F1** | `hparams/training_args/opd.py` | TeacherConfig +mode/cond_source/image_cond_keys | 25 |
| **F2** | `trainers/opd/common.py` | dispatch table + self_lora 早返回 | 20 |
| **F3** | `trainers/opd/trainer.py` | `_forward_step` 加 `cond_override` | 12 |
| **F4** | `trainers/trellis2_opd.py` | **新文件** Trellis2OPDTrainer | 180 |
| **F5** | `models/trellis2/flow_match_euler_discrete.py` | scatter-mean 替换 | 30 |
| **F6** | `trainers/__init__.py` 或 registry | 注册 trellis2_opd | 3 |
| **F7** | `configs/trellis2_opd_dense_self.yaml` | 最小 config | 50 |

### v2.4 TeacherConfig 三态枚举（Q2 future-proof）

```python
mode: Literal["external_lora", "self_lora", "self_full"] = "external_lora"
```
- Phase 1: `external_lora` + `self_lora`
- Phase 2: `self_full` 占位 → `NotImplementedError`

**关键洞察**：`use_named_parameters` 是 swap 上下文，`add_named_parameters` 已 mode-agnostic（`requires_grad=True` 对 LoRA = LoRA 适配器，对 Full-FT = 所有可训参数）。**Phase 1 写 self_lora 时调用与 self_full 完全一致。**

**dispatch table 模式**避免 if/elif：
```python
_TEACHER_LOADERS = {
    "external_lora": _load_external_lora_teacher,
    "self_lora":     _load_self_lora_teacher,
    "self_full":     _load_self_full_teacher,
}
```

### v2.5 Sparse Scheduler 升级 — 具体改动

文件：`models/trellis2/flow_match_euler_discrete.py`

**替换 1（Flow-SDE，L392-403）**：
```python
# 旧: std_dev_scalar = float(std_dev_t.mean().item())  # ← 0-d 标量，崩溃源
# 新: scatter-mean to (B,)
batch_idx = latents.coords[:, 0].long()
B = int(batch_idx.max().item()) + 1
std_dev_B = torch.zeros(B, ...); std_dev_B.scatter_add_(0, batch_idx, std_dev_t.squeeze(-1))
counts = torch.zeros(B, ...); counts.scatter_add_(0, batch_idx, ones); counts.clamp_min_(1.0)
return SDESchedulerOutput.from_dict({"std_dev_t": std_dev_B / counts, ...})
```

**替换 2（CPS, L460-469）**：同形态

**推荐抽 helper** `_reduce_per_point_to_per_sample(value, batch_idx, B)` — 与 `trellis2.py:755-772` 的 `_reduce_sparse_log_prob` 复用

**语义**：scatter-mean 在数学上**精确**（被 reduce 的值在 sample 内已经相同），不是近似

**调用点影响**：
- OPD `opd/trainer.py:357` ✅ 修复 bug
- NFT `nft.py:371` ✅ 无影响（不读）
- GRPO `grpo.py:506-509` ✅ 共享 t 时数值相同；分散 t 时更正确

### v2.6 Day-1 立即写的 4 个文件（按依赖顺序）

#### 文件 1: `hparams/training_args/opd.py`
```python
mode: Literal["external_lora", "self_lora", "self_full"] = "external_lora"
cond_source: Literal["dataset", "teacher"] = "dataset"
image_cond_keys: Optional[List[str]] = None

# __post_init__:
if teacher.mode == "external_lora" and not teacher.path:
    raise ValueError(f"... mode=external_lora requires path.")
if teacher.mode == "self_full":
    raise NotImplementedError("Phase 2")
```

#### 文件 2: `trainers/opd/common.py` — dispatch table

#### 文件 3: `trainers/opd/trainer.py`
```python
def _forward_step(..., cond_override: Optional[Dict[str, Any]] = None):
    forward_inputs = {**self.training_args, **batch, "t": t, ...}
    if cond_override:
        forward_inputs.update(cond_override)
    ...
```

#### 文件 4: `trainers/trellis2_opd.py` 新文件骨架
```python
@register_trainer("trellis2_opd")
class Trellis2OPDTrainer(Trellis2TrainerMixin, DiffusionOPDTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._init_trellis2()

    def sample(self) -> List[BaseSample]:
        # PASS A: base rollout (LoRA disabled) → render → c_tgt
        # PASS B: student rollout for trajectory
        ...

    def _rollout_base_and_render(self, merged_batch):
        raise NotImplementedError("Day-1 占位；Day-2 实现")
```

**Day-1 验收 4 步**：
```bash
# 1) Schema 加载
uv run python -c "from flow_factory.hparams.training_args.opd import TeacherConfig"
# 2) Loader dispatch
uv run python -c "from flow_factory.trainers.opd.common import _TEACHER_LOADERS"
# 3) Trainer 类可 import + MRO 正确
uv run python -c "from flow_factory.trainers.trellis2_opd import Trellis2OPDTrainer"
# 4) 用最小 yaml 实例化（不调 .start()）
uv run python scripts/smoke_init_trellis2_opd.py
```

### v2.7 仍需用户澄清的 5 个开放问题

#### 7.1 ⚠ **Trellis2 `adapter.forward` 是否支持 OPD 调用约定？**
未验证。Day-1 第一个 sub-task 写 smoke：
```python
out = adapter.forward(t=..., t_next=..., latents=..., next_latents=...,
                     return_kwargs=["next_latents_mean", "std_dev_t", "dt"])
assert out.next_latents_mean is not None
```
**如果失败，T1 LOC 翻倍**（要在 Trellis2 adapter 加 forward 适配层）。

#### 7.2 c_tgt 的 encode 视角选哪个？
Trellis2 condition 编码器输入 `(C, H, W)`；渲染输出有 `mesh / clay_video / mask_video`。三选一：
- (a) 单视角（固定/随机？）
- (b) 多视角拼接
- (c) clay_video 某帧

每种影响 `_rollout_base_and_render` 实现 20-50 LOC。

#### 7.3 base model = "LoRA disabled" 的具体语义
需要审计 `models/abc.py` 是否有现成 `disable_lora` context；若无，加 F8 文件（~15 LOC）。
**注意**：`use_named_parameters(teacher_name)` 跟 `disable_adapters()` 不能嵌套。

#### 7.4 Batch 内 t 是否共享？
`opd/trainer.py:482` 看是 per-sample。F5 完成后用 single vs multi-sample loss 对照测确认。

#### 7.5 Phase 2 sparse `_distill` redesign 工作量
3 处 dense 耦合（mu_teacher 存储、`(mu_S - mu_T).pow(2)` reduction、`torch.stack`）。Phase 2 shape stage 必撞，估计 50-80 LOC。**写进 Phase 2 todo。**

### v2.8 worktree 状态

- 路径：`/Users/zhiyuanma/Desktop/codes/Flow-Factory/.claude/worktrees/opd-baseline-test`
- 分支：`opd-test`（tracks `origin/feat/diffusion-opd`）
- 已 ExitWorktree (action=keep)，session 回到主 repo `trellis2` 分支
- 下次 work 进 worktree：`cd .claude/worktrees/opd-baseline-test`
