# OPD 原理入门 — Flow-OPD 与 DiffusionOPD

> 给完全新手的笔记。On-Policy Distillation (OPD) 是 2026 年把 LLM 领域的"在线策略蒸馏"思路搬到图像生成模型（diffusion / flow matching）的两篇论文。

## 2026/06/02

### 0. 一句话讲清楚 OPD 在解决什么问题

> 让一个图像生成模型"同时擅长很多任务"（OCR、构图、美感）很难，因为不同任务的奖励信号会互相打架（"跷跷板效应"）。OPD 的解法：**先训好每个任务的小专家（teacher），再让一个学生模型去"同时模仿所有专家"**。

- 类比：与其让一个学生同时学语数英（互相干扰），不如先让三个老师各自精通一门，再让学生跟着三个老师轮流学。

---

### 1. 几个必须知道的前置概念

| 概念 | 一句话解释 |
|---|---|
| **Diffusion / Flow Matching** | 图像生成模型，从纯噪声出发，分 N 步（如 10 步）一步步去噪，最后变成图 |
| **Reverse Step** | 每一步从 `x_t`（含噪图）→ `x_{t-1}`（更清晰图）。这一步本质是从一个高斯分布 `N(μ, σ²)` 采样 |
| **μ (mean)** | 模型预测的"下一步均值" — 这是一切的核心，记住这个变量名 `prev_sample_mean` |
| **σ (sigma)** | 这一步的噪声强度 |
| **SDE / ODE** | SDE = 带随机噪声的采样（σ>0）；ODE = 确定性的采样（σ=0）。OPD 论文说"两者通过 mean-matching 统一" |
| **LoRA** | 给大模型挂的小补丁（几十 MB），可以在同一个模型上挂多个，运行时切换 |
| **PPO / GRPO** | 强化学习算法。GRPO 是 flow_grpo 仓库的方案，用 reward 训练生成模型 |
| **KL 散度** | 衡量两个概率分布的"距离"。本笔记的核心公式就是它的一种特例 |

---

### 2. 核心公式（只有一行，不用怕）

两个高斯分布 `N(μ_s, σ²)` 和 `N(μ_t, σ²)` 的 KL 散度，化简后就是：

```
KL(student ‖ teacher) = ‖μ_student − μ_teacher‖² / (2 · σ²)
```

> **直觉**：让学生预测的均值 `μ_s` 去贴近老师预测的均值 `μ_t`。就这么简单。

代码里实现就一行：
```python
kl = ((mu_student - mu_teacher) ** 2).mean(...) / (2 * sigma ** 2)
```

ODE 极限（σ→0）就丢掉分母：
```python
kl = 0.5 * (mu_student - mu_teacher) ** 2
```

---

### 3. 两篇论文 / 两个 GitHub 仓库

| | Flow-OPD | DiffusionOPD |
|---|---|---|
| arXiv | [2605.08063](https://arxiv.org/abs/2605.08063) | [2605.15055](https://arxiv.org/abs/2605.15055) |
| GitHub | [CostaliyA/Flow-OPD](https://github.com/CostaliyA/Flow-OPD) | [ali-vilab/DiffusionOPD](https://github.com/ali-vilab/DiffusionOPD) |
| 团队 | USTC | 复旦 + 通义万相 |
| 本地路径 | `_reference_codes/Flow-OPD/` | `_reference_codes/DiffusionOPD/` |
| 一句话定位 | **全家桶** — PPO + KL + MAR 都能开 | **极简版** — 纯 distillation，删掉 PPO |

> **共同祖先**：两个仓库都 fork 自 [yifan123/flow_grpo](https://github.com/yifan123/flow_grpo)（GRPO for flow matching）。两边那个 92 行的 SDE 采样器 `sd3_sde_with_logprob.py` **完全一模一样**，是 OPD 的数学钩子。

---

### 4. SDE 采样器为什么是关键？

`sd3_sde_with_logprob.py:49-67` 把每一步的高斯参数都暴露出来：
```python
prev_sample_mean = ...   # μ — 模型预测的均值
std_dev_t = ...           # σ — 这一步的噪声强度
prev_sample = prev_sample_mean + std_dev_t * sqrt(dt) * noise   # 采样得到 x_{t-1}
log_prob = log N(prev_sample; mean, std)                         # 这一步的 log 概率
return prev_sample, log_prob, prev_sample_mean, std_dev_t        # ← 四件套
```

这四件套支持两种训练范式：
- **GRPO 老路**：用 `log_prob` 算 PPO ratio
- **OPD 新路**：用 `(μ, σ)` 直接算 KL，**完全不需要 log_prob**

---

### 5. 两个仓库的关键差异

| 维度 | Flow-OPD | DiffusionOPD |
|---|---|---|
| **Loss 形态** | 混合：PPO + β·KL，可切三种模式 (`task_only` / `kl_only` / `gkd`) | 纯 distillation，一行 `loss = distill_loss` |
| **Loss 公式** | `½‖μ_s−μ_ref‖²/σ²` (file: `train_sd3_opd_mix.py:1709, 1732, 1748`) | `½‖μ_s−μ_t‖²/σ²` (file: `train_sd3_opd.py:1171-1191`) |
| **多教师策略** | 多数据集**轮流训**（alternate epochs） | 多教师**每个 batch round-robin** (`teacher_idx = i % len(teachers)`) |
| **Teacher 怎么 forward** | 训练循环里**每步**重新跑 teacher（慢） | rollout 阶段**一次性算好 μ_teacher 存起来**，训练时只 detach 取用（快） |
| **多教师内存** | 加载多个独立 PeftModel（N 倍占用） | 同一 transformer 上挂 N 个 LoRA adapter，`set_adapter()` 切换（省显存） |
| **ODE 支持** | 没有 | 有 — `noise_level=0` 触发 ODE 分支去掉 σ 分母 |
| **冷启动** | 多种：SFT / 模型 merge / 用 reference LoRA | 直接从 SD3.5 底模启动 |

---

### 6. Flow-OPD 独有的招数

#### 6.1 MAR (Manifold Anchor Regularization)

> **解决什么问题**：student 在追 reward 的时候，容易"为了得分牺牲画质"（图像出现 saturation、artifacts 之类）。MAR 加一个"任务无关的锚"把它拉回原始流形。

实现 (`train_sd3_opd_mix.py:1748-1750`)：
```python
loss = gkd_loss + β · ‖μ_student − μ_MAR_anchor‖² / (2σ²)
#       └─模仿 task expert┘   └─别离原始流形太远─┘
```
- `mar_lora` 是一个**冻结**的 LoRA，挂在 base 上当锚（第 916-926 行）
- 不参与梯度，只 forward

#### 6.2 三态切换的训练脚本

`train_sd3_opd_mix.py` 同一个文件里塞了三种范式：
- `reward_mode="task_only"` → 纯 GRPO
- `reward_mode="kl_only"` → KL 当 reward 喂给 PPO
- `reward_mode="gkd"` → 纯 OPD distillation

> **意思**：这个仓库其实是个"对照实验框架"，"OPD"对应的是 `gkd` 模式。

---

### 7. DiffusionOPD 独有的招数

#### 7.1 Teacher mean 提前算 + detach

DiffusionOPD 训练 loop **完全没有 teacher forward**。在 rollout 阶段：
```python
# train_sd3_opd.py:933-965
for k, teacher_lora in enumerate(teachers):
    transformer.set_adapter(f"teacher_{teacher_lora.name}")  # 切到老师
    teacher_means[k] = [compute_step_mean(...) for j in range(num_steps)]
transformer.set_adapter("default")  # 切回学生
sample["teacher_prev_sample_means"] = teacher_means
```
训练时直接用：
```python
# train_sd3_opd.py:1167-1191
teacher_mean = sample["teacher_prev_sample_means"][:, j].detach()
delta = mu_student - teacher_mean
kl = (delta ** 2) / (2 * sigma ** 2)
loss = kl.mean()
```

> **省在哪里**：教师不在训练计算图里，不占梯度内存；每条轨迹每个老师只算 num_steps 次而不是 N 个 inner_epoch 次。

#### 7.2 LoRA adapter 切换

不加载 N 个完整 transformer，而是**同一个 transformer 上挂 N 个 LoRA**：
- N 个 teacher 只多几十 MB（LoRA 参数量小）
- Flow-OPD 是真的加载多个 PeftModel，N 倍显存

---

### 8. 接到 Flow-Factory 的改造点

> Flow-Factory 是当前所在仓库，是 RL fine-tuning for diffusion/flow matching 的框架。要把 OPD 接进去，最小改动：

1. **Sampler 暴露 μ, σ** — 检查现有 SDE sampler 是否返回 `prev_sample_mean, std_dev_t`，没有的话加上
2. **Rollout 后跑 teacher** — 学 DiffusionOPD：每条轨迹后切 LoRA 跑教师，存 `teacher_prev_sample_means` 进 trajectory dict
3. **Loss 替换** — 把 GRPO 的 `-A·ratio` 换成 `½‖μ_s−μ_t‖²/σ²`
4. **Config 加 `train.teachers = [...]`** — 抄 `DiffusionOPD/config/opd.py:81-106` 的格式

如果想加 MAR：再加一个冻结的 anchor LoRA，loss 多一个 β 项。

---

### 9. 阅读路线（5 个文件，1 小时通透）

按这个顺序读：

1. **`Flow-OPD/flow_grpo/diffusers_patch/sd3_sde_with_logprob.py`** (92 行)
   > 数学起点。看清 μ/σ 怎么从 flow matching 速度场推出来。
2. **`DiffusionOPD/scripts/train_sd3_opd.py:1150-1210`** (~60 行)
   > 最干净的 OPD loss 实现，无 PPO 干扰。
3. **`DiffusionOPD/scripts/train_sd3_opd.py:860-995`**
   > 看多教师 round-robin 和 teacher mean 预计算。
4. **`DiffusionOPD/config/opd.py`** (240 行)
   > 完整 multi-teacher 配置 schema，最适合直接抄。
5. **`Flow-OPD/scripts/train_sd3_opd_mix.py:1530-1810`**
   > 看 PPO + KL + GKD + MAR 怎么共存于一个 loop。

---

### 10. 常见困惑

| 困惑 | 解答 |
|---|---|
| "GKD" 是什么？ | Generalized Knowledge Distillation —— 在 Flow-OPD 里就是它对 OPD 模式的命名（`reward_mode="gkd"`）|
| `noise_level` 干什么的？ | 控制 SDE 的随机性强度。`=0` 退化成 ODE（确定性采样），`>0` 是 SDE |
| `teacher` 和 `reference` 一样吗？ | 在 DiffusionOPD 里基本一样（都是 LoRA）。在 Flow-OPD 里，`kl_ref_lora` 是 task-specific reference，`mar_lora` 是 task-agnostic anchor，是两个不同角色 |
| 为什么 DiffusionOPD 没 PPO？ | 论文宣称 closed-form KL 比 PPO 方差更低，所以直接换掉。代码里就是不用 `log_prob` 了 |
| 论文里"continuous-state Markov processes 推导"？ | 数学等价于"两个同 σ 高斯的 KL"，代码就一行。理论篇幅大但实现轻 |

---

### 速查 — 核心代码位置

| 想看什么 | 文件:行号 |
|---|---|
| Flow-OPD 的 KL loss 公式 | `Flow-OPD/scripts/train_sd3_opd_mix.py:1709, 1732, 1748, 1763` |
| Flow-OPD 的 MAR 实现 | `Flow-OPD/scripts/train_sd3_opd_mix.py:914-926, 1597-1623` |
| Flow-OPD 多任务交替配置 | `Flow-OPD/config/grpo.py:128-156` |
| DiffusionOPD 的 KL loss 公式 | `DiffusionOPD/scripts/train_sd3_opd.py:1171-1191` |
| DiffusionOPD 多教师 LoRA 切换 | `DiffusionOPD/scripts/train_sd3_opd.py:933-967` |
| DiffusionOPD teacher 配置 | `DiffusionOPD/config/opd.py:81-106` |
| SDE 采样器（两边一样） | `*/flow_grpo/diffusers_patch/sd3_sde_with_logprob.py:49-93` |
