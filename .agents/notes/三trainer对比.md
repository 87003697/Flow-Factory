# Flow-Factory 三 Trainer 代码对比 — GRPO / NFT / OPD

> 在 `feat/diffusion-opd` 分支上对比三个 trainer 的实现差异。
> 文件位置：`src/flow_factory/trainers/{grpo,nft,opd/trainer}.py`

## 2026/06/02

---

### 一句话区分

| Trainer | 核心思路 |
|---|---|
| **GRPO** | reward → advantage → PPO clipped ratio |
| **NFT** | reward → 正/负样本权重 → 在 v/x0 空间做 weighted MSE |
| **OPD** | 没有 reward → student 学 teacher 的 μ（mean-matching） |

---

### 三个 sample() 的差异

| Trainer | 输出存的字段 | 需要 reward？ |
|---|---|---|
| **GRPO** (`grpo.py:94-104`) | trajectory + `log_prob` | ✅ |
| **NFT** (`nft.py:194-201`) | trajectory + `clean_latents`（作正样本 target） | ✅ |
| **OPD** (`opd/trainer.py:170-189`) | trajectory + 缓存 `mu_teacher` | ❌ |

> OPD 是唯一不进 reward_buffer 的 trainer，`prepare_feedback` 是 no-op (`opd/trainer.py:191-194`)

---

### 三个 loss 公式的代码对比

#### GRPO loss (`grpo.py:200-209`)
```python
ratio = torch.exp(output.log_prob - old_log_prob)   # π_new / π_old
unclipped = -adv * ratio
clipped   = -adv * torch.clamp(ratio, 1+lo, 1+hi)   # PPO clip
loss = max(unclipped, clipped).mean()
```
**核心**：advantage × ratio。**需要 log_prob**。

#### NFT loss (`nft.py:339-360`)
```python
positive_pred = β·v_new + (1-β)·v_old
negative_pred = (1+β)·v_old - β·v_new

x0_pred     = noised - σ·positive_pred
neg_x0_pred = noised - σ·negative_pred

pos_loss = ((x0_pred     - clean)² / weight).mean(...)
neg_loss = ((neg_x0_pred - clean)² / neg_weight).mean(...)

r = clamp(advantage / max_adv / 2 + 0.5, 0, 1)      # reward → [0,1] 权重
loss = (r·pos_loss + (1-r)·neg_loss) / β
```
**核心**：reward 转成 [0,1] 权重 `r`，对 `clean_latents` 做正负 MSE 的凸组合。

#### OPD loss (`opd/trainer.py:348-361`)
```python
# Student 当前 forward
mu_S, std_dev_t, dt = student.forward(...)["next_latents_mean"]

# PASS 1 已缓存的 teacher μ
mu_T = mu_teacher_all[:, idx]

per_sample_mse = (mu_S - mu_T).pow(2).flatten(1).mean(dim=1)    # ‖μ_s - μ_t‖²
denom = scheduler.get_kl_divergence_denominator(std_dev_t, dt)   # 1.0 (ODE) 或 σ² (SDE)
loss = (0.5 * per_sample_mse / denom).mean()
```
**核心**：mean-matching KL。**就一行**。

---

### 关键架构差异表

| 维度 | GRPO | NFT | OPD |
|---|---|---|---|
| 是否需要 `log_prob` | ✅ 必须 | ❌ | ❌ |
| 是否需要 reward | ✅ | ✅ | ❌ |
| 学习信号空间 | 概率比（ratio） | x0-space (sample) | μ-space (mean of next step) |
| optimize PASS 数 | 1 | **2** | **2** |
| PASS 1 内容 | n/a | sampling policy 的 `v_old` | teacher 权重的 `μ_T` |
| 能跑 ODE 模式？ | ❌（log_prob 在 σ=0 时退化） | ✅ | ✅ |
| 能否完全无 reward model | ❌ | ❌ | ✅ |

---

### 简化关系链

```
GRPO:    经典 RL（log_prob + reward + ratio + advantage）
  ↓ 去掉 log_prob，把 reward 转成 [0,1] 权重
NFT:     supervised-style RL（v/x0 空间 MSE）
  ↓ 去掉 reward，把 target 从 clean_latents 换成 teacher μ
OPD:     纯蒸馏（最简单的 mean-matching MSE）
```

> **OPD 是 NFT 的更简化变体**：丢掉正负样本对比、丢掉 reward，只保留"学一个 target"。差别只在 target 来源：
> - NFT target = `clean_latents`（数据集真值）
> - OPD target = teacher 模型在同 z_t 下的预测

---

### 数学公式对照

```
GRPO:  L = -A · π_new(z_t)/π_old(z_t)              [+ PPO clip]
NFT:   L = r·‖x0(v_pos) - clean‖² + (1-r)·‖x0(v_neg) - clean‖²
OPD:   L = ‖μ_student - μ_teacher‖² / (2σ²)
```

---

### 两段式（PASS1 + PASS2）的内涵差异

| | PASS 1（no_grad 预计算） | PASS 2（grad） |
|---|---|---|
| **NFT** | 用 sampling policy（旧 student）跑一遍存 `old_v_pred` 作为参考点 | 当前 student forward 算 NFT loss |
| **OPD** | 切到 teacher 权重，跑一遍存 `mu_teacher` 作为目标 | student forward 算 KL loss |

> 都是"预计算一份 detached tensor 当训练目标/参考"，但 NFT 是"自己的旧版本"，OPD 是"另一个模型"。

---

### 关键代码定位

| 想看什么 | file:line |
|---|---|
| GRPO PPO ratio loss | `grpo.py:200-209` |
| NFT 正负样本构造 | `nft.py:339-340` |
| NFT weighted MSE | `nft.py:343-360` |
| OPD mean-matching KL | `opd/trainer.py:348-361` |
| OPD PASS 1 teacher μ 缓存 | `opd/trainer.py:215-271` |
| OPD scheduler denominator | `opd/trainer.py:352-359` (调 `scheduler.get_kl_divergence_denominator`) |
| 三个 sample() | `grpo.py:94`, `nft.py:194`, `opd/trainer.py:170` |

---

### 给 self-distillation OPD 的实现启示

1. **不需要 log_prob 通路** — 跟 NFT 一样可以用 `return_kwargs=["noise_pred"]` 风格的 forward，省 SDE log_prob 计算
2. **OPD 已经实现了 ODE/SDE 自适应** — `scheduler.get_kl_divergence_denominator()` ODE 时返回 1.0，SDE 时返回 σ²·|dt|，trainer 不用关心
3. **PASS 1 的钩子点**（`opd/trainer.py:215-271`）正好是注入 "self-cond teacher" 的位置 — 不切权重，只换 cond
