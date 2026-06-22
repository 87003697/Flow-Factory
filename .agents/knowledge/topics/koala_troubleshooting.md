# Koala 踩坑记录 — Flow-Factory (Trellis2)

按日期倒序追加。**遇到新问题解决后，立即在此文件顶部追加一条。**

> 平台通用 Gotchas 见 `~/Desktop/codes/.agents/KOALA.md`（末尾大表）。
> 本文件只记录 Flow-Factory 项目的特有问题。

---

## 2026-06-17 — `python -m flow_factory.train` 直接运行导致 World Size=1

### 核心教训

**`flow_factory.train` 不会内部调用 `accelerate launch`。必须用 `accelerate launch -m flow_factory.train` 启动，否则 `Accelerator()` 默认单进程。**

### 背景

提交 8 GPU normal 训练，`setup_koala.sh` 已有 `unset RANK WORLD_SIZE ...`（修复 Koala RANK 注入问题）。但训练仍然 World Size=1，sampler 报错 `num_replicas=1`。

### 根因

`train.py` → `load_trainer()` → `Accelerator()` 创建时读环境变量。`unset RANK` 后没有任何分布式上下文，`Accelerator()` 默认 `world_size=1`。之前的 `unset` 只解决了"假 RANK 导致跳过 accelerate"的问题，但 `python -m flow_factory.train` 本身不会 spawn 多进程。

### 解法

用 `accelerate launch` 包裹：
```bash
/tmp/uv-venv/bin/accelerate launch \
    --config_file config/accelerate_configs/multi_gpu.yaml \
    -m flow_factory.train examples/opd/lora/trellis2/<config>.yaml
```

### 关键认识

| 之前的理解 | 实际情况 |
|-----------|---------|
| `python -m flow_factory.train` 内部会处理多 GPU 启动 | 不会。`Accelerator()` 只是读取已有的分布式环境，不主动 spawn |
| `unset RANK` 就够了 | 不够。还需要 `accelerate launch` 来建立真正的多进程上下文 |

---

## 2026-06-17 — visibility-overlay smoke test 八连调

### 核心教训

**`ff-train` wrapper 内部调 `uv run`，即使 venv 已装好一切也会 re-resolve。直接用 `/tmp/uv-venv/bin/python -m flow_factory.train` 最可靠。**

### 新发现（前序 session 未记录的）

| 问题 | 现象 | 解法 |
|------|------|------|
| Job name 含保留字 | `koala submit -j ericzyma-vis-overlay-debug` 被拒 | Job name 不能含 `debug` / `normal`（Koala CLI 校验） |
| `ff-train` 也触发 uv sync | `nohup /tmp/uv-venv/bin/ff-train ...` 仍报 mmcv 构建失败 | `ff-train` 是 uv wrapper，内部调 `uv run`。用 `python -m flow_factory.train` |
| SSH 场景下 stdout 全缓冲 | SSH 执行训练命令，长时间无输出 | `python -u`（unbuffered）或 `nohup > /tmp/log 2>&1 &` + `tail -f` |
| Monkey-patch 在 accelerate 子进程失效 | 写 `save_vis.py` 做 monkey-patch 后调 `main()`，patch 不生效 | `accelerate launch` fork 新进程。必须直接改源文件（sed / rsync 推送） |
| s5cmd sync 遇 broken symlink 崩溃 | `.claude/worktrees/*/` 下残留 broken symlink | sync 前 `find . -xtype l -delete` 或精确 `rm -f` |
| Worktree 默认从 main 创建 | `EnterWorktree` 默认 `origin/main`，trellis2 文件不存在 | 手动 `git worktree add -b feat/xxx .claude/worktrees/xxx trellis2` |
| `Trellis2Adapter` 不是 nn.Module | `next(self.parameters()).device` → `AttributeError` | 用 `self.device`（BaseAdapter 属性，来自 `self.accelerator.device`） |
| Debug 模式静默忽略 `-c` | `koala submit -c "bash train.sh"` 在 debug 模式下被忽略，pod 启动交互 shell | Debug 不支持 `-c`。需要启动命令时用 `-m normal -g 1`，或 SSH 进去手动跑 |
| `UV_FROZEN=1` 不够，还需 `UV_NO_SYNC=1` | 设了 UV_FROZEN 后 `uv run` 仍卸载/重装 5 个包 | UV_FROZEN 阻止 lockfile 更新但不阻止 env sync。加 `UV_NO_SYNC=1` 或直接绕过 uv |

### 可靠 smoke test 命令

```bash
# Pod 内（setup 后）
/tmp/uv-venv/bin/python -m flow_factory.train examples/opd/lora/trellis2/shape_smoke_1gpu.yaml
```

---

## 2026-06-15 — koala v1.6.0 默认镜像升级引发三连崩

### 核心教训

**koala CLI 升级 = 默认镜像可能变 → 所有与 CUDA/NCCL/torch ABI 相关的假设都要重新验证。**

### 背景

提交 OPD ref-KL regularization 训练，koala CLI 从 1.5.1 自动升级到 1.6.0。默认镜像变为 `cuda12.8-efa1.44-ubuntu24.04-uvcache`（之前是 cuda12.4 系列）。连续 4 次提交失败，每次修一个 bug 暴露下一个。

### Bug 链

| 顺序 | 根因 | 表现 | 修复 |
|------|------|------|------|
| 1 | `uv run ff-train` 触发隐式 `uv sync`，尝试 resolve geneval → mmcv，mmcv 依赖 `pkg_resources`（Python 3.12 已移除） | `ModuleNotFoundError: No module named 'pkg_resources'` | `export UV_FROZEN=1` 在 setup 顶部（**注意：UV_FROZEN 阻止 lockfile 更新但不完全阻止 env sync，见 #2**） |
| 2 | 即使 `UV_FROZEN=1`，`uv run` 仍然 sync env（"Uninstalled 5 packages / Installed 5 packages"），替换了 torch 2.6.0+cu124 的 NCCL 依赖 | `ImportError: libtorch_cuda.so: undefined symbol: ncclCommResume` | **不用 `uv run`**，直接调 `/tmp/uv-venv/bin/ff-train`（setup 已装好一切） |
| 3 | Koala PyTorchJob 注入 `RANK=0` / `WORLD_SIZE` 等环境变量，但并未用 torchrun 管理进程 → `ff-train` CLI 看到 `RANK` 就跳过 accelerate launch | `World Size: 1`，batch geometry 校验失败：`num_replicas=1, per_device_batch_size=2, group_size=14` | setup 末尾 `unset RANK WORLD_SIZE LOCAL_RANK MASTER_ADDR MASTER_PORT GROUP_RANK LOCAL_WORLD_SIZE` |

### 关键认识

1. **`uv run` ≠ 单纯的 venv 包装器**：它会在每次调用时 sync env 到 lockfile 状态（即使 `UV_FROZEN=1`），可能卸载/安装包。正式训练永远直接用 venv binary。
2. **`LD_LIBRARY_PATH` 对 NCCL 无效的原因**：不是 torch 找不到 bundled NCCL，而是 `uv run` 的 env sync 替换了 torch 的 NCCL 依赖包。根因在 `uv run` 而非 library path。
3. **Koala PyTorchJob 的 RANK 注入**：单节点多卡时，容器环境有 `RANK=0` 但没有 torchrun。任何检测 `RANK` 来判断"是否有外部 launcher"的逻辑都会误判。

### v1.6.0 好消息

| 特性 | 说明 |
|------|------|
| `--s3-log` | 内置 s3tee，日志自动落盘 `s3://.../$USER/.koala-logs/$JOB_NAME/`。无需手动安装，开箱即用 |
| 自动注入 secrets | `WANDB_API_KEY`、`HF_TOKEN` 等从宿主机自动带入容器，不再需要 `-c "export ..."` |
| 密钥不明文显示 | `koala get` 不再在任务详情里显示密钥 |

### setup_koala.sh 修改

```bash
# [顶部] 阻止 uv run 隐式 re-resolve（仍不够，见 #2）
export UV_FROZEN=1

# [torch 安装后] 确保 torch 的 bundled NCCL 优先（防御性，主要靠不用 uv run）
export LD_LIBRARY_PATH="${VENV}/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

# [末尾] 清除 Koala 注入的分布式环境变量
unset RANK WORLD_SIZE LOCAL_RANK MASTER_ADDR MASTER_PORT GROUP_RANK LOCAL_WORLD_SIZE
```

### 提交命令模板（v1.6.0）

```bash
# 注意：直接调 ff-train，不经过 uv run
koala submit -m normal -g 8 --s3-log -j <job-name> \
    --code "$S3:/data/work/run_codes" \
    -c "cd /data/work/run_codes && . scripts/setup_koala.sh && /tmp/uv-venv/bin/ff-train <config.yaml>"
```

---

## 2026-06-08 — OPD 训练连续失败：三层 bug 互相遮掩

### 核心教训

**当「回退配置也崩」时，问题大概率不在配置，而在环境/基础设施变化。**

### 背景

OPD 自蒸馏训练在对齐 DiffusionOPD 参数后反复崩溃（~8 次提交全失败）。将配置完全回退后仍然崩溃，排除了配置问题。最终发现是 3 个独立 bug 叠加，修一个才暴露下一个：

| 顺序 | 根因 | 表现 | 修复 |
|------|------|------|------|
| 1 | `PROJECT_DIR` 硬编码 `/data/work/flow-factory`，但 Koala v1.4.0 代码目录改为 `/data/work/run_codes` | symlink（pretrained_weights、dataset、third_party）建在错误路径；模型文件找不到；7 个 rank 同时尝试从 HF 下载 16GB 模型 → 看起来像 OOM Kill | `PROJECT_DIR="$(pwd)"` 动态检测 |
| 2 | 启动命令用 `uv run accelerate launch`，`uv run` 隐式 resolve 整个项目依赖（含 geneval → mmcv），mmcv 构建失败 | 进程 exit 1，但 pod 被回收后无日志，误以为还是 OOM | 改用 venv 的 `accelerate launch` 直接调用（setup 已将 venv/bin 加入 PATH） |
| 3 | `WANDB_API_KEY` 为空时 `wandb.init()` 抛 `UsageError` 而非优雅降级 | Python traceback，rank 0 exitcode=1 | setup 中检测空 key 则 `export WANDB_MODE=disabled` |
| (附) | 启动命令中 accelerate config 路径写错（`configs/accelerate/deepspeed_zero2.yaml` 不存在） | `FileNotFoundError` | 改为正确路径 `config/accelerate_configs/multi_gpu.yaml` |

### 为什么这么难调

1. **Pod 回收后日志消失**：前 6 次失败完全没有 Python traceback，所有失败看起来都像 SIGKILL/OOM
2. **三个 bug 串联**：修了路径问题 → 暴露 uv run 问题 → 修了 uv run → 暴露 wandb 问题。每次只能看到当前最先触发的 bug
3. **误导性假设**：最初假设是「配置参数改动导致 OOM」，浪费大量时间在配置回退和增量测试上

### 方法论

| 原则 | 说明 |
|------|------|
| **回退全部配置仍崩 → 看环境** | 配置变更不是唯一变量，Koala 平台版本、镜像、代码目录结构都可能变 |
| **优先拿到 Python traceback** | 没有 traceback 就是盲调。应在第一次失败后就加 s3tee 或日志上传 |
| **每次修复只修一个变量** | 多个修复同时提交，无法确认哪个生效了（本次恰好每次只暴露一个 bug，所以自然做到了） |
| **检查 koala get 的启动命令** | `koala get <job>` 会显示完整的 `启动命令:`，可以直接看到路径、环境变量是否正确 |

### 相关 commit

- `2c68794` fix(setup): detect PROJECT_DIR dynamically instead of hardcoding
- `2932780` fix(setup): disable wandb when WANDB_API_KEY is not set

### 最终结果

训练成功完成（Succeeded），产出 8 个 checkpoint（0 到 140 步，约 35 epoch），PickScore baseline = 0.7242。

---

## 2026-06-05 — merge origin/main 后 S3 残留旧文件导致 pod 拉到旧代码

### 现象

`training_args.py` 被重构为 package（`training_args/`），但 pod 拉到的 S3 还有旧的 `training_args.py`，import 冲突导致训练崩溃。

### 根因

`aws s3 sync` 默认是增量上传，不删除远端多余文件。本地已删除/重命名的文件在 S3 上持久存在，pod 拉代码时会拿到两个版本。

### 解决

`scripts/koala_submit.sh` 的 `aws s3 sync` 必须加 `--delete`（现已修复）：

```bash
aws s3 sync . "${S3}/" --delete --exclude '...'
```

### 教训

- **merge 后必须做 `aws s3 sync --delete`**
- 任何文件重命名/删除操作后，旧文件会在 S3 上残留

---

## 2026-06-05 — `get_eval_dataloaders` 中 `dataset_cls` 未定义（NameError）

### 现象

```
NameError: name 'dataset_cls' is not defined
```

发生在 `src/flow_factory/data_utils/loader.py` 的 `get_eval_dataloaders`。

### 根因

`main` 分支重构了 eval dataloader，`dataset_cls` 变量在 merge 后没有被正确解析传入。

### 解决

在 `get_eval_dataloaders` 中加了 `dataset_cls` 解析 + 传参（同 train dataloader 的处理方式）。

---

## 2026-06-05 — BiRefNet (rembg_model) device mismatch

### 现象

```
RuntimeError: Expected all tensors to be on the same device
```

BiRefNet 的 `rembg_model` 留在 CPU，而其他模型在 GPU。

### 根因

`preprocessing_modules` 是白名单，不在列表里的模块不会被 `on_load_components` 移到 GPU。`rembg_model` 漏加了。

### 解决

在 `src/flow_factory/models/trellis2/trellis2.py` 的 `preprocessing_modules` 中加入 `rembg_model`：

```python
preprocessing_modules = [..., "rembg_model"]
```

---

## 2026-06-05 — eval reward processor 在 Trellis2 报 ValueError（tokenizer）

### 现象

`self.adapter.tokenizer` 在 Trellis2 会 raise `ValueError`（Trellis2 是纯图像模型，没有 tokenizer）。

### 根因

`src/flow_factory/trainers/abc.py` 的 `_init_reward_model` 中使用了 `self.adapter.tokenizer`，而 Trellis2 的 adapter 不支持此属性。

### 解决

改用局部变量 `tokenizer`（在初始化时已安全解析，不依赖 adapter）：

```python
# 修改前
processor = RewardProcessor(tokenizer=self.adapter.tokenizer, ...)
# 修改后
processor = RewardProcessor(tokenizer=tokenizer, ...)  # tokenizer 是安全的局部变量
```

---

## 2026-06-05 — per-source allocation 缺失（DGA sampler）

### 现象

单 source 训练时 `_align_for_distributed_group_aligned` 没有调用 `_align_unique_sample_num()`，导致 per-source allocation 数据结构缺失，后续报 KeyError 或数量不匹配。

### 根因

`main` 分支的 multi-dataset 架构要求 `_align_unique_sample_num()` 必须在 `_align_for_distributed_group_aligned` 中被调用，但 merge 时漏了这一步。

### 解决

在 `src/flow_factory/hparams/args.py` L780 的 `_align_for_distributed_group_aligned` 中补充调用 `_align_unique_sample_num()`。

---

## 2026-06-05 — 单 source 配置 cache fingerprint 变化

### 现象

单 source 配置下每次启动 cache fingerprint 不一致，导致 dataset cache 失效、每次都重新预处理。

### 根因

`main` 的 multi-dataset 路径会加 `train_source:default` tag，但单 source 时不应该加（和之前版本的 fingerprint 不兼容）。

### 解决

在 `src/flow_factory/data_utils/loader.py` 的 fingerprint 生成逻辑中，单 source 时跳过 `train_source:` tag。

---

## 2026-05-27 — setup 完成后秒崩（UV_FROZEN 未设置）

### 现象

setup 打印 `=== Setup complete ===` 后，训练命令在毫秒内崩溃：

```
× No solution found when resolving dependencies ...
```

### 根因

`uv run` 默认在每次启动前做隐式 `uv sync`（re-resolve），撞上 lockfile 依赖冲突。

### 解决

`scripts/setup_koala.sh` 顶部加 `export UV_FROZEN=1`，让所有 `uv run` / `uv sync` 信任 lockfile 不重新解析。

### 教训

- "setup 全部打印 OK，紧接着训练命令毫秒级崩" = 几乎必然是 `uv run` 隐式 re-resolve
- `UV_FROZEN=1` 要在 setup 最顶部 export，覆盖整个生命周期

---

## 2026-05-27 — `koala delete --force` 误黑健康节点

### 现象

应用层报错后用了 `koala delete --force`，把健康节点加入黑名单 12h。

### 根因

`--force` 的语义是"强制删除 + 把节点拉黑"，无法区分节点级故障和应用层失败。

### 解决

按失败类型选路径：

```bash
# 节点级故障（Updating / PodInitializing 卡死 5+ 分钟）→ 用 --force
echo "y" | koala delete --force <job>

# 应用层失败（任务进入过 Running，setup 跑完，用户代码崩）→ 不带 --force
koala delete <job>
```

判断方法：`koala get <job> | grep "Running: True"` —— 有则是应用层失败，不要 `--force`。
