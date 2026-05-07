# Session Handoff: trellis2 ⇄ upstream/main 合并审计

## 任务目的
在 trellis2 把 upstream/main 合并进来的过程中，全面审计**当前工作树**相对 trellis2 备份分支的改动，确认所有为了对齐 upstream 而做的修改都不会破坏 `.agents/sessions/` 里记录的、为支持 trellis2 图生 3D GRPO 训练所做的功能。

## 当前 git 状态

| 引用 | SHA | 含义 |
|------|-----|------|
| `backup/trellis2-before-upstream-merge` | `f4f07c9` | trellis2 合并 upstream 之前的最后状态（参考基线） |
| `backup/main-before-upstream-merge` | （另一基线，本次审计可不重点关注） | |
| `upstream/main` | `5c79203` | 要对齐的上游主线（`Jayce-Ping/Flow-Factory`） |
| `origin/trellis2` | `3bc4384` | 个人远端 trellis2（`87003697/Flow-Factory`），落后于本地 |
| `merge-base` | `724c911` | trellis2 与 upstream 的共同祖先 |
| 当前 `HEAD` | `f4f07c9`（尚未 commit merge） | 在 `trellis2-merge-upstream` 分支，`.git/MERGE_HEAD = 5c79203` |

> ⚠️ remote 配置已修正：`origin = 87003697/Flow-Factory.git`（自有），`upstream = Jayce-Ping/Flow-Factory.git`（主开发线）。

## 上一阶段执行内容（按时间倒序）
- 修正 origin/upstream remote 反置；fetch all + 重设 trellis2 上游
- 提交了 `chunked.py` perf 改动 + `.gitignore` + `continue-session` skill（已落到 `f4f07c9`）
- 创建集成分支 `trellis2-merge-upstream`，`git merge --no-commit --no-ff upstream/main` 执行后产生 9 个冲突文件
- 逐个解决冲突：
  - `models/registry.py`、`rewards/registry.py`：纯 union（保留 trellis2 的 `trellis2`/6 个 `unified_reward_*`，加入 upstream 的 `ltx2_t2av`/`ltx2_i2av`/`rational_rewards_*`）
  - `logger/formatting.py`：全盘采纳 upstream 音视频写出（trellis2 没碰过此文件）
  - `trainers/grpo.py`：采纳 upstream 的 lazy stacking + `_maybe_offload_samples_to_cpu`（trellis2 不重写 `optimize`，sample 默认 GPU-resident，等价 + 省显存）；保留 trellis2 自加的 `_extra_eval_inference_kwargs` hook
  - `data_utils/loader.py`：把 upstream 的 consolidate pipeline 用 `dataset_cls=` 参数化，让 `Image3DDataset` 走同一路径
  - `data_utils/sampler.py` + `sampler_loader.py` + `hparams/data_args.py` + `hparams/args.py`：保留 trellis2 的 `DistributedGroupAlignedSampler` **和** upstream 的 `GroupDistributedSampler` 共存；新加 `_align_for_distributed_group_aligned` helper 与 dispatcher 分支
  - 三份 trellis2 yaml 从 `examples/grpo/lora/trellis2_*.yaml` 迁到 `examples/grpo/lora/trellis2/{shape,shape_unified_reward,tex_unified_reward}.yaml`，`sampler_type` 维持 `distributed_group_aligned`
- **回滚两次"自以为可行"的错误**：
  1. 曾把 `DistributedGroupAlignedSampler` 删除 → 已恢复，与 `GroupDistributedSampler` 共存
  2. 曾把 `distributed_group_aligned` 加进 async-reward override 集合 → 已移除（HEAD 与 upstream 都不含此项；加入会让 trellis2 + unified_reward 配置被静默改成 `group_contiguous`，破坏 cross-GPU upstream sharing）
- 静态校验：`ReadLints` 全绿；三份 yaml 端到端 `Arguments.load_from_yaml` dry-run 全部维持 `distributed_group_aligned` 并通过对齐
- **当前状态**：合并完成、全部冲突已解、静态/import 检查通过，但**尚未 `git commit`**。`git status -s | wc -l = 131 个待提交条目`

## 调试经验（坑）
- 不能假设两个名字相似的 sampler（`DistributedGroupAlignedSampler` vs `GroupDistributedSampler`）功能等价：前者支持 `K < W` 且做 prompt-索引跨 rank scatter；后者要求 `K % W == 0` 且每 rank 同 prompt 序（DGPO 不变量）。**必须**在合并时同时保留。
- `GroupDistributedSampler` 用于 DGPO（rank 间 prompt 同序，依赖此不变量做无 collective 的 group-id 推断），不能用 `distributed_group_aligned` 替换。
- async-reward → `group_contiguous` 的 hard override 集合**只**应包含 `distributed_k_repeat`、`group_distributed`，不应扩展到 `distributed_group_aligned`，否则 trellis2 配置会被静默回退、cross-GPU upstream sharing 失效。
- `git merge-tree --write-tree` 在本仓库 git 版本里不支持；用 `git merge-base + git merge-tree <base> A B` 模拟。
- HEAD 与 upstream 的 stdout 包含非 utf-8 字节（如 0x89），Python 解码必须加 `errors='replace'`。
- `Image3DDataset` 仅覆盖 `_load_image`，所有 `compute_cache_path/load_merged/build_part_arrow_path/consolidate_parts/collate_fn/check_exists` 类方法以及 `target_arrow_path/num_shards/shard_index` init 参数都从 `GeneralDataset` 继承——`loader.py` 里参数化 `dataset_cls=` 是安全的。

## 参考代码

### 高冲突敏感文件（HEAD 和 upstream 都改过 → 重点审计目标）
| 文件 | 关键位置 | 说明 |
|------|---------|------|
| `src/flow_factory/data_utils/sampler.py` | `DistributedGroupAlignedSampler`、`GroupDistributedSampler`、`GroupContiguousSampler`、`DistributedKRepeatSampler` | 4 个 sampler 共存，前者是 trellis2 cross-GPU upstream sharing 的关键 |
| `src/flow_factory/data_utils/sampler_loader.py` | `SAMPLER_REGISTRY`、`get_data_sampler` | 4 个 sampler 都注册 |
| `src/flow_factory/data_utils/loader.py` | `_create_or_load_dataset(dataset_cls=...)` | 参数化后 `Image3DDataset` 走同一 consolidate pipeline |
| `src/flow_factory/data_utils/dataset.py` | `compute_cache_path/load_merged/build_part_arrow_path/consolidate_parts` | `Image3DDataset` 的父类，所有方法被 `loader.py` 通过 `dataset_cls.xxx` 调用 |
| `src/flow_factory/hparams/args.py` | `_resolve_sampler_type`、`_align_batch_geometry` 与 4 个 `_align_for_*` helpers | 新加 `_align_for_distributed_group_aligned` |
| `src/flow_factory/hparams/data_args.py` | `sampler_type` Literal 含 5 个值（含 `auto`） | |
| `src/flow_factory/hparams/training_args.py` | merge 后 trellis2 字段是否仍然完整 | 待审计 |
| `src/flow_factory/trainers/grpo.py` | `sample()` 末尾的 `_maybe_offload_samples_to_cpu`、`optimize()` 内的 lazy stacking、`_extra_eval_inference_kwargs` hook | trellis2_grpo 重写 `sample`、保留 hook，不重写 `optimize` |
| `src/flow_factory/trainers/abc.py` | `_maybe_offload_samples_to_cpu` 定义 | upstream 新增，trellis2 间接依赖 |
| `src/flow_factory/trainers/registry.py` | trellis2_grpo 注册保留 | |
| `src/flow_factory/logger/formatting.py` | `LogVideo.audio/_write_mp4_with_audio`、`from_i2av_samples`、`_process_t2av_samples` | 全盘 upstream 增量，trellis2 没动过 |
| `src/flow_factory/models/registry.py` | union | |
| `src/flow_factory/rewards/registry.py` | union | |
| `src/flow_factory/rewards/reward_processor.py` | merge 后 trellis2 unified_reward 路径仍可走 | 待审计 |
| `src/flow_factory/rewards/pick_score.py` | upstream 与 trellis2 都改过 | 待审计 |
| `src/flow_factory/samples/samples.py` | upstream 新增 T2AV/I2AV，trellis2 改了什么 | 待审计 |
| `src/flow_factory/models/abc.py` | upstream 与 trellis2 都改过 | 待审计 |

### trellis2 独有文件（merge 不该影响，但要确认依赖未断）
- `src/flow_factory/data_utils/image_3D_dataset.py`
- `src/flow_factory/models/trellis2/{__init__,trellis2,pipeline,chunked,chunked_mixin,flow_match_euler_discrete}.py`
- `src/flow_factory/trainers/trellis2_grpo.py`
- `src/flow_factory/rewards/{unified_reward,unified_reward_pairwise}.py`
- `src/flow_factory/samples/samples.py` 中的 `Trellis2Sample`

### 已迁移的 trellis2 example
- `examples/grpo/lora/trellis2/shape.yaml`
- `examples/grpo/lora/trellis2/shape_unified_reward.yaml`
- `examples/grpo/lora/trellis2/tex_unified_reward.yaml`

### `.agents/sessions/` 里需要逐个核对的 trellis2 历史 session（按时间）
| 日期 | session 文件 | 涉及功能 |
|------|------|------|
| 04-17 | `2026-04-17-trellis2-upstream-share.md` | cross-GPU upstream stage sharing 设计 |
| 04-19 | `2026-04-19-trellis2-grpo-reward-config.md` | reward 配置 |
| 04-19 | `2026-04-19-trellis2-grpo-debug-run.md` | debug run |
| 04-20 | `2026-04-20-trellis2-grpo-7gpu-launch.md` | 7-GPU 启动配置 |
| 04-20 | `2026-04-20-trellis2-grpo-device-mismatch.md` | device 修复 |
| 04-20 | `2026-04-20-trellis2-grpo-run.md` | 训练 run |
| 04-21 | `2026-04-21-trellis2-grpo-profiling-and-ddp-fix.md` | DDP 修复（`_set_static_graph`） |
| 04-21 | `2026-04-21-trellis2-grpo-sampling-profile.md` | sampling profile |
| 04-22 | `2026-04-22-trellis2-distributed-upstream-sharing.md` | **distributed upstream sharing 落地（DGA sampler 关键依据）** |
| 04-22 | `2026-04-22-trellis2-grpo-batchize-and-next-steps.md` | batch 设计 |
| 04-23 | `2026-04-23-trellis2-dga-optimize-phase-bug.md` | **DGA optimize 阶段 bug 修复** |
| 04-23 | `2026-04-23-trellis2-scoring-{plan,impl}.md` | scoring 实现 |
| 04-23 | `2026-04-23-trellis2-vllm-serve-setup.md` | vLLM serve |
| 04-24 | `2026-04-24-trellis2-rgba-bg-redesign.md` | **RGBA 背景重设计（Image3DDataset 关键依据）** |
| 04-24 | `2026-04-24-trellis2-tex-grpo-setup.md` | tex GRPO 配置 |
| 04-25 | `2026-04-25-trellis2-grpo-cfg-fix.md` | CFG 修复 |
| 04-25 | `2026-04-25-trellis2-cfg-refactor.md` | CFG 重构 |
| 04-25 | `2026-04-25-trellis2-decode-oom-{plan,fix}.md` | **decode OOM 修复（chunked.py canonical-sort 跳过依据）** |

## 最终方案
合并工作树已就绪，4 个 sampler 共存，`Image3DDataset` 走 upstream consolidate pipeline，trellis2 cross-GPU upstream sharing 经 `distributed_group_aligned` 不被 async-reward 静默 override，trellis2 examples 迁移到子目录但 sampler/参数维持原 trellis2 语义。下一步要做正式 cross-check 后才能 commit + push。

## 下一步任务
**审计当前工作树（merge in progress）相对 `backup/trellis2-before-upstream-merge` 的所有改动，逐一对照 `.agents/sessions/` 里 trellis2 image-to-3D GRPO 训练相关的设计/修复，确认没有任何被破坏。**

## 初步方案

> 进入新 session 时建议第一时间执行的步骤。

1. **掌握全局**：
   - `git rev-parse --abbrev-ref HEAD`（应显示 `trellis2-merge-upstream`）
   - `cat .git/MERGE_HEAD`（应等于 `5c79203`，即 upstream/main）
   - `git status -sb | head -20` 浏览待提交规模（约 131 条）
   - `git log --oneline -10 backup/trellis2-before-upstream-merge` / `git log --oneline -10 upstream/main`

2. **建立 3 路对比命令**（写入临时变量便于复用）：
   ```bash
   BASE=$(git merge-base backup/trellis2-before-upstream-merge upstream/main)   # 共同祖先
   BACKUP=backup/trellis2-before-upstream-merge                                  # trellis2 原状
   # 当前是工作树 (in-merge state)
   ```
   - 看 trellis2 自身改了什么：`git diff $BASE $BACKUP -- <path>`
   - 看 upstream 改了什么：`git diff $BASE upstream/main -- <path>`
   - 看当前合并版本与 trellis2 原状的差异：`git diff $BACKUP -- <path>`（这是审计核心）

3. **重点审计顺序**（按风险从高到低）：
   1. `data_utils/sampler.py` + `sampler_loader.py`：确认 4 个 sampler 共存、`DistributedGroupAlignedSampler` 类内部逻辑与 backup 完全一致
   2. `hparams/args.py` 的 `_align_for_distributed_group_aligned` 与 `_resolve_sampler_type` async-reward 集合
   3. `trainers/grpo.py`：lazy stacking + `_maybe_offload_samples_to_cpu` 不破坏 trellis2_grpo 调用链
   4. `data_utils/loader.py`：`dataset_cls=` 参数化对 `Image3DDataset` 端到端正确（最好在新 session 里做一次 `Arguments.load_from_yaml + 实例化 dataset`，并验证缓存路径与 backup 行为一致）
   5. `data_utils/dataset.py`、`hparams/training_args.py`、`samples/samples.py`、`models/abc.py`、`trainers/abc.py`、`rewards/reward_processor.py` 等"两边都改过"的文件，逐一 diff 与 backup 对比 trellis2 行为

4. **逐 session 核对**：拿上面表格列出的 sessions，对每条记录的"修改文件 / 关键函数"在当前工作树里 grep 验证仍然存在且语义未变。重点 4 篇：
   - `2026-04-22-trellis2-distributed-upstream-sharing.md`（→ DGA sampler / trellis2_grpo）
   - `2026-04-23-trellis2-dga-optimize-phase-bug.md`（→ DGA + optimize 路径）
   - `2026-04-24-trellis2-rgba-bg-redesign.md`（→ Image3DDataset / loader）
   - `2026-04-25-trellis2-decode-oom-{plan,fix}.md`（→ chunked.py，已 commit 不会被 merge 影响，但 import 路径要在 ltx2 等新模块旁边继续工作）

5. **端到端 dry-run**（不需要实际启动训练）：
   ```bash
   /home/zhiyuan_ma/anaconda3/envs/grpo3d_trellis2/bin/python -c "
   import os; os.environ['WORLD_SIZE']='7'; os.environ['LOCAL_WORLD_SIZE']='7'
   from flow_factory.hparams import Arguments
   for y in [
       'examples/grpo/lora/trellis2/shape.yaml',
       'examples/grpo/lora/trellis2/shape_unified_reward.yaml',
       'examples/grpo/lora/trellis2/tex_unified_reward.yaml',
   ]:
       cfg = Arguments.load_from_yaml(y)
       assert cfg.data_args.sampler_type == 'distributed_group_aligned', (y, cfg.data_args.sampler_type)
       print(y, 'OK')
   "
   ```
   并对 `Image3DDataset` 做一次 mini 实例化（用 `dataset/trellis2_debug/` 之类小数据集）验证 consolidate pipeline 正常出 cache。

6. **审计通过后**才执行最后一步：
   ```bash
   git commit -m "..."         # 写一段说明合并策略 + sampler 共存 + Image3DDataset 兼容的 message
   git push origin trellis2-merge-upstream      # 或 ff trellis2 → trellis2-merge-upstream → push origin trellis2
   ```
   > 注意：是否 ff `trellis2` 分支 + push `origin/trellis2` 取决于用户最终决定（之前的计划是 ff + push，但应在审计完成后再确认）。

7. **若审计发现回归**：先在 `trellis2-merge-upstream` 分支单独 commit 修复，避免污染 `backup/trellis2-before-upstream-merge`；最差情况 `git merge --abort` 重新走一次。
