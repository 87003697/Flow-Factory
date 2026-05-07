# Session Handoff: trellis2-merge-upstream 审计与提交

## 任务目的
对 `trellis2-merge-upstream` 分支（in-merge，9 个冲突已解但未 commit）做完整审计，确认 upstream 改动没有破坏 trellis2 image-to-3D GRPO 训练相关的全部 session 设计/修复；审计通过后正式 commit，但暂不 push（等冒烟验证）。

## 当前 git 状态

| 引用 | SHA | 含义 |
|------|-----|------|
| **新 merge commit** | `00a5add` | 双 parent = `f4f07c9` + `5c79203`，落在 `trellis2-merge-upstream` 分支 |
| `backup/trellis2-before-upstream-merge` | `f4f07c9` | trellis2 合并前的最后状态（审计基线，仍指向同一 SHA） |
| `upstream/main` | `5c79203` | 已合并的 upstream 主线 |
| `merge-base` | `724c911` | trellis2 与 upstream 共同祖先 |
| 工作树 | 干净 | 无 uncommitted 变更 |
| 是否 push | **否** | 仅本地 commit |

## 执行内容（按时间顺序）

- 读取上一 session 笔记 `.agents/sessions/2026-04-27-trellis2-merge-audit.md`，按其"初步方案"7 步执行
- 三路对比 17 个高冲突敏感文件（`git diff $BACKUP -- <path>`）：
  - `trainers/trellis2_grpo.py`、`models/trellis2/image_3D_dataset.py`、`models/trellis2/chunked.py` 等 trellis2 独有/关键文件 diff 完全为空
  - `data_utils/sampler.py` 中 `DistributedGroupAlignedSampler` 类逻辑与 backup 100% 等价（仅 `_dataset_size(dataset)` cast helper + docstring 增量）
  - `data_utils/sampler_loader.py` 注册 4 个 sampler；`hparams/data_args.py` Literal 5 项；`hparams/args.py` 4 个 `_align_for_<sampler>` helper
  - async-reward override 集合是 `{distributed_k_repeat, group_distributed}`，不含 DGA（trellis2 + unified_reward 不会被静默回退）
  - `trainers/grpo.py` 的 lazy stacking + offload helper 对 trellis2 兼容（trellis2_grpo 重写 sample，offload 默认 False）
  - `samples/samples.py` `_id_fields` 加入 `negative_prompt`（trellis2 上永远 None，hash 等价）
  - `models/abc.py` 的 `_resolve_component_names(None)` 行为变化被 `Trellis2Adapter._resolve_component_names` 完全屏蔽
- 跑了 `Arguments.load_from_yaml` dry-run（W=7 mock），三份 yaml 全部满足约束：`sampler_type=distributed_group_aligned`、`dataset_type=image_3d`、`trainer_type=trellis2_grpo`、`K=14, B=2, M=4 (W*B=14, W*B%K=0)`
- 核对 4 篇关键 session 的全部约束/修复保留：
  - `2026-04-22 distributed-upstream-sharing` → `DistributedGroupAlignedSampler` + `Trellis2Sample._STAGE_BROADCAST/METADATA_FIELDS` + `_distributed_upstream_stages / _broadcast_tensor / _broadcast_upstream_for_uid` 全在
  - `2026-04-23 dga-optimize-phase-bug` → `_broadcast_upstream_for_uid` 中**没有** `copy_stage_metadata_from` 调用（grep 0 命中）
  - `2026-04-24 rgba-bg-redesign` → `Trellis2Sample(I2VSample)` + `_composite_rgba_pil` + `_apply_bg_to_condition_images` + `render_bg_color` 全保留
  - `2026-04-25 decode-oom-fix` → `chunked.py:385` 的 `if torch.is_grad_enabled():` 守门保留
- ReadLints 在 `data_utils / hparams / trainers / models/trellis2` 全绿
- **commit 前发现 4 个文件处于 `MM` 状态**，staged 区是上次"曾把 DGA 删除"那次错误回滚的快照（DGA 类被删、`SAMPLER_REGISTRY` 缺一项、Literal 缺第 5 项、缺 `_align_for_distributed_group_aligned` helper）；用 `git add` 把工作树版本重新覆盖 staged 区
- 顺手 stage 了 3 份 trellis2 yaml 的良性注释更新（描述详化 + 命令行示例路径修正）
- `git commit` 生成 merge commit `00a5add`（双 parent: `f4f07c9` + `5c79203`），工作树干净

## 调试经验

- **merge 中途的 staged 区不可信**：可能是中间错误回滚的 frozen 快照。commit 前必须用 `git diff --cached <file>` 与工作树对比，发现差异要 `git add` 重新覆盖。`MM` 状态在 long-lived merge 工作流里特别危险——本次的 4 个文件正是上一 session 笔记里"曾把 DGA 删除→已恢复"那次回滚的痕迹，如果直接 commit 就会 silent regression 丢掉整个 cross-GPU upstream sharing。
- **"待确认"项目大多是良性差异**：
  - `trellis2_grpo.sample()` 不调 `_maybe_offload_samples_to_cpu` → trellis2 默认 `offload_samples_to_cpu=False`，等价 no-op
  - `_id_fields` 加 `negative_prompt` → trellis2 sample 永远 `negative_prompt=None`，hash 不变
  - yaml 加载时 8+7 条"未知 key"WARNING → 全部是 `extra_kwargs` 通路的合法字段，功能正确
  - 表面看 `BaseAdapter._resolve_component_names(None)` 改成枚举 `pipeline.components` 似乎危险，深挖发现 `Trellis2Adapter._resolve_component_names`（trellis2.py L733）+ `_freeze_vae` no-op（L547）+ `_freeze_text_encoders` no-op（L551）三层防御，base class 改动对 trellis2 完全无效
- **`Arguments.from_dict` 的 unknown-key 警告**：upstream 新加的 typo 检测器，对 trellis2 / unified_reward 的 `extra_kwargs` 字段会刷警告，**保留警告比削弱检测更值得**（不修）
- **第一次启动会重建 dataset cache**：upstream consolidate pipeline 改了 `part_arrow_path` 布局（加了 sentinel + `_build_meta.json`），现有 trellis2 cache 会被判 stale 强制重建，一次性代价不可避免

## 参考代码

| 文件 | 关键位置 | 说明 |
|------|---------|------|
| `src/flow_factory/data_utils/sampler.py` | L36/L96/L166/L267 | 4 个 Sampler 类共存（KRepeat / GroupContiguous / **DistributedGroupAligned** / GroupDistributed）|
| `src/flow_factory/data_utils/sampler_loader.py` | `SAMPLER_REGISTRY` | 4 项注册全在 |
| `src/flow_factory/hparams/data_args.py` | `sampler_type` Literal | 5 项（auto + 4 sampler）|
| `src/flow_factory/hparams/args.py` | `_resolve_sampler_type` async-reward 集合 | `{distributed_k_repeat, group_distributed}`，**DGA 不在内** |
| `src/flow_factory/hparams/args.py` | 4 个 `_align_for_<sampler>` helper | dispatcher pattern |
| `src/flow_factory/trainers/grpo.py` | `optimize` lazy stacking + `sample` offload helper | trellis2 兼容（offload=False 时 no-op）|
| `src/flow_factory/trainers/trellis2_grpo.py` | 整文件 diff = 0 | `sample()` 重写 + 不重写 `optimize` |
| `src/flow_factory/trainers/abc.py` | `_maybe_offload_samples_to_cpu` 新增 | trellis2_grpo.sample 没继承调用，但 trellis2 不开 offload，无影响 |
| `src/flow_factory/data_utils/loader.py` | `_create_or_load_dataset(dataset_cls=...)` | Image3DDataset 走同一 consolidate pipeline |
| `src/flow_factory/models/trellis2/trellis2.py` | L547/L551/L733 | `_freeze_vae` / `_freeze_text_encoders` / `_resolve_component_names` 全 override，屏蔽 base class 改动 |
| `src/flow_factory/models/trellis2/chunked.py` | L385 | `if torch.is_grad_enabled():` 守门保留 |
| `examples/grpo/lora/trellis2/{shape,shape_unified_reward,tex_unified_reward}.yaml` | `sampler_type` / `dataset_type` | 迁到子目录，sampler_type=`distributed_group_aligned`，dataset_type=`image_3d` |

## 最终方案

合并完成 + 审计通过 + commit 落盘。核心策略：

1. **4 个 sampler 在 `SAMPLER_REGISTRY` 共存**（DGA + `GroupDistributedSampler`）；upstream 改动只增不删
2. **async-reward override 集合保持 trellis2 行为**：DGA 不被静默回退到 `group_contiguous`
3. **`Image3DDataset` 通过 `dataset_cls=` 参数复用** upstream consolidate pipeline，避免分支
4. **`trellis2_grpo` 重写 `sample`** 实现 cross-GPU upstream sharing，但**继承** `GRPOTrainer.optimize` 的 lazy stacking（GPU-resident 下 no-op）
5. **3 份 trellis2 yaml 迁移到 `examples/grpo/lora/trellis2/{shape,shape_unified_reward,tex_unified_reward}.yaml`**，`sampler_type=distributed_group_aligned`，`dataset_type=image_3d`

回滚保险：备份 `backup/trellis2-before-upstream-merge` 仍指向 `f4f07c9`，可随时 `git reset --hard` 回退。

## 下一步任务

1. **小规模冒烟训练**验证 merge 后训练通路 OK：
   - dataset cache 重建是否顺利
   - reward 计算（unified_reward / default reward）是否正常
   - epoch 0 是否走完（rollout → reward → optimize）且 reward_mean / ratio_mean 合理
2. **冒烟通过后决定 push 策略**：
   - PR 流程：`git push origin trellis2-merge-upstream` → 开 PR review
   - 直接落 trellis2：`git checkout trellis2 && git merge --ff-only trellis2-merge-upstream && git push origin trellis2`
3. **冒烟翻车**：
   - 小问题在 `trellis2-merge-upstream` 分支单独 commit fix（不污染 backup）
   - 大问题 `git reset --hard backup/trellis2-before-upstream-merge` 回到合并前

## 初步方案

冒烟测试推荐路径：

- 拷贝 `examples/grpo/lora/trellis2/shape.yaml` 到 `/tmp/trellis2_smoke.yaml`，临时调小：
  ```yaml
  data:
    max_dataset_size: 16        # 限制小数据集
  training:
    num_train_epochs: 1
    per_device_batch_size: 2
    group_size: 4               # 配合 W=2 → W*B=4, K=4, W*B%K=0 OK
    unique_sample_num_per_epoch: 4
    num_inference_steps: 12     # 维持 trellis2 sparse stage scheduler 对齐
  ```
- 启动：
  ```
  PATH=/home/zhiyuan_ma/anaconda3/envs/grpo3d_trellis2/bin:$PATH \
    CUDA_VISIBLE_DEVICES=0,1 ff-train /tmp/trellis2_smoke.yaml
  ```
- **关注点**：
  1. `_create_or_load_dataset` 是否走完 upstream consolidate pipeline 重建 cache（看到 "Consolidated N part(s) for train split"）；如果旧 cache 命中导致跳过，强制 `force_reprocess: True` 一次再删
  2. `Trellis2Adapter._freeze_vae` no-op 是否生效（不应有 `'NoneType' has no attribute 'requires_grad_'` 错误）
  3. cross-GPU sharing 在 W=2 配置下退化是否正常（DGA owner-broadcast 退化为自拷贝）
  4. epoch 0 末尾 `reward_mean > 0` + `ratio_mean ≈ 1.0` + 训练不 crash
- **潜在风险**：
  - `data.cache_dir: "/data/zhiyuan_ma/.cache/flow_factory/datasets"` 是绝对路径，确保 `/data` 节点可写
  - `Image3DDataset` 仅 override `_load_image`，base class 现在多走 `cache_file_name=target_arrow_path` 路径——理论上兼容但需 runtime 验证
  - `CUDA_VISIBLE_DEVICES=0,1` 下 `Trellis2GRPOTrainer._distributed_upstream_stages` 强依赖 PG 已初始化（`accelerator.gather` 调用），单进程会 raise
- **如果冒烟时遇到 OOM**：先确认 `chunked.py:385` 的守门是否生效（`@torch.no_grad()` rollout 下应该跳过 canonical sort），不行再按 `2026-04-25-trellis2-decode-oom-fix.md` 的 fallback 路径处理
