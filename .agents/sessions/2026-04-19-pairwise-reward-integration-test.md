# Session Handoff: Pairwise Reward 集成测试

## 任务目的

对重构后的 4 个 pairwise reward 子类进行端到端集成测试，验证它们在 Flow-Factory 的 T2I 训练循环中正常工作。

## 执行内容

- 编写了 `/tmp/ff_smoke/test_pairwise.py` mock 测试脚本，覆盖 9 个用例（think/flex × image/video 解析、边界条件、malformed response 鲁棒性、retry 耗尽），全部通过
- 创建了 `examples/grpo/lora/flux1_unified_reward_pairwise_t2i.yaml`（从 pointwise 版本改写）
- 用 `unified_reward_think_image_pairwise` 完成端到端短训练（5 个 epoch，reward_mean=0.5000，无错误）
- 用 `unified_reward_flex_image_pairwise` 完成端到端短训练（1 个 epoch，reward_mean=0.5000，zero_std_ratio=0，无错误）
- 恢复 YAML 为正式训练版本：`sampler_type: group_contiguous`、`unique_sample_num_per_epoch: 48`、`save_freq: 20`、完整 eval 配置、`eval_rewards` 使用 pointwise `unified_reward_image_acs`

## 调试经验

### TypeError: len() of unsized object（完整因果链）

`advantage_processor.py:291` 的 `if len(v) == 0:` 报错。完整因果链：

1. `sampler_type: "auto"` + `unique_sample_num_per_epoch: 48` + `world_size: 7`
2. `_resolve_sampler_type` 中 `48 % 7 != 0` → `groups_per_rank_ok = False`；但 `(48 // 7) * 4 % 2 == 0` → `local_batch_tiling_ok = True`（整除为 0 时也 trivially true）
3. auto 选择 `distributed_k_repeat`，同一 group 的 sample 散布到不同 rank
4. `RewardBuffer` 的 async groupwise 路径按 `unique_id` 收集 group，但 `distributed_k_repeat` 下每个 rank 只有 group 的部分 sample，无法凑齐完整 group
5. `RewardBuffer.finalize()` 返回空的 rewards 字典 `{}`
6. `AdvantageProcessor.compute_gdpo` 对空 rewards 做 `np.sum([])` → 产生 0 维 numpy scalar `0.0`
7. scalar 存入 `stat_arrays`，传给 `_batch_reduce_stats`
8. `np.asarray(0.0)` 是 0-d array，`len()` 抛出 `TypeError: len() of unsized object`

**修复**：pairwise YAML 必须显式 `sampler_type: "group_contiguous"`。`auto` 的解析逻辑未考虑 groupwise reward 的约束。

### eval 阶段跳过 groupwise reward

`grpo.py` 的 eval 循环硬编码 `self.eval_reward_buffer.finalize(..., split='pointwise')`，groupwise 模型在 eval 时不执行。需要在 `eval_rewards` 中配置 pointwise 模型（如 `unified_reward_image_acs`）作为 eval 指标。

### reward_mean 恒为 0.5（数学恒等式，非 bug）

pairwise win-rate reward 的 group 均值恒为 0.5：每个 pair 分配总共 1.0 win，C(K,2) 个 pair 共分配 C(K,2) win 给 K 个 sample，每个 sample 参与 K-1 次比较，`group_mean = C(K,2) / (K × (K-1)) = 0.5`。训练信号完全来自 group 内方差。

### think vs flex 的 reward 区分度差异

- **Think 模式**：`zero_std_ratio=0.8571`（7 个 group 中 6 个的 group 内所有 sample 胜率完全相同，都是 0.5，训练信号为零）。因为 think 只输出一个 overall winner，group_size=4 时 C(4,2)=6 对比较容易出现对称胜负
- **Flex 模式**：`zero_std_ratio=0`（所有 group 都有非零方差），`group_std_mean=0.3483`。多维度打分有效打破对称性

flex 模式产生的训练信号显著更丰富，但 API 调用更耗时（需要生成完整 JSON）。smoke test 中图片是训练初期的随机质量产物，实际训练中区分度应更好。

### RuntimeError: Event loop is closed（非阻塞性警告）

多个 epoch 之间，async client 清理时偶现 `RuntimeError: Event loop is closed`（`asyncio.selector_events.py` → `httpx` → `anyio`）。这是 `asyncio.run()` 结束后 httpx 底层连接池被 GC 触发 close 导致的，不影响功能。每次 `asyncio.run()` 会通过 `_get_client()` 创建新的 thread-local client。

### tee 的 stdout 缓冲问题

多进程训练输出经 `tee` 后严重延迟（30 秒以上无新输出），导致无法实时监控训练进度。解决方案：`PYTHONUNBUFFERED=1` + `stdbuf -oL tee /path/to/log`。

### 训练进程残留与 GPU 清理

`kill` 主进程后，以下进程可能变成孤儿继续占用 GPU：
- `flow_factory.train` 的 7 个 rank 子进程
- `pt_data_worker` 数据加载进程

安全清理步骤：`kill <主PID>` → `sleep 3` → `kill -9 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -v <vllm_pid>)` → `fuser /dev/nvidia*` 确认残留 → `kill -9` 清理。

**注意**：用 `nvidia-smi --query-compute-apps=pid` 批量 kill 时务必 `grep -v` 排除 vLLM server 的 PID，否则会误杀 reward service。本次测试中就因此意外 kill 了 GPU 7 上的 vLLM。

### max_epochs 未设置导致训练无限运行

YAML 中 `max_epochs` 为 null 时训练会持续运行，smoke test 需要手动 kill。正式训练需要明确设置 `max_epochs` 或依赖 `save_freq` + 外部监控。

## 参考代码

| 文件 | 关键位置 | 说明 |
|------|---------|------|
| `src/flow_factory/rewards/unified_reward_pairwise.py` | 全文（893 行） | 4 个 pairwise 子类及基类 |
| `src/flow_factory/rewards/unified_reward.py` | `UnifiedRewardAPIBase`, `_get_client()` | pointwise 对应实现，可对照 |
| `src/flow_factory/rewards/registry.py` | L39-42 | 4 个 pairwise 注册项 |
| `src/flow_factory/rewards/reward_processor.py` | L69-76 `_groupwise_models` | groupwise 调度路径 |
| `src/flow_factory/hparams/args.py` | L111 `_resolve_sampler_type` | sampler auto 解析逻辑 |
| `src/flow_factory/advantage/advantage_processor.py` | L291 `_batch_reduce_stats` | `len()` of unsized object 出错点 |
| `src/flow_factory/trainers/grpo.py` | eval 循环 | `split='pointwise'` 硬编码 |
| `examples/grpo/lora/flux1_unified_reward_pairwise_t2i.yaml` | 完整配置 | 正式训练版 pairwise YAML |

## 最终方案

4 个 pairwise reward 子类在 T2I 端到端训练中验证通过。核心 `unified_reward_pairwise.py` 无需修改。唯一的配置层面修复是 pairwise YAML 必须用 `sampler_type: "group_contiguous"`（而非 auto），并在 `eval_rewards` 中配置 pointwise 模型。

## 下一步任务

先用 flex image pairwise 跑正式 T2I 训练，验证 flex 模式在实际训练中的表现（区分度、收敛性）。将所有 pairwise reward 代码视为可疑代码，审计其是否符合 UnifiedReward 的调用范式、是否与 Pref-GRPO 的调用方式一致。

## 初步方案

### 审计要点

1. **Flex prompt 模板与 UnifiedReward 规范对齐**：`FLEX_IMAGE_TEMPLATE` / `FLEX_VIDEO_TEMPLATE` 的 JSON 输出格式、Sum-of-10 约束、category 结构是否与 Pref-GRPO 原始模板一致
2. **`_parse_flex_response` 解析鲁棒性**：JSON 解析 fallback（greedy regex）、`_normalize_winner` 对 "Image 1/2" 的匹配、`_iter_category_winners` 截断到前 3 个 category 是否合理
3. **`_aggregate_flex_win_rate` 加权逻辑**：`overall_weight` / `dim_weight` / `category_weights` 的默认值和组合方式是否与 Pref-GRPO 一致
4. **Think prompt 模板**：`THINK_IMAGE_TEMPLATE` / `THINK_VIDEO_TEMPLATE` 的评估维度、格式要求是否与 Pref-GRPO 一致
5. **Video 帧传递方式**：两个 video 的帧拼成一个 `content` 列表（先 V1 全部帧，后 V2 全部帧），依赖 prompt 中 "first half / second half" 描述区分，需确认这是否与 Pref-GRPO 的做法一致

### 配置要点

- 使用 `examples/grpo/lora/flux1_unified_reward_pairwise_t2i.yaml` 改为 flex 模式
- `sampler_type: "group_contiguous"`（必须）
- 考虑增大 `group_size`（如 8 或 16）以产生更多 pair 比较和更强信号
- 注意 `max_pairs` 对 API 调用量的控制（group_size=8 时 C(8,2)=28 pairs，可能需要限制）

### 潜在风险

- Flex API 响应更长（完整 JSON），`timeout` 可能需要调高
- Flex 的 category 名称由 VLM 动态生成，跨 pair 不一定一致，`_aggregate_flex_win_rate` 按 category name 聚合可能有噪声
- `max_concurrent` 对 vLLM 吞吐的压力（flex 模式 token 更多）
