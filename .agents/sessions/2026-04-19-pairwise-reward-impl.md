# Session Handoff: Pairwise Reward 实现

## 任务目的

为 Flow-Factory 新增 pairwise win-rate reward（基于 UnifiedReward VLM API），同时修复现有 pointwise `UnifiedRewardAPIBase` 在 `num_workers > 1` 时的 thread-safety bug。

## 执行内容

- 调研确认 pairwise 必须走 `GroupwiseRewardModel`，不能复用现有 `PointwiseRewardModel` 继承链（`RewardProcessor` 按 isinstance 分两个字典调度）
- 从 `_reference_codes/Pref-GRPO/fastvideo/rewards/` 提取 think / flex 两种 prompt 模板和 win-rate 聚合逻辑
- 修复 `unified_reward.py`：`self.client` → `threading.local()` + `_get_client()`；`self.semaphore` → `_async_score_batch()` 内部本地创建；`_query_api_text()` 改为接收 semaphore 参数
- 新建 `unified_reward_pairwise.py`（~580 行）：基类 `UnifiedRewardPairwiseBase` + 4 个子类（ThinkImage, ThinkVideo, FlexImage, FlexVideo）
- `registry.py` 新增 4 条注册项，验证 `get_reward_model_class()` 解析通过
- `guidance/rewards.md` 更新：内置模型表、类继承图、新增 Pairwise Win-Rate Models 章节

## 调试经验

- `asyncio.Semaphore` 必须在 `asyncio.run()` 内部创建才属于当前 event loop，不能存为实例属性
- `httpx.AsyncClient`（AsyncOpenAI 底层）的连接池绑定到创建时的 event loop，跨线程复用会报连接错误
- 不能用多重继承 `(UnifiedRewardAPIBase, GroupwiseRewardModel)` 来复用 transport 层，因为 `RewardProcessor` 的 isinstance 检查会把同一个模型同时放入 pointwise 和 groupwise 字典

## 参考代码

| 文件 | 关键位置 | 说明 |
|------|---------|------|
| `src/flow_factory/rewards/unified_reward.py` | `UnifiedRewardAPIBase.__init__`, `_get_client()`, `_query_api_text()` | thread-safety 修复点 |
| `src/flow_factory/rewards/unified_reward_pairwise.py` | `UnifiedRewardPairwiseBase`, 4 个子类 | 新增的 pairwise 实现 |
| `src/flow_factory/rewards/registry.py` | `_REWARD_MODEL_REGISTRY` | 4 条新增注册项 |
| `src/flow_factory/rewards/reward_processor.py` | L69-76 `_pointwise_models` / `_groupwise_models` | 调度逻辑（未改动，但必须理解） |
| `src/flow_factory/rewards/abc.py` | `GroupwiseRewardModel.__call__` | 接口签名（未改动） |
| `_reference_codes/Pref-GRPO/fastvideo/rewards/unifiedreward_think.py` | `_pairwise_win_rate`, `cal_win_rate_images` | think 模式参考 |
| `_reference_codes/Pref-GRPO/fastvideo/rewards/unifiedreward_flex.py` | `_pairwise_win_rate`, `_iter_category_winners` | flex 模式参考 |
| `_reference_codes/Pref-GRPO/fastvideo/rewards/templates.py` | 全部模板函数 | prompt 模板来源 |
| `examples/grpo/lora/flux1_unified_reward_t2i.yaml` | 完整训练配置 | 现有 pointwise 集成测试参考 |

## 最终方案

新增独立的 `unified_reward_pairwise.py`，与现有 `unified_reward.py` 平行，不共享继承链。基类自带 thread-local client 和 per-call semaphore 设计，天然支持 `async_reward=True, num_workers>1`。同时顺手把现有 pointwise 的 thread-safety bug 也修了，改动约 15 行。

## 下一步任务

集成测试 pairwise reward 功能是否正常工作。

## 初步方案

需要端到端验证 pairwise reward 在 Flow-Factory 训练循环中正常运行，建议分两层测试：

1. **模型实例化 + mock API 冒烟测试**（无需 GPU / VLM 服务）：
   - 从 registry 加载 4 个 pairwise 模型类
   - mock `AsyncOpenAI.chat.completions.create` 返回固定的 think / flex 格式文本
   - 构造 group_size=4 的假 prompt + image/video 输入，调用 `__call__`
   - 验证返回的 `RewardModelOutput.rewards` shape 为 `(4,)`，值在 [0, 1] 范围
   - 验证 flex 模式的 `extra_info` 包含 `overall_win_rate` 和 `dim_mean_win_rate`

2. **真实 VLM 服务集成测试**（需要 GPU + vLLM 服务）：
   - 参考 `examples/grpo/lora/flux1_unified_reward_t2i.yaml` 写一个 pairwise 版本的 YAML
   - 启动 vLLM 服务 + 短训练（`unique_sample_num_per_epoch: 4, group_size: 4`）
   - 检查训练 log 中 reward 值是否合理（非全 0、非全 NaN）
   - 检查 `async_reward=true, num_workers=2` 是否能正常完成而不报连接错误

关键风险点：
- `RewardProcessor._compute_groupwise_rewards()` 是否正确传递 `image`/`video` 字段给 pairwise 模型的 `__call__`
- 分布式场景下 `required_fields` 是否能正确 gather/scatter
- `max_pairs` 随机采样在 group_size 较小时是否退化正常
