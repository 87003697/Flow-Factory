# Session Handoff: Pairwise Reward 架构重构

## 任务目的

审查并重构 `unified_reward_pairwise.py` 的类层次和私有函数封装设计，消除 Mixin 多继承、重复的 score/aggregate/call 管道、以及 flex 概念对 think 的泄漏。

## 执行内容

- 审查了 7 个设计问题（基类职责、Mixin 必要性、score_pair 重复、aggregate 重复、__call__ 入口覆盖、4 个子类差异、vs Pref-GRPO 函数式设计）
- 删除 `_ThinkParseMixin`（1 方法）和 `_FlexParseMixin`（7 方法 + `_flex_call` 入口覆盖），消除 Mixin 多继承
- 新增 `PairResult` dataclass 统一 `_score_pair` 返回值
- 统一 `_score_pair` + `_score_pair_flex` → 单一 `_score_pair` 返回 `PairResult`
- 将 `_parse_winner` (abstract) → `_parse_response` (abstract)，返回 `(overall_winner, cat_winners)` 元组
- 解析逻辑提取为模块级纯函数：`_parse_think_winner`、`_parse_flex_response`、`_parse_json_payload`、`_normalize_winner`、`_iter_category_winners`
- 提取 `_sample_frames` 为模块级纯函数（消除 video 子类间重复）
- 删除 `_flex_call`，统一为基类 `__call__` 单一入口
- 将 flex 权重配置（`overall_weight`/`dim_weight`/`category_weights`）从基类移出：
  - 基类 `_aggregate_win_rate` 简化为 overall-only
  - `_init_flex_weights` 和 `_aggregate_flex_win_rate` 提取为模块级函数
  - Flex 子类 override `_aggregate_win_rate` 委托给 `_aggregate_flex_win_rate`
- 基类 `UnifiedRewardPairwiseBase` 现在零 flex 概念

## 参考代码

| 文件 | 关键位置 | 说明 |
|------|---------|------|
| `src/flow_factory/rewards/unified_reward_pairwise.py` | 全文（893 行） | 重构后的完整实现 |
| `src/flow_factory/rewards/registry.py` | L39-42 | 4 个 pairwise 注册项（路径未变） |
| `src/flow_factory/rewards/abc.py` | `GroupwiseRewardModel.__call__` | 接口约束（未改动） |
| `src/flow_factory/rewards/reward_processor.py` | L69-76 | 调度逻辑（未改动） |

## 最终方案

```
模块级纯函数:
  _parse_think_winner, _parse_flex_response, _parse_json_payload,
  _normalize_winner, _iter_category_winners, _sample_frames,
  _init_flex_weights, _aggregate_flex_win_rate

UnifiedRewardPairwiseBase(GroupwiseRewardModel)    # 零 flex 概念
├── __init__()              # API 传输配置
├── _get_client()           # thread-local AsyncOpenAI
├── _query_api_text()       # 指数退避重试
├── _build_pair_messages()  # abstract
├── _parse_response()       # abstract → (winner, cat_winners)
├── _score_pair()           # 统一编排 → PairResult
├── _aggregate_win_rate()   # overall-only 聚合
└── __call__()              # 唯一入口

4 个具体类（单继承，无 Mixin）:
├── UnifiedRewardThinkImagePairwise   # _build_pair_messages + _parse_response
├── UnifiedRewardThinkVideoPairwise   # + max_frames
├── UnifiedRewardFlexImagePairwise    # + _init_flex_weights + override _aggregate_win_rate
└── UnifiedRewardFlexVideoPairwise    # + max_frames + 同上
```

## 下一步任务

集成测试 pairwise reward 重构后功能是否正常。

## 初步方案（集成测试计划）

### 第 1 层：Mock API 冒烟测试（无需 GPU / VLM 服务）

1. **Think 模式解析正确性**
   - mock `AsyncOpenAI.chat.completions.create` 返回包含 `<answer>Image 1 is better</answer>` 的文本
   - 构造 `group_size=4` 的假 prompt + PIL Image 列表
   - 调用 `UnifiedRewardThinkImagePairwise.__call__`
   - 断言 `rewards.shape == (4,)`，值在 `[0, 1]`，winner 的 win rate > loser
   - 对 Video 变体同理，传入 `List[List[Image]]`

2. **Flex 模式解析 + 加权聚合正确性**
   - mock API 返回固定 JSON（含 `winner` + `categories[].cat_winner`）
   - 调用 `UnifiedRewardFlexImagePairwise.__call__`
   - 断言 `extra_info` 包含 `overall_win_rate` 和 `dim_mean_win_rate`
   - 用已知 JSON 手算期望值，对比 `rewards` 数值

3. **边界情况**
   - `group_size=2`（只有 1 对）→ rewards shape `(2,)` 且和为 1.0
   - `max_pairs=1` + `group_size=4` → 只评估 1 对，其余样本 win_rate=0.5
   - API 全部失败（重试耗尽）→ 抛出 `RuntimeError`

4. **`_parse_response` 健壮性**
   - 无 `<answer>` 标签 → `(None, [])` → tie 处理
   - JSON 格式损坏 → `(None, [])` → 不崩溃
   - `cat_winner` 字段缺失 → 该 category 的 winner 为 None → tie

### 第 2 层：真实 VLM 服务端到端测试（需要 GPU + vLLM）

5. **短训练冒烟**
   - 基于 `examples/grpo/lora/flux1_unified_reward_t2i.yaml` 改写 pairwise YAML
   - `unique_sample_num_per_epoch: 4, group_size: 4`，跑 1 epoch
   - 检查训练 log 中 reward 值合理（非全 0、非全 NaN、有区分度）

6. **多线程安全**
   - `async_reward=true, num_workers=2`
   - 确认无 httpx 连接池错误或 event loop 冲突

### 风险点

- `RewardProcessor._compute_groupwise_rewards()` 是否正确传递 `image`/`video` 给 pairwise `__call__`
- 分布式场景下 `required_fields` gather/scatter 是否正确
- flex `_aggregate_win_rate` override 在多线程下是否安全（应该是，因为无共享状态）
