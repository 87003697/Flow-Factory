# Session Handoff: PickScore Text+Image Sum Reward

## 任务目的

为 Trellis2 3D 纹理生成训练新增一个基于 PickScore 的 reward，同时对渲染视频帧计算 text-frame 和 condition_image-frame 的 CLIP 相似度，帧级求和而非平均，以期比现有的 Qwen-VL side-by-side reward 提供更稳定的训练信号。

## 执行内容

- 探讨了现有 Qwen-VL reward (`qwen_vl_side_by_side`) 改成 PickScore 的可行性及语义差异。
- 规划新 reward `PickScoreTextImageSumRewardModel`：`required_fields = ("prompt", "video", "condition_images")`，text 和 condition image 各只 encode 一次，帧级分数 `scale*dot(text, frame) + scale*dot(cond_img, frame)` 求和，最终 `/ 26`。
- 新建 `src/flow_factory/rewards/pick_score_text_image_sum.py`；`_extract_feature_tensor` 从 `pick_score.py` import 而非复制；`__init__` 与现有 PickScore 完全相同，无额外超参。
- 三层输入校验（参照 `QwenVLSideBySideReward`）：必填字段检查 → batch 长度一致性 → 逐样本非空+带 index 错误信息。
- 在 `registry.py` 注册 `pickscore_text_image_sum` 和 `pickscore_textimage_sum`（两个等价 key，后者为 YAML 名称小写后的兼容 key）。
- 新建 `examples/grpo/lora/trellis2/tex_pickscore_image+text.yaml`（从 Qwen YAML 复制 Trellis2 配置，仅替换 reward 块）；原 `tex_qwen_vl_reward.yaml` 保持原状不改动。
- 在 `guidance/rewards.md` Built-in Reward Models 表格中新增 `PickScore_TextImage_Sum` 条目。
- 将训练命令 `ff-train examples/grpo/lora/trellis2/tex_pickscore_image+text.yaml` 发送到 `screen 3297487.train_flowfactory1`，已确认启动并跑到 Epoch 6，reward 计算步骤正常通过（`Pointwise Rewards: pick_score_text_image_sum: 100%`）。

## 调试经验

- `screen -X stuff` 第一次发送时和残留的旧命令拼接，导致路径被破坏并立刻报 `FileNotFoundError`。正确做法是先用 `$'\025...'`（`^U` 清行）再发送命令，或等确认 prompt 干净后再 stuff。
- `pgrep` 确认进程时发现旧的 `tex_qwen_vl_reward.yaml` 训练进程还在运行（PID 3305593-3305598），可能占用 GPU，若后续出现显存不足应先检查是否残留。

## 参考代码

| 文件 | 关键位置 | 说明 |
|------|---------|------|
| `src/flow_factory/rewards/pick_score_text_image_sum.py` | 全文 | 新 reward 实现 |
| `src/flow_factory/rewards/pick_score.py` | L27-40 `_extract_feature_tensor`; L43-138 `PickScoreRewardModel` | 被 import 的工具函数；对比基准实现 |
| `src/flow_factory/rewards/registry.py` | L29-32 | 已注册 `pickscore_text_image_sum` / `pickscore_textimage_sum` |
| `examples/grpo/lora/trellis2/tex_pickscore_image+text.yaml` | rewards / eval_rewards 块 | 新配置文件，指向 `PickScore_TextImage_Sum` |
| `examples/grpo/lora/trellis2/tex_qwen_vl_reward.yaml` | 全文 | 原 Qwen-VL 配置，未改动，保留对比 |
| `src/flow_factory/rewards/qwen_vl_video_reward.py` | L525-566 `__call__` | 校验写法参考 |

## 最终方案

新建独立的 `PickScoreTextImageSumRewardModel` 类，不修改现有 `PickScoreRewardModel`：
- text 和 condition_image 在 `__call__` 入口各 encode 一次（B 个向量），避免旧实现中 prompt 随 frame batch 重复 encode 的计算浪费。
- 帧级分数：`scale * dot(text_emb, frame_emb) + scale * dot(cond_img_emb, frame_emb)`，权重硬编码为 1，不可配置。
- 样本级汇聚：对帧求和（不是 mean），再 `/ 26` 与现有 PickScore 量纲一致。
- 和 Qwen-VL reward 的关键区别：不构建 side-by-side 拼图，直接用 CLIP embedding 计算 cosine 相似度，本地 GPU 推理，无需 vLLM 服务。

## 下一步任务

观测并验证 `PickScore_TextImage_Sum` 相比现有 Qwen-VL reward 是否提供了更稳定的训练 reward 信号（方差、零标准差组比例等指标）。

## 初步方案

- **wandb 对比**：在 wandb 上对比 `tex_pickscore_image+text` run 和历史 `tex_qwen_vl_reward` run 中的 `reward/pick_score_text_image_sum_mean`、`reward_std`、`zero_std_ratio` 等指标曲线，重点看 reward 的组内标准差和 advantage 分布。
- **关键指标**：`train/reward_{name}_std`（组内方差）和 `train/advantage_std`；若两者均更稳定则方案有效。
- **定量基准**：对比 Qwen-VL reward 的 `zero_std_ratio`（一个 epoch 中标准差接近 0 的组的比例），`PickScore_TextImage_Sum` 因是纯 CLIP 连续分数，预期此比例应更低。
- **如果方差过大**：考虑改回 mean pooling（修改 `_compute_video_scores` 中 `s.sum()` → `s.mean()`），或对 text/image 分项分别加权再合并。
- **如果收敛方向不对**：考虑引入 `PickScore_Rank`（`GroupwiseRewardModel`）替代 pointwise PickScore，利用组内排名信号代替绝对分数。
