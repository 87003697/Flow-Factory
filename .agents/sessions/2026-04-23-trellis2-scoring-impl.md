# Session Handoff: Trellis2 Scoring 改进实现

## 任务目的
实现 Trellis2 GRPO 训练的评分改进方案：RGBA 图像处理优化、可配置渲染背景色、起始方位角校正、日志可视化增强、UniReward 集成。

## 执行内容
- `pipeline.py`：`preprocess_image` 改为返回 RGBA PIL 图像，`get_cond` 内部处理 RGBA→RGB(黑底) 转换
- `trellis2.py`：新增 `_composite_rgba` 和 `_split_rgba_image` 两个工具函数；`Trellis2Sample.__post_init__` 拦截 RGBA 图像存储 tensor，`set_render_bg_color` 支持动态背景色切换
- `trellis2.py`：`render_latents` 接受 `bg_color` 参数，起始 yaw 改为 `np.pi`（正面朝前）
- `trellis2_grpo.py`：`_render_kwargs` 新增 `render_bg_color` 从 `model_args.extra_kwargs` 读取
- `formatting.py`：`LogFormatter` 新增 `ImageConditionSample` 处理，复用 `LogTable.from_i2v_samples` 展示条件图 + 多视角视频
- `registry.py` + `unified_reward.py` + `unified_reward_pairwise.py`：从 upstream 合并 6 个 UniReward 模型注册
- 所有 trellis2 YAML 新增 `render_bg_color` 参数，`trellis2_shape.yaml` 改为白色 `[1,1,1]`
- 新建 `trellis2_shape_unified_reward.yaml`，reward 配置为 `unified_reward_image_acs`

## 调试经验
- RGBA 处理经过三轮重构：最终方案是 `_split_rgba_image`（PIL→tensor+RGB-PIL）+ `_composite_rgba`（tensor 上做 alpha 合成），职责分离最清晰
- `get_cond` 内部做 RGBA→RGB 转换比在调用侧做更简洁，因为有两处调用 `get_cond`
- `_condition_images_rgba` 用 `Dict[int, Tensor]` 比 `List[Optional[Tensor]]` 好——稀疏存储，遍历时无需 None 检查

## 参考代码
| 文件 | 关键位置 | 说明 |
|------|---------|------|
| `src/flow_factory/models/trellis2/trellis2.py` | `_composite_rgba` (L83-94), `_split_rgba_image` (L97-110), `Trellis2Sample.__post_init__` (L252-261), `set_render_bg_color` (L263-275) | RGBA 处理核心逻辑 |
| `src/flow_factory/models/trellis2/pipeline.py` | `preprocess_image` (L351-352), `get_cond` (L371-377) | 图像预处理 + RGBA→RGB 转换 |
| `src/flow_factory/trainers/trellis2_grpo.py` | `_render_kwargs` (L89-94) | 渲染参数传递 |
| `src/flow_factory/logger/formatting.py` | `_process_image_condition_samples` | 日志可视化 |
| `src/flow_factory/rewards/unified_reward.py` | `UnifiedRewardAPIBase`, `UnifiedRewardImageACS` | UnifiedReward API 客户端 |
| `examples/grpo/lora/trellis2_shape_unified_reward.yaml` | reward 部分 | UnifiedReward 训练配置 |

## 最终方案
采用两层分离的 RGBA 处理架构：
1. **存储层**：`Trellis2Sample.__post_init__` 用 `_split_rgba_image` 将 RGBA PIL 拆为 `(4,H,W)` float32 tensor（存到 `_condition_images_rgba` dict）+ RGB-on-black PIL（传给父类供推理用）
2. **合成层**：`set_render_bg_color` 用 `_composite_rgba` 在 tensor 上做 alpha 合成，按需换背景色
3. **推理兼容**：`get_cond` 内部自动将 RGBA PIL 合成为 RGB-on-black 传给图像编码器

渲染背景改为白色 `[1,1,1]`；reward 使用 `unified_reward_image_acs` 通过 vLLM API 评估渲染图的 Alignment/Coherence/Style。

## 下一步任务
启动 vLLM 服务部署 UnifiedReward 模型，然后使用 `trellis2_shape_unified_reward.yaml` 运行 Trellis2 GRPO 训练。

## 初步方案
- 用 vLLM 启动 UnifiedReward 模型服务，监听 `http://localhost:8080/v1`，确保与 YAML 中 `api_base_url` 一致
- 确认 UnifiedReward 模型权重路径和 `--served-model-name UnifiedReward` 参数
- 启动训练：`accelerate launch --config_file ... train.py examples/grpo/lora/trellis2_shape_unified_reward.yaml`
- 潜在风险：vLLM 服务网络超时或 OOM，可调整 `max_concurrent`、`timeout`、`batch_size` 参数
- 关注首轮 reward 输出是否合理（ACS 三个子分数各自范围），必要时调整 `alignment_weight`/`coherence_weight`/`style_weight`
