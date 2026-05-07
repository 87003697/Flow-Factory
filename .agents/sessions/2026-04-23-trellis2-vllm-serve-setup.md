# Session Handoff: Trellis2 vLLM Serve 部署与调试

## 任务目的
部署 UnifiedReward-2.0 vLLM 服务，修正 reward 配置为 video APS 方案，前台试跑 Trellis2 GRPO 训练验证端到端链路。

## 执行内容
- 创建 `vllm` conda 环境 (Python 3.11) 并安装 vllm 0.19.1
- 创建 `vllm-serve` screen session，在 GPU 7 上启动 `CodeGoat24/UnifiedReward-2.0-qwen35-9b` 服务（port 8080，已验证 `/v1/models` 正常返回）
- 修正 `trellis2_shape_unified_reward.yaml`：`unified_reward_image_acs` → `unified_reward_video_aps`（Trellis2 渲染 multi-view 存在 `sample.video`，需要 video 评分方案评估所有视角）
- YAML 同步修改：`render_num_frames: 24` → `16`；`batch_size: 8` → `4`；`coherence_weight` → `physics_weight`；新增 `max_frames: 32`
- 文件头部注释从 ACS 更新为 APS
- 在 `unified_reward.py` 和 `reward_processor.py` 插入 3 处临时 debug 日志（**尚未清理**）
- 前台试跑训练命令遇到两个错误（均已定位，尚未修复）

## 调试经验
- `ff-train` 的 shebang 指向正确的 `grpo3d_trellis2/bin/python3.10`，但 `accelerate launch` 的子进程会继承 shell PATH，如果 PATH 中 miniconda3 排在前面就会用错 Python。解法：在命令前显式设置 `PATH=/home/zhiyuan_ma/anaconda3/envs/grpo3d_trellis2/bin:$PATH`
- `conda activate` 在 Cursor Shell tool 中不可靠（非 interactive shell），使用完整路径更稳妥
- vLLM screen 中同样用完整路径 `/home/zhiyuan_ma/anaconda3/envs/vllm/bin/python -m vllm.entrypoints.openai.api_server` 替代 `vllm serve`

## 当前阻塞问题
reward model 加载失败，错误信息：
```
ImportError: Failed to load reward model 'unified_reward_video_aps'.
Available models: ['pickscore', ..., 'unified_reward_video_aps', ...]
```
模型名在 registry 中存在，但 `get_reward_model_class` 动态 import 时抛出了 ImportError（被外层 catch 后重包装，原始错误被吞掉）。最可能原因：
1. `openai` 包未安装在 `grpo3d_trellis2` 环境中（`UnifiedRewardAPIBase.__init__` 第一行就 `import openai`）
2. 或者 `unified_reward.py` 中新插入的 debug 代码有语法问题

需要在 `loader.py:64` 的 `except ImportError as e` 中打印 `e` 的原始 traceback 来定位根因。

## 临时 debug 代码清单

以下 3 处 debug 日志需要在确认链路通顺后清理：

### 1. `src/flow_factory/rewards/unified_reward.py` L552-560
`UnifiedRewardVideoGenAPSRewardModel.__call__` 入口，打印传入的 video/condition_images 形状：
```python
        logger.info(
            "DEBUG VideoAPS.__call__: prompt=%d, video=%s (frames=%s, type=%s), "
            "condition_images=%s",
            len(prompt),
            len(video) if video else None,
            [len(v) for v in video[:2]] if video else None,
            type(video[0][0]).__name__ if video and video[0] else None,
            len(condition_images) if condition_images else None,
        )
```
清理方式：删除 L552-560 这 9 行。

### 2. `src/flow_factory/rewards/unified_reward.py` L689-694
`UnifiedRewardVideoGenAPSRewardModel._score_single` 返回前，打印 VLM 原始响应和解析分数：
```python
        scores = self._scores_from_text(text)
        logger.info(
            "DEBUG video._score_single: raw_text=%r, scores=%s",
            text[:300], scores,
        )
        return scores
```
清理方式：将 L689-694 恢复为原来的 `return self._scores_from_text(text)`。

### 3. `src/flow_factory/rewards/reward_processor.py` L157-160 + L19 + L23
`_compute_pointwise_batch` 中 `model(**batch_input)` 前打印 keys：
```python
        logger.info(
            "DEBUG _compute_pointwise_batch: model=%s, keys=%s, batch_size=%d",
            name, list(batch_input.keys()), len(batch_samples),
        )
```
清理方式：删除 L157-160 这 4 行，同时删除文件顶部新增的 `import logging`(L22) 和 `logger = logging.getLogger(__name__)`(L24)。

### 需要补充的 debug 代码

当前 debug 覆盖了 reward 调用侧，但还缺少以下关键环节的可见性：

**建议在修复 ImportError 后、重新试跑前追加：**

4. **`loader.py` L64**：当前 `except ImportError as e` 吞掉了原始错误。建议临时加一行 `import traceback; traceback.print_exc()` 在 raise 前，方便定位根因。（这条应在首次排查时立即加，排查完立即删）

5. **`trellis2_grpo.py` `_rollout_group` 返回前**（约 L164）：打印每个 sample 的 `video` 形状和 `condition_images` 类型，确认从渲染到 reward 的数据交接正确：
```python
for s in sample_batch[:2]:
    logger.info("DEBUG _rollout_group: video=%s, cond_imgs=%s",
                s.video.shape if s.video is not None else None,
                type(s.condition_images))
```

6. **`unified_reward.py` `_build_messages` 返回前**（约 L662）：打印发给 VLM 的 image 数量，确认 condition image + 16 帧 = 17 张：
```python
logger.info("DEBUG _build_messages: %d images in message", len(content) - 1)
```

## 参考代码
| 文件 | 关键位置 | 说明 |
|------|---------|------|
| `examples/grpo/lora/trellis2_shape_unified_reward.yaml` | rewards / eval_rewards 部分 | 已改为 `unified_reward_video_aps`，`max_frames: 32` |
| `src/flow_factory/rewards/unified_reward.py` | `__call__` (L552), `_score_single` (L689) | 2 处 DEBUG 日志 |
| `src/flow_factory/rewards/reward_processor.py` | `_compute_pointwise_batch` (L157), 顶部 import (L22,24) | 1 处 DEBUG 日志 + logger import |
| `src/flow_factory/rewards/loader.py` | `load_reward_model` (L59-69) | ImportError 被 catch 后重包装，原始错误丢失 |
| `src/flow_factory/rewards/registry.py` | `_REWARD_MODEL_REGISTRY` (L34-39) | 6 个 UnifiedReward 模型注册 |
| `src/flow_factory/trainers/trellis2_grpo.py` | `_rollout_group` (L143-164) | 渲染结果传递给 reward 的交接点 |

## 当前环境状态
- `vllm-serve` screen 已运行，GPU 7 占用 ~89GB，`curl localhost:8080/v1/models` 返回正常
- `vllm` conda 环境：Python 3.11 + vllm 0.19.1
- 训练环境：`grpo3d_trellis2` (Python 3.10)，需确认 `openai` 包是否已安装
- 3 处 debug 日志已插入，建议追加 3 处（共 6 处）

## 下一步任务
1. 定位并修复 `unified_reward_video_aps` 加载失败的 ImportError 根因
2. 补充 debug 代码（loader traceback、rollout video 形状、messages image 数量）
3. 成功启动训练后观察首轮 APS 分数是否合理
4. 确认链路无误后清理全部 debug 代码
5. **整理 Trellis2 统一环境安装方案**：当前 `grpo3d_trellis2` 环境的 PATH 中混入了 miniconda3，导致 `accelerate launch` 子进程用错 Python，需要靠 `PATH=...` hack 绕过。正式跑通后需要清理环境，确保 `conda activate grpo3d_trellis2 && ff-train ...` 一条命令就能正常工作，方便在其他服务器上复现。

## 初步方案

### A. 修复 ImportError 并跑通训练
- 先在 `grpo3d_trellis2` 环境中手动测试 `from flow_factory.rewards.unified_reward import UnifiedRewardVideoGenAPSRewardModel`，看真实 ImportError
- 如果是缺 `openai` 包：`pip install openai`
- 如果是 debug 代码语法问题：检查 `unified_reward.py` 的 diff
- 补充建议的 3 处 debug 代码
- 修复后重新跑训练，观察 DEBUG 日志确认数据流正确

### B. 统一环境安装方案
当前问题根因：`~/.bashrc` 或 conda 初始化将 miniconda3 加入了 PATH，导致即使 activate 了 grpo3d_trellis2，`accelerate launch` 的子进程仍可能 resolve 到 miniconda3 的 python/diffusers/flash_attn。

目标：一个干净的 conda 环境 + `pip install -e .` 即可运行 Trellis2 全链路（包括 UnifiedReward），无需任何 PATH hack。

要做的事：
- 排查 `grpo3d_trellis2` 环境中 `which python`、`which accelerate` 是否指向正确位置，`sys.path` 是否混入了 miniconda3 的 site-packages
- 如果是 `.bashrc` / `.condarc` / `PYTHONPATH` 环境变量污染，清理掉
- 确认 `grpo3d_trellis2` 中的 `flash_attn`、`diffusers`、`openai` 等依赖版本正确且自洽
- 整理一份可复现的安装脚本或文档（`conda create` + `pip install` 步骤），确保在新服务器上可以一键搭建
- 可考虑在 `pyproject.toml` 的 `[project.optional-dependencies]` 中增加 `trellis2` extra，收纳 trellis2 特有依赖（如 `openai`、trellis2 第三方包等）
