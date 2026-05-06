# Session Handoff: Trellis2 Tex GRPO Tuning

## 任务目的
为 `examples/grpo/lora/trellis2/tex_qwen_vl_reward.yaml` 启动 Trellis2 tex-stage GRPO + Qwen-VL reward 训练，并排查启动前的 KL、reasoning、vLLM 和 conda 环境问题。下一步用户要开始调参。

## 执行内容
- 确认 KL regularization 的开关是 `train.kl_beta > 0`；当前配置已设为 `kl_type: 'v-based'` 和 `kl_beta: 1e-3`。
- 确认 `enable_reason: false` 会让 Qwen-VL reward 直接做 1-token Yes/No 打分，不再生成 reasoning，也不会写 `extra_info['reasons']` 到 wandb caption。
- 确认 vLLM server 命令和 YAML 匹配：`served-model-name` 为 `Qwen/Qwen3.5-9B`，HTTP endpoint 为 `http://localhost:8000/v1`，GPU 7 用于服务。
- 排查训练启动失败：旧日志中的根因是 `accelerate` 被解析到 `/home/zhiyuan_ma/.local/bin/accelerate`，进而使用 `/home/zhiyuan_ma/miniconda3` 的 Python 3.13 包，导致 `flash_attn_2_cuda` ABI 不匹配。
- 已通过 `export PATH="$CONDA_PREFIX/bin:$PATH"` 和 `hash -r` 修正命令解析，确认 `python`、`ff-train`、`accelerate` 都指向 `/home/zhiyuan_ma/anaconda3/envs/grpo3d_trellis2/bin/`。

## 调试经验
- 分布式训练里大量 rank 的 `SIGTERM` 是连带现象；真正 root cause 是 first observed failure 前面的 import error。
- `flash_attn_2_cuda undefined symbol` 在这次日志中由错误 Python/accelerate 环境触发，不应先改训练配置。
- `ff-train` 内部调用字符串 `accelerate launch ...`，所以当前 shell 的 `PATH` 会决定实际使用哪个 `accelerate`。
- 若之后仍然报同一个 `flash_attn` ABI 错误，再检查并重装 `grpo3d_trellis2` 环境内与当前 torch 匹配的 `flash-attn`。

## 参考代码
| 文件 | 关键位置 | 说明 |
|------|---------|------|
| `examples/grpo/lora/trellis2/tex_qwen_vl_reward.yaml` | `train`, `rewards`, `eval_rewards` | 当前调参主配置；包含 KL、batch/group、optimizer、Qwen-VL reward 和 vLLM 说明。 |
| `src/flow_factory/rewards/qwen_vl_video_reward.py` | `QwenVLSideBySideReward._score_single` | `enable_reason` 的实际行为：true 时 reason + Yes/No 两次请求，false 时直接 Yes/No。 |
| `src/flow_factory/trainers/grpo.py` | `GRPOTrainer.enable_kl_loss` | KL 是否启用由 `training_args.kl_beta > 0.0` 决定。 |
| `src/flow_factory/cli.py` | `train_cli` | `ff-train` 会调用 `accelerate launch`，依赖 shell `PATH` 找到正确 accelerate。 |

## 最终方案
当前推荐启动方式是先在 `grpo3d_trellis2` 环境里确保 PATH 正确：

```bash
export PATH="$CONDA_PREFIX/bin:$PATH"
hash -r
which python
which ff-train
which accelerate
```

然后 GPU 7 跑 vLLM server，GPU 0-6 跑训练：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 ff-train examples/grpo/lora/trellis2/tex_qwen_vl_reward.yaml
```

## 下一步任务
开始调 `tex_qwen_vl_reward.yaml` 的 Trellis2 tex GRPO 训练参数，重点观察训练稳定性、reward 吞吐、显存和 wandb 指标。

## 初步方案
- 先用当前配置跑通一小段，确认 vLLM endpoint、数据 cache、Trellis2 render、reward 和 distributed group-aligned sampler 都正常。
- 优先观察 `kl_div` / `kl_loss`、reward 均值与方差、advantage 分布、OOM/吞吐；`kl_beta: 1e-3` 是当前 v-based KL 起点。
- 如果 reward server 成为瓶颈，先调低 `num_workers` 或 `max_concurrent`；如果训练侧显存紧张，再考虑 `per_device_batch_size`、`ref_param_device` 或 render 参数。
- 由于 `enable_reason: false`，当前 reward 更快但不可在 wandb caption 中查看 VLM reasoning；如需要诊断 reward 判断质量，可临时打开 reason 并降低并发。
