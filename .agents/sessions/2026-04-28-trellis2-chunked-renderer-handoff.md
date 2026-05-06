```markdown
## Session Handoff: Trellis2 Chunked Renderer / OOM Follow-up / Qwen3.5-VL Reward Plan

日期：2026-04-28

本 handoff 记录本次 session 围绕 Trellis2 渲染器迁移、`nvdiffrast` overflow、OOM 风险、reward 稳定性观察，以及下一阶段 Qwen3.5-VL reward model 设计的全部上下文。

## 当前目标

短期目标：

- 明天继续观察 tex GRPO 训练是否还会 OOM。
- 如果训练稳定，不再 OOM，就清理剩余 debug / memory probe 代码。
- 同时观察 reward 指标是否稳定，尤其确认换 chunked renderer 后 reward 曲线和 logged videos 是否一致。

中期目标：

- 设计新的 reward model，不再依赖当前 UniReward。
- 新 reward model 使用 pretrained Qwen3.5-VL。
- 输入不是旧的 frame concat 形式，而是仿照 wandb logging 的左右拼接视频：
  - 左侧：condition image
  - 右侧：rendered video frame
  - 每一帧都保持这种 side-by-side 布局
- 将拼接后的视频送入 Qwen3.5-VL 做打分。

## 背景问题

最初遇到的问题是 Trellis2 PBR renderer 在大 mesh 上触发 `nvdiffrast` 内部限制：

```text
RuntimeError: subtriangle count overflow
```

根据排查，根因是 generated mesh 的 face count 可能超过 `nvdiffrast` rasterization 内部 buffer 限制，大约在 `2^24 ≈ 16.7M` triangles 附近。旧代码中曾有：

```python
mesh.simplify(16777216)
```

但这个 simplify 不是一个可靠的防溢出机制：

- 它会改变 mesh 拓扑和几何细节。
- 它不保证所有 case 都低于 `nvdiffrast` 实际内部限制。
- 它不能解决“希望保留高面数 mesh 但绕开 rasterizer 单次 face 上限”的问题。

因此本 session 选择移植 reference repo 里的 chunked PBR renderer。

## 参考实现

参考代码位置：

```text
/home/zhiyuan_ma/code/flow_grpo_custom/edit4shape/renderers/pbr_peeled_trellis2.py
```

该实现的关键思路：

- 将 `faces` 按固定大小分 chunk。
- 每个 chunk 单独跑 `dr.DepthPeeler`。
- 收集每个 chunk / layer 的 depth、alpha、shaded、normal 等结果。
- 在 chunk 之间做 per-pixel depth sort。
- 再做 front-to-back alpha compositing。

## 已完成的主要改动

### 1. 新增 chunked PBR renderer

新增文件：

```text
src/flow_factory/models/trellis2/pbr_mesh_renderer_chunked.py
```

核心常量：

```python
_MAX_FACES_PER_CHUNK = 4_000_000
```

设计意图：

- `nvdiffrast` 单次 rasterization 不再直接吃完整 `mesh.faces`。
- 如果一个 mesh 有 40M faces，会被拆成约 10-11 个 chunk。
- 每个 chunk 最多 4M faces，低于 `nvdiffrast` 的 subtriangle overflow 风险区间。
- 最终通过 per-pixel depth sort + alpha compositing 恢复等价渲染结果。

### 2. 与上游 PBR renderer 对齐

移植时保留了上游 `trellis2.renderers.pbr_mesh_renderer.PbrMeshRenderer` 的核心行为：

- face normal 计算方式保持一致。
- normal two-sided 翻转逻辑保持一致。
- PBR attribute sampling 保持一致。
- envmap shading 保持一致。
- SSAO + background + tonemapping 顺序保持一致。
- tonemapping 使用上游的 `aces_tonemapping(...)` + `gamma_correction(...)`。

注意：reference `flow_grpo_custom` 中使用过 `srgb_transfer`，但 HuggingFace mirror 上的 upstream `microsoft/TRELLIS.2` 使用的是 `gamma_correction`。当前 Flow-Factory chunked renderer 选择对齐 upstream，而不是 reference 的自定义 `srgb_transfer`。

### 3. 简化 renderer options 注入

按用户要求，`PbrMeshRendererChunked.__init__` 不再使用 `rendering_options` dict / `edict` 注入，而是直接用显式 keyword arguments：

```python
PbrMeshRendererChunked(
    resolution=512,
    near=1.0,
    far=100.0,
    ssaa=2,
    peel_layers=8,
    device='cuda',
)
```

内部访问也从 `self.rendering_options["resolution"]` 改为 `self.resolution`。

### 4. 修改 `render_latents` 使用 chunked renderer

主要集成点：

```text
src/flow_factory/models/trellis2/trellis2.py:render_latents
```

旧路径：

```python
ret = render_utils.render_frames(...)
```

新路径：

```python
ret = render_frames_chunked(
    mesh, extrinsics, intrinsics,
    {'resolution': resolution},
    envmap=envmap,
    verbose=render_kwargs.pop('verbose', False),
    **render_kwargs,
)
```

### 5. 简化 tensor data flow

旧 upstream `render_utils.render_frames` 会把每帧 renderer output 转成 uint8 numpy，然后 `trellis2.py` 中又转回 float tensor。

新实现去掉这段 round trip。`render_frames_chunked` 直接返回 stacked CUDA tensor：

- `ret['shaded']`: `(T, 3, H, W)` float CUDA `[0, 1]`
- `ret['alpha']`: `(T, 1, H, W)` float CUDA `[0, 1]`

然后直接在 CUDA 上做背景合成：

```python
shaded = ret['shaded']                                       # (T, 3, H, W)
alpha = ret['alpha']                                         # (T, 1, H, W)
bg = torch.tensor(
    bg_color, dtype=shaded.dtype, device=shaded.device,
).reshape(1, 3, 1, 1)                                        # (1, 3, 1, 1)
frames = (shaded + bg * (1 - alpha)).clamp(0, 1).cpu()       # (T, 3, H, W)
```

收益：

- 减少 CPU / GPU 同步。
- 避免 uint8 量化损失。
- 避免 `alpha` 单通道被无谓扩成 3 通道。

## 期间遇到并修复的问题

### 问题 1：`flash_attn` ABI / Python 环境错误

曾遇到错误：

```text
flash_attn_2_cuda.cpython-313-x86_64-linux-gnu.so: undefined symbol ...
```

根因：

- `ff-train` 子进程调用 `accelerate launch`。
- shell 的 `PATH` 先找到了 `~/.local/bin/accelerate`。
- 该 accelerate 属于另一个 Python 3.13 / miniconda 环境。
- 当前训练实际需要 `/home/zhiyuan_ma/anaconda3/envs/grpo3d_trellis2` 环境。

修复/规避方式：

```bash
export PATH="/home/zhiyuan_ma/anaconda3/envs/grpo3d_trellis2/bin:$PATH"
```

验证过正确路径：

```text
/home/zhiyuan_ma/anaconda3/envs/grpo3d_trellis2/bin/ff-train
/home/zhiyuan_ma/anaconda3/envs/grpo3d_trellis2/bin/accelerate
/home/zhiyuan_ma/anaconda3/envs/grpo3d_trellis2/bin/python
```

明天如果开新 shell 继续训练，先确认 PATH。

### 问题 2：chunked split log 太冗余

曾有类似日志：

```text
[PbrMeshRendererChunked] faces=... > ..., splitting into ... chunks
```

用户认为终端太冗余。已删除普通 logging 版本，只保留必要 `_mem_probe` 信息用于短期调试。

### 问题 3：`shaded` / `alpha` frame count mismatch

错误：

```text
RuntimeError: The size of tensor a (16) must match the size of tensor b (14) at non-singleton dimension 0
```

失败位置：

```python
frames = (shaded + bg * (1 - alpha)).clamp(0, 1).cpu()
```

含义：`shaded` 有 16 帧，但 `alpha` 只有 14 帧。

根因是 `_peel_all_chunks` 有早停：

```python
if (rast[0, ..., -1] == 0).all():
    break
```

如果 layer 0 就全空，会跳过 first-layer attrs 的记录。若某一帧所有 chunk 都这样，则 `fl_data_list = []`，`_merge_first_layer` 返回空 dict，`out_dict` 缺失 `alpha` / `normal` / `mask` / `base_color` / `metallic` / `roughness`。

修复：

```python
if layer_idx > 0 and (rast[0, ..., -1] == 0).all():
    break
```

解释：layer 0 必须完整走完，用来写 first-layer attrs；layer 1+ 如果全空，可以安全 early break。这个修复必须保留。

## 关于 normal flipping

用户担心 chunked renderer 是否会翻转 normal，从而影响 PBR 效果。

结论：chunked renderer 的 normal flipping 与 upstream `PbrMeshRenderer` 保持一致，是 two-sided shading：

```python
gb_normal = torch.where(
    torch.sum(gb_normal * (pos - rays_o), dim=-1, keepdim=True) > 0,
    -gb_normal,
    gb_normal,
)
```

chunked 只按 face index 切 `face_normal`，不会引入新的 normal 方向错误。

## 关于 tonemapping

用户担心新 renderer 颜色偏白，怀疑 tonemapping。

对照结果：

- upstream `microsoft/TRELLIS.2` 使用 `aces_tonemapping(...)` + `gamma_correction(...)`。
- 当前 chunked renderer 也使用同样组合。

结论：当前 tonemapping 与 upstream 一致。reference repo 中 `srgb_transfer` 是额外改动，不是 upstream 行为。

## 视觉对比验证

用户提供过一张 wandb 截图：左边 pair 是新 renderer 运行结果，看起来右侧 render 偏白；右边 pair 是旧 renderer 运行结果，颜色正常。

为了验证是否 chunked renderer 导致颜色变化，曾临时加入 `FF_RENDER_DEBUG_COMPARE=1` debug block。

### Debug 方案 A

触发方式：

```bash
FF_RENDER_DEBUG_COMPARE=1 ff-train examples/grpo/lora/trellis2/shape_unified_reward.yaml
```

对每个 rank 第一次进入 `render_latents` 时，只渲第一帧，并比较 4 种 setup：

1. `simp16_upstream`: `mesh.simplify(16_000_000)` + upstream `PbrMeshRenderer`
2. `simp16_chunked`: `mesh.simplify(16_000_000)` + `PbrMeshRendererChunked`
3. `orig_chunked`: 原始 mesh + `PbrMeshRendererChunked`
4. `orig_upstream`: 原始 mesh + upstream `PbrMeshRenderer`

输出：

- stdout 打印每组 `mean / max / p99 / std`
- 打印相对 `simp16_upstream` 的 `mean_diff / max_diff`
- 保存图到 `/tmp/ff_render_compare_r{rank}.png`

### 实际验证结果

命令运行成功，产生了 7 个 rank 的对比图。视觉结果是 4 张图从左到右基本完全一致，没有复现“全白”问题。

代表性数值：

```text
rank0:
  simp16_chunked vs simp16_upstream: mean_diff=0.0015
  orig_chunked   vs simp16_upstream: mean_diff=0.0015
  orig_upstream  vs simp16_upstream: mean_diff=0.0015

rank1:
  simp16_chunked vs simp16_upstream: mean_diff=0.0015
  orig_chunked   vs simp16_upstream: mean_diff=0.0015
  orig_upstream  vs simp16_upstream: mean_diff=0.0015

rank3:
  simp16_chunked vs simp16_upstream: mean_diff=0.0008
  orig_chunked   vs simp16_upstream: mean_diff=0.0008
  orig_upstream  vs simp16_upstream: mean_diff=0.0008

rank6:
  simp16_chunked vs simp16_upstream: mean_diff=0.0011
  orig_chunked   vs simp16_upstream: mean_diff=0.0011
  orig_upstream  vs simp16_upstream: mean_diff=0.0011
```

结论：

- 对测试到的 first frame，chunked renderer 与 upstream 等价。
- `mesh.simplify(16M)` 与否在该样本上也几乎无差。
- 原始 mesh + upstream 当时也没有 crash，说明该 debug sample 不是之前的 41.8M face overflow case。
- “全白”问题可能是特定 sample / 特定 view / 特定 training step / wandb visualization path 才触发。

### Debug 已撤掉

用户要求撤掉 debug 后，已删除：

- `FF_RENDER_DEBUG_COMPARE` block
- `/tmp/ff_render_compare_r*.png`
- `/tmp/ff_render_debug_run.log`
- `_debug_log` helper
- `.cursor/debug-744985.log`

当前没有保留这套 comparison debug 代码。

## 当前代码状态

重点文件：

```text
src/flow_factory/models/trellis2/pbr_mesh_renderer_chunked.py
src/flow_factory/models/trellis2/trellis2.py
```

当前应保留的修复：

```python
if layer_idx > 0 and (rast[0, ..., -1] == 0).all():
    break
```

当前 `render_latents` 预期仍使用 `render_frames_chunked(...)`。当前 `mesh.simplify(16777216)` 未加回，chunked renderer 是面数 overflow 的主要处理方式。

## 明天优先检查事项

### 1. tex GRPO 是否还 OOM

重点看这些阶段：sample、decode shape、decode texture、mesh construction、render latents、reward scoring、backward / optimizer step。

如果没有 OOM，可以认为 chunked renderer 至少解决了本轮 render overflow / memory pressure 的主要问题，下一步清理 `_mem_probe` 调试代码。

如果还有 OOM：

- 看 OOM 发生在 render 前还是 render 后。
- 如果发生在 render 前，可能是 decode texture / mesh construction 的问题。
- 如果发生在 render 中，继续看 `_mem_probe` 的 `n_faces` / `num_chunks` / memory peak。
- 如果发生在 reward 中，重点看 reward model batching / async queue / video tensor retention。

### 2. reward 指标是否稳定

需要关注 wandb：reward mean、reward std、per-component reward、group 内方差、loss 是否有异常 spike、generated video 质量是否和 reward 一致。

尤其观察：

- 换 renderer 后 reward 是否突然整体偏高或偏低。
- reward 是否与全白 / 空 render / 透明 render 强相关。
- async reward 是否可能把旧 sample 的 reward 对到新 sample 上。

### 3. 全白 / 颜色异常是否复现

如果再次出现全白，记录 run name、step、rank、sample index、prompt、condition image、rendered video artifact、对应 reward 值。

下一步 debug 不要只看第一帧。应做 all-frame comparison：

- 对所有 `num_frames` 渲染 upstream vs chunked。
- 每帧计算 mean/max diff。
- 保存 diff 最大的帧。
- 同时保存该帧的 `shaded`、`alpha`、`base_color`、`normal`、`mask`，判断是 texture、alpha、camera，还是 wandb composition 问题。

## 清理建议

如果明天 tex GRPO 跑稳定：

1. 清理 render path 中为这次排查加入的 `_mem_probe`。
2. 保留项目通用、已有的 `_mem_probe` 逻辑即可。
3. 检查 `pbr_mesh_renderer_chunked.py` 顶部 docstring 是否还写着“输出 dict-of-uint8-list”，如果有应更新，因为当前实际返回 tensor。
4. 检查注释里“无需改动下游消费侧”是否仍准确，因为下游已改为 tensor direct path。
5. 跑一次 `ReadLints`。
6. 如果用户确认，才考虑 commit。

## Qwen3.5-VL Reward Model 设计方向

用户计划做一个新的 reward model，替代或对照 UniReward。

### 总体目标

输入：condition image + rendered video。

不是将 frames 简单 concat 给 reward model，而是构造类似 wandb logging 的可视化视频：

```text
+-------------------+-------------------+
| condition image   | rendered frame t  |
+-------------------+-------------------+
```

每一帧左侧 condition image 固定，右侧 rendered frame 随时间变化。生成一个 side-by-side video 后送入 Qwen3.5-VL。

输出：scalar reward score，可选 textual reason / critique。

### 需要先探索的代码

下一 session 开始做 reward model 前，先找：

- reward base class / interface
- reward registry
- UniReward implementation
- async reward wrapper
- reward config schema
- wandb video logging / side-by-side visualization 代码
- current condition image 与 rendered video 在 sample 中的存放位置

建议搜索关键词：

```text
RewardModel
reward_model
unified_reward
UniReward
async_reward
wandb
video
condition_images
sample.video
```

### Reward interface 设计问题

需要明确：

- 新 reward class 放在哪里。
- 是否复用现有 registry。
- `__call__` 输入是 samples、videos、prompts，还是 batch dict。
- 是否必须支持 async reward。
- 是否需要 `dtype` / `device` 配置。
- 是否支持 multi-GPU / rank-local model loading。
- 是否要缓存 processor / tokenizer / model。

### Qwen3.5-VL 输入格式

需要确认 pretrained Qwen3.5-VL 的实际 API：

- 是否支持 video path。
- 是否支持 list of PIL frames。
- 是否要求 chat template。
- max frames / fps 限制是多少。
- 是否支持 batched video inference。
- 显存需求是多少。

建议先实现最小同步版本：

1. 从 sample 中取 condition image 和 rendered video。
2. 构造 side-by-side frames。
3. 送入 Qwen3.5-VL。
4. 让模型输出严格 JSON。
5. 解析 `score`。

之后再做 async / batching。

### Prompt 设计草案

```text
You are evaluating a 3D generation result.
The video shows a side-by-side comparison.
Left: the input condition image.
Right: rendered views of the generated 3D asset.

Score how well the rendered 3D asset matches the condition image in identity,
geometry, texture, color, style, and multi-view consistency.

Return strict JSON only:
{"score": <float from 0 to 10>, "reason": "<short reason>"}
```

后续可拆分维度：alignment、geometry、texture、color、artifact、multiview consistency，再按权重合成最终 score。

### Side-by-side video 构造注意事项

- condition image 与 rendered frame 需要同分辨率。
- 如果 condition image 是 RGBA，需要按训练使用的 bg color compositing 到 RGB。
- rendered video 当前是 `(T, C, H, W)` float `[0, 1]`。
- 拼接时应统一到 uint8 RGB 或 PIL。
- 左侧 condition image 应 repeat `T` 次。
- 右侧 rendered frame 使用对应 `t`。
- 最终视频帧形状大约是 `(H, 2W, 3)`。

### Score 归一化

要决定 Qwen output scale：

- 如果 Qwen 输出 0-10，内部 reward 可以除以 10 转 `[0, 1]`。
- 如果希望和 UniReward 权重接近，应看当前 GRPO reward scale。
- 最好在 config 中显式写 `score_min` / `score_max` / `normalize`。

初版 example config 可以类似：

```yaml
reward:
  reward_0:
    name: qwen35vl_video_reward
    reward_model: qwen35vl_video_reward
    weight: 1.0
    dtype: bfloat16
    device: cuda
    model_path: /path/to/qwen3.5-vl
    max_frames: 16
    fps: 4
    side_by_side: true
    score_min: 0
    score_max: 10
    normalize: true
    async_reward: false
```

先不要一开始就启用 async reward，等同步版本稳定后再接入。

## 下一 session 建议执行顺序

1. 看 tex GRPO 训练是否 OOM。
2. 如果 OOM，先定位 OOM 阶段，不要马上改 reward。
3. 如果无 OOM，看 wandb reward 曲线和 logged videos。
4. 如果 reward / video 稳定，清理剩余 `_mem_probe`。
5. 如果再次出现全白，设计 all-frame comparison debug。
6. 等 renderer 稳定后，再开始 Qwen3.5-VL reward model。

## 可给下一会话的启动 Prompt

继续 Flow-Factory Trellis2 chunked renderer session。

当前状态：

- `pbr_mesh_renderer_chunked.py` 已集成，用 4M faces chunk 绕开 `nvdiffrast subtriangle count overflow`。
- `trellis2.py:render_latents` 已改用 `render_frames_chunked`，返回 tensor 后直接做 CUDA bg compositing。
- 临时 comparison debug 已删除。
- 必须保留 `_peel_all_chunks` 中 `layer_idx > 0` 的 early-break guard，否则会再次出现 `shaded` 与 `alpha` frame 数不一致。
- 明天需要先看 tex GRPO 是否 OOM，以及 reward 指标是否稳定。
- 如果稳定，清理剩余 `_mem_probe`。
- 后续计划新增 Qwen3.5-VL reward model，输入为 condition image 与 rendered video 的 side-by-side video。

请先检查最新训练日志 / wandb 情况，再决定是否清理 debug probe 或进入 Qwen reward 设计。
```