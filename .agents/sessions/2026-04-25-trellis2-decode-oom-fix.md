# Session Handoff: trellis2 rollout decode/render OOM 修复（chunked decoder 真正启用）

## 任务目的

执行上一 session（`2026-04-25-trellis2-decode-oom-plan`）制定的方案，把 `Trellis2Adapter.decode_*` / `render_latents` 路径上的 OOM 修掉，并在 2 GPU 冒烟上跑通。

## 执行内容

- `load_pipeline()`：在 `decoder.convert_to_fp32()` 旁边对 `shape_decoder` / `tex_decoder` 各调一次 `ChunkedDecoderMixin.inject_to(decoder)`，删掉 `decode_shape` 里的延迟 inject；之前的延迟 inject 只挂方法不覆盖 `__call__`，**chunked 路径从未生效**
- `decode_shape()`：`decoder(slat, return_subs=True)` → `decoder.forward_chunked(slat, return_subs=True)`；`forward_chunked` 绕过了 `FlexiDualGridVaeDecoder.forward()` 的 mesh 包装，需手动复现 `(1+2*voxel_margin) * sigmoid(h.feats[..., 0:3]) - voxel_margin`、`intersected = h.feats[..., 3:6] > 0`、`quad_lerp = softplus(h.feats[..., 6:7])` 三个流水线，再调 `flexible_dual_grid_to_mesh`
- `decode_texture()`：`decoder(slat, guide_subs=subs) * 0.5 + 0.5` → `decoder.forward_chunked(...)`，tex decoder 是 plain `SparseUnetVaeDecoder`，直接替换即可
- `decode_latents()`：删 `mesh.fill_holes()`（`cumesh.get_edges()` driver-level 申请显存是当时的 OOM 触发点，且只补 `<3e-2` 的小破洞 reward 影响可忽略）；在 `decode_texture` 之后对 `subs` 每个元素调 `clear_spatial_cache() + del subs + empty_cache`
- `render_latents()`：`decode_latents` 返回后、`mesh.simplify` 之前加 `torch.cuda.empty_cache()`（防御 H-D 转移）
- 临时加了 `_debug_mem_snapshot` + `D1`~`D5` 5 处 NDJSON 探针验证，跑通后清理；driver_free 在 D2→D3 阶段回升 +2-3 GB，确认 `clear_spatial_cache + empty_cache` 真的回收了 sub 持有的 layout cache
- 把所有函数内 import 提到文件顶部：`torch.nn.functional as F`、`from .chunked_mixin`、`trellis2.{modules.sparse, representations, renderers.pbr_mesh_renderer, utils}`、`o_voxel.convert`；注意 trellis2/* 必须放在 `_setup_trellis_path()` **之后**（runtime 注入 sys.path），`chunked_mixin` 模块级也 import `trellis2.modules.sparse`，所以 `from .chunked_mixin import ChunkedDecoderMixin` 也得放后面，否则 `ModuleNotFoundError`
- 2 GPU + `max_dataset_size=16` + `render_num_frames=4` + reward=UnifiedReward APS 冒烟跑完整 epoch 0：reward_mean=0.6234、ratio_mean=1.0000、clip_frac_total=0、exit_code=0
- 7 GPU 全量训练触发 chunked decoder Stage 2 merge 处的二次 OOM（`chunked.py:393 merged_feats = old_feats[sort_idx]`，需 8.68 GiB，剩 8.49 GiB）。把 `_merge_tensors` 末尾的 canonical sort 用 `if torch.is_grad_enabled():` 守门——只服务 ckpt recompute 对齐的排序，在 `@torch.no_grad()` rollout 路径下完全跳过，省掉 (N, C) 的峰值分配
- merge 后再跑触发**第三次 OOM**，落在更早的 `chunked.py:363 merged_feats = torch.cat(all_feats)`（GPU 4，需 9.09 GiB，剩 8.02 GiB，PyTorch 持有 81.12 GiB）。根因是 `merge()` 把 `(c._result, c._meta)` 打包传给 `_merge_tensors` 时，**外层 `ChunkableSparseTensor._chunks[*]._result` 仍持有原 chunk 张量**（含 halo），cat 时三份大张量同时驻留：原 chunk(~9 GiB) + all_feats 切片(~9 GiB) + cat 输出(~9 GiB) ≈ 3× peak。`_merge_tensors` 加 `on_consumed: Callable[[int], None]` 回调，每抽完第 i 个 chunk 的有效切片就立即让调用方把 `chunks_with_result[i]._result = None` / `_result_attached.pop(name)`；切片循环末尾再 `empty_cache()`，把回调释放出来的 ~9 GiB 还回 allocator pool，给 cat 让出空间

## 调试经验

- **`inject_to()` 只挂方法不覆盖 `__call__`**：调用 `decoder(slat, ...)` 走的还是父类原版 `forward()`，必须显式 `decoder.forward_chunked(slat, ...)` 才走 chunked 路径。这是这次 session 之前 chunked 优化"看起来在用其实没用"的根因
- **`forward_chunked` 绕过 `FlexiDualGridVaeDecoder.forward()` 的 mesh 包装**：shape decoder 必须在 adapter 层手动复现 sigmoid/intersected/quad_lerp 后处理；tex decoder 是 plain VAE 不需要
- **`trellis2.*` 顶层 import 必须放在 `_setup_trellis_path()` 之后**：trellis2 是通过 `sys.path.insert(0, ...)` runtime 注入的；`chunked_mixin.py` 模块级也 `from trellis2.modules.sparse import SparseTensor`，所以 `from .chunked_mixin` 也必须延后，否则在 `from .pipeline import` 之后立即 import 就会触发 `ModuleNotFoundError: No module named 'trellis2'`
- **driver_free vs torch_reserved 必须同时打**：`clear_spatial_cache + empty_cache` 真正的效果只在 driver_free 上看得到（D3 比 D1 还高），torch_reserved 数值差不大；如果只看 torch 视角会误判为"没什么效果"
- **`grpo3d_trellis2` 才是正确环境**：`/home/zhiyuan_ma/miniconda3/bin/python` 是 3.13 的 base 环境，`flash_attn_2_cuda` 不兼容；正确路径 `/home/zhiyuan_ma/anaconda3/envs/grpo3d_trellis2/bin/ff-train`，启动时 `PATH=...grpo3d_trellis2/bin:$PATH ff-train ...`
- **chunked merge 的 canonical sort 仅服务 ckpt recompute**：`chunked.py:_merge_tensors` 末尾按 `(b, x, y, z)` 排序的 8.68 GiB 峰值开销，本意是让 forward / recompute 的输出顺序对齐，避免 grad_output 串行错位；inference 路径全程 `@torch.no_grad()` 下根本不走 ckpt，排序就是纯浪费。下游 `_align_guide_sub`/`_split`/`flexible_dual_grid_to_mesh`/`MeshWithVoxel` 都按 `(coord, feat)` 配对消费，不依赖行序——`torch.is_grad_enabled()` 守门跳过即可。冒烟时用 2 GPU 小数据集 chunk 数为 1，只走 `_split` 即返回的早退分支，所以一直没暴露这个问题
- **`merge()` 把 chunks 拍成 tuple 列表会偷偷延长 chunk 张量寿命**：`[(c._result, c._meta) for c in self._chunks if c._result is not None]` 看上去把 `_result` 抽出来传给 `_merge_tensors`，但 `_chunks` 本身仍然引用原 chunk 对象，`c._result` 也仍然指向同一份 GPU 张量。`_merge_tensors` 内部的 `tensors[i] = None` 只清掉局部 list 引用，对 `c._result` 一无所知。结果就是 cat 时原 chunk + 切片 + cat 输出三份同时在显存里。**修法**：在 `_merge_tensors` 切片循环里加 `on_consumed(i)` 回调，让 `merge()` 用 `lambda i: setattr(chunks_with_result[i], '_result', None)` 即时解除外部引用，循环结束再 `empty_cache` 把 chunk 张量真正还回 allocator pool

## 参考代码

| 文件 | 关键位置 | 说明 |
|------|---------|------|
| `src/flow_factory/models/trellis2/trellis2.py` | 顶部 import 区 L83-89 | 4 个 `trellis2.*` + `o_voxel.convert` 在 `_setup_trellis_path()` 之后；`from .chunked_mixin` 也在这之后 |
| `src/flow_factory/models/trellis2/trellis2.py` | `load_pipeline()` L450-456 | `ChunkedDecoderMixin.inject_to(decoder)` 一次性注入 |
| `src/flow_factory/models/trellis2/trellis2.py` | `decode_shape()` L2125-L2170 | `forward_chunked` + 手动 `flexible_dual_grid_to_mesh` 后处理 |
| `src/flow_factory/models/trellis2/trellis2.py` | `decode_texture()` L2180-2210 | `forward_chunked(slat, guide_subs=subs)` |
| `src/flow_factory/models/trellis2/trellis2.py` | `decode_latents()` L2235-2270 | 删 `fill_holes`；`subs.clear_spatial_cache + empty_cache` |
| `src/flow_factory/models/trellis2/trellis2.py` | `render_latents()` L2305+ | `empty_cache()` after `decode_latents` 防御 |
| `src/flow_factory/models/trellis2/chunked_mixin.py` | `forward_chunked` / `_run_chunked_stages` | 真正的 chunked 实现；每层 merge 前已有 `empty_cache` |
| `src/flow_factory/models/trellis2/chunked.py` | `merge()` L273-313 | 主 / attached tensor 都向 `_merge_tensors` 传 `on_consumed` 回调，即时清空 `chunk._result` / `chunk._result_attached[name]` |
| `src/flow_factory/models/trellis2/chunked.py` | `_merge_tensors` L315-... | 切片循环末尾 `on_consumed(i) + empty_cache`；canonical sort 加 `if torch.is_grad_enabled():` 守门 |
| `third_party/TRELLIS.2/trellis2/representations/mesh/base.py` | `fill_holes()` L35-57 | 已删除调用（cumesh.get_edges 是当时 OOM 触发点） |

## 最终方案

1. **真正启用 chunked decoder**：`load_pipeline()` 一次 inject + `decode_*` 显式调 `forward_chunked`
2. **mesh 后处理上移**：shape decoder 的 sigmoid/intersected/quad_lerp + `flexible_dual_grid_to_mesh` 从 `FlexiDualGridVaeDecoder.forward()` 上移到 `decode_shape`
3. **删 fill_holes + 清 sub cache**：`decode_latents` 在 tex decode 后立即 `clear_spatial_cache + del + empty_cache`，删 `mesh.fill_holes()`
4. **render 前防御性 empty_cache**：H-D（OOM 转移到 renderer）情况下保留余地
5. **chunked merge canonical sort 仅训练时执行**：`chunked.py:_merge_tensors` 末尾的规范排序加 `if torch.is_grad_enabled():` 守门——只在 ckpt recompute 需要对齐时才付出 (N, C) 峰值开销，rollout 全程 `@torch.no_grad()` 直接跳过
6. **`merge()` 用 `on_consumed` 即时清空 chunk 引用**：`_merge_tensors` 加可选回调，切片完一份立即让 `merge()` 把 `chunk._result = None`，循环末尾 `empty_cache` 把 chunk 张量还给 allocator；避免 cat 时原 chunk + 切片 + 输出三份同时驻留的 3× 峰值

为什么不只是补显存而是改架构：因为 chunked 路径根本没生效是定性问题，而不是定量调参；canonical sort 同理，是"训练正确性"的代码，被错误地强加到了"推理省显存"的路径上；`on_consumed` 也是同理——`ChunkableSparseTensor` 的对外 API 让 chunk 持有结果到 `merge()` 末尾，但 merge 内部其实切完一份就再也不需要原 chunk 了，回调让所有权契约更紧

## 下一步任务

完整 7 GPU 训练验证 `chunked.py` 三连击修复（canonical sort 守门 + `on_consumed` 回调 + cat 前 `empty_cache`）后能否跑过 epoch 0。

## 初步方案

- **验证修复**：在 `train_flowfactory1` screen 里重跑：

  ```
  PATH=/home/zhiyuan_ma/anaconda3/envs/grpo3d_trellis2/bin:$PATH \
    ff-train examples/grpo/lora/trellis2/shape_unified_reward.yaml
  ```

  跑过 epoch 0 即可（rollout/render 是 OOM 触发点，过 0 就证明 cat OOM 已解）
- 如果还 OOM，先试 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 缓解碎片（最近一次 reserved 2.92 GiB 还有点空间，但已不像第二次 OOM 那样夸张）
- 仍 OOM 则考虑：
  - `ref_param_device: 'cuda'` + `kl_beta: 0` 是浪费；改成 `cpu` 省 ~16 GiB（yaml 改动）
  - 把 `decode_texture` 的 `guide_subs` 用完一份就 `subs[i] = None`，省下中间层 sub 的 GPU 显存
  - 考虑 trellis2 base 4B 模型权重 fp32 → fp16/bf16（model.bf16=True 已经在 mixed_precision，但 ref model + decoder 仍是 fp32）

潜在风险：
- 极端大 mesh（比如复杂场景）下 81 GiB 基础占用即使省下 9 GiB chunk 也可能撞上限；当前只观察到 Stage 2 OOM
- 长期看 trajectory CPU offload + ref model on-demand 是更彻底的方案，但本次不动
