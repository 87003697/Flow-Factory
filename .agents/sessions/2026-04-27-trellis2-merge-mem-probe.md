# Session Handoff: trellis2 chunked merge generator 重构 + 显存 NDJSON probe

## 任务目的

执行 [`refactor_merge_generator_235b4cf3.plan`](../../.cursor/plans/refactor_merge_generator_235b4cf3.plan.md) 计划：

1. 把 `chunked.py` 里 `merge()` / `_merge_tensors()` 之间通过 `on_consumed` callback 桥接 chunk 所有权释放的丑写法，重构成 generator 流式 API（功能等价、2× peak 不变、代码显著变干净）。
2. 加入按需启用的 NDJSON 显存 probe（`FF_TRELLIS2_MEM_PROBE=1`），覆盖 `render_latents` / `decode_latents` / `decode_shape` / `decode_texture` / `forward_chunked` / `_run_chunked_stages` / `merge` / `_merge_tensors` 各阶段，方便下个 session 用 debug 模式精确定位 OOM 真实爆点。

## 执行内容

### 1. `chunked.py` refactor（generator 化）

- `_merge_tensors` 签名 `(self, tensors: List[Tuple[...]], *, on_consumed: Optional[Callable])` → `(self, source: Iterable[Tuple[SparseTensor, ChunkMeta]])`，纯函数语义、无副作用回调。
- 主循环 `for i in range(len(tensors)): tensor, meta = tensors[i]; tensors[i] = None` → `for tensor, meta in source:`，循环结束自然 GC，不再需要 `tensors[i] = None` hack 释放列表槽位。
- 退化分支合并：删掉前置的 `if not tensors:`，统一在循环结束后 `if not all_feats:` 同时 cover "源空" 和 "全部跳过" 两种情况；保留 cat 后的 `if merged_coords.shape[0] == 0:` 处理 "halo 全部吃掉点"。
- 按 `.cursor/rules/no-section-divider-comments.mdc` 规则去掉 `_merge_tensors` 内部的 `─────` 分割线注释。
- OOM-fix 行为完全保留：cat 前的 `torch.cuda.empty_cache()`、grad-only canonical sort（守门 `if torch.is_grad_enabled():`）、分步排序 `coords→empty_cache→feats`。
- 新增两个 generator helper：
  - `_take_chunk_results()`：`yield` 之前清掉 `chunk._result`，下游 `del tensor` 即真正释放。
  - `_take_attached_results(name)`：`yield` 之前 `chunk._result_attached.pop(name, None)`，同上。
- `merge()` 行数从 ~28 行（含 lambda + 默认参数闭包 trick）压到 ~13 行；删掉末尾的兜底清理循环（generator 已在 yield 前清掉外部引用）。
- import 加上 `Iterable`，`Optional` / `Callable` 由别处仍在用，保留。

### 2. `_mem_probe.py`（新模块，按需启用）

- 路径：`src/flow_factory/models/trellis2/_mem_probe.py`。
- 启用门：`FF_TRELLIS2_MEM_PROBE=1`。`probe()` 第一行 `if not _enabled() or not torch.cuda.is_available(): return`，**默认零开销**（无 IO、无 CUDA 同步）。
- 输出：`<dir>/mem-rank{r}-{tag}.ndjson`，每行一个 JSON event：`{ts, stage, rank, alloc_mb, reserved_mb, max_alloc_mb, max_reserved_mb, **ctx}`。
  - 默认目录：`~/.cursor/projects/home-zhiyuan-ma-code-Flow-Factory/debug-mem/`
  - 可用 `FF_TRELLIS2_MEM_PROBE_DIR` / `FF_TRELLIS2_MEM_PROBE_TAG` 覆盖。
- `reset_peak()`：每条样本起点调用，清掉 `torch.cuda.reset_peak_memory_stats`，让每阶段的 `max_alloc_mb` 反映"本阶段内"真实峰值。

### 3. 各文件 probe 插点

| 文件 | 阶段 stage | 关键 ctx |
|------|----------|--------|
| `trellis2.py:render_latents` | `render_latents/{enter, after_simplify, before_render, after_render, exit}` | `resolution`, `num_frames`, `path` |
| `trellis2.py:render_latents` 入口 | `_mem_probe_reset_peak()` | — |
| `trellis2.py:decode_latents` | `decode_latents/{enter, after_decode_shape, after_decode_texture, exit}` | `resolution`, `has_mesh`, `n_subs`, `has_tex`, `path` |
| `trellis2.py:decode_shape` | `decode_shape/{before_forward_chunked, after_forward_chunked, before_dual_grid_to_mesh, after_dual_grid_to_mesh}` | `n_points`, `feat_dim`, `resolution`, `has_mesh` |
| `trellis2.py:decode_texture` | `decode_texture/{before_forward_chunked, after_forward_chunked}` | `n_points`, `feat_dim`, `n_guide_subs` |
| `chunked_mixin.py:forward_chunked` | `forward_chunked/{enter, level_done, exit}` | `n_points`, `feat_dim`, `n_levels`, `level_idx`, `use_checkpoint`, `path` |
| `chunked_mixin.py:_run_chunked_stages` | `_run_chunked_stages/{enter, before_merge_s1, after_merge_s1, before_merge_s2, after_merge_s2, exit}` | `n_points`, `feat_dim`, `chunk_size`, `has_upsample` |
| `chunked.py:merge` | `merge/{enter, before_attached, exit}` | `n_chunks`, `coord_scale`, `axis`, `name` |
| `chunked.py:_merge_tensors` | `_merge_tensors/{loop_start, after_loop, before_cat, after_cat, before_sort, after_sort}` | `n_chunks`, `total_valid_points`, `feat_dim`, `merged_n` |

注：`trellis2.py:rollout`（716 行）实际只是 train/eval 模式分发的 4 行 pass-through（不是 RL rollout 主循环），不插桩——`render_latents` 的 enter probe 已是每条样本一次的天然边界，`reset_peak()` 也挂在那里。

## 调试经验

- **callback → generator 重构的本质**：`on_consumed(i)` 模式把"释放外部引用"的语义伸到 `_merge_tensors` 函数内部，违反单一职责；generator helper 把这个语义内化成"yield 前先解引用"的本地约定，调用方只看 `_merge_tensors(self._take_chunk_results())` 一行就懂得所有权契约。peak 显存行为完全等价，因为关键时序点都没变：循环内 yield 前清外部引用 → 循环内 `del tensor` 真释放 → 循环结束 `empty_cache` → cat。
- **闭包 late-binding 默认参数 trick 的隐藏成本**：原 `merge()` 里 `def _drop_attached(i, _name=name, _chunks=chunks_with_attr):` 是为了避免 for 循环中闭包对 `name` / `chunks_with_attr` 的延迟绑定坑。generator 直接读循环局部变量 `chunk`，没有 list 索引间接，自然没这个问题。
- **probe 选址原则**：每个阶段 enter / exit 必须成对，关键中间点（cat / sort 前后）独立打点。同一阶段内的 `max_alloc_mb` 反映"本阶段内峰值"——前提是上层调用方在该阶段起点调过 `reset_peak()`。当前实现里 `reset_peak()` 只在 `render_latents` 起点调一次，所以单条样本内部各阶段的 `max_alloc_mb` 是累计水位，**只在样本与样本之间清零**。这是有意为之：训练里 K=14 个样本同进同出，跨样本的 peak 才是真正决定 OOM 与否的指标。
- **NDJSON vs print log**：OOM crash 时 stderr 可能丢失最后几行（buffering），但 NDJSON `with f.open('a')` 每行立即 close，磁盘上的最后一行就是 crash 前最后一个 probe 点。这就是用 NDJSON 而不是 logger 的原因。

## 参考代码

| 文件 | 关键位置 | 说明 |
|------|---------|------|
| `src/flow_factory/models/trellis2/_mem_probe.py` | 整文件 | 新增；`probe(stage, **ctx)` + `reset_peak()`，env-gated |
| `src/flow_factory/models/trellis2/chunked.py` | `merge()` L275-300 | generator 版，~13 行；调用 `_take_chunk_results` / `_take_attached_results` |
| `src/flow_factory/models/trellis2/chunked.py` | `_take_chunk_results()` / `_take_attached_results()` | 新增私有 generator helper |
| `src/flow_factory/models/trellis2/chunked.py` | `_merge_tensors()` | 签名 `Iterable[Tuple[...]]`，主循环 `for tensor, meta in source` |
| `src/flow_factory/models/trellis2/chunked.py` | `merge` / `_merge_tensors` | 9 个 `_mem_probe(...)` 调用 |
| `src/flow_factory/models/trellis2/chunked_mixin.py` | `forward_chunked` / `_run_chunked_stages` | 9 个 `_mem_probe(...)` 调用 |
| `src/flow_factory/models/trellis2/trellis2.py` | `render_latents` / `decode_latents` / `decode_shape` / `decode_texture` | 14 个 `_mem_probe(...)` 调用 + 1 个 `_mem_probe_reset_peak()` |

## 最终方案

### refactor 部分

- `_merge_tensors` 接口纯函数化：`Iterable[Tuple[SparseTensor, ChunkMeta]]` → `SparseTensor`。
- 所有权契约由 `_take_chunk_results` / `_take_attached_results` 两个 generator helper 内化：yield 前清外部引用，下游 `del tensor` 即释放。
- 退化分支统一到 `if not all_feats:`，加 `if merged_coords.shape[0] == 0:` 双重保护。
- OOM 行为等价：cat 前 `empty_cache`、grad-only sort、分步排序全部保留。

### 显存 probe 部分

- 默认关闭，零成本。启用方式：

  ```bash
  export FF_TRELLIS2_MEM_PROBE=1
  export FF_TRELLIS2_MEM_PROBE_TAG=run1   # 可选，区分多次运行
  # 默认输出目录：~/.cursor/projects/home-zhiyuan-ma-code-Flow-Factory/debug-mem/
  ```

- 启动训练（沿用现有 7 GPU 命令）：

  ```bash
  FF_TRELLIS2_MEM_PROBE=1 FF_TRELLIS2_MEM_PROBE_TAG=oom-run1 \
    PATH=/home/zhiyuan_ma/anaconda3/envs/grpo3d_trellis2/bin:$PATH \
    ff-train examples/grpo/lora/trellis2/shape_unified_reward.yaml
  ```

- NDJSON 分析速查：

  ```bash
  cd ~/.cursor/projects/home-zhiyuan-ma-code-Flow-Factory/debug-mem

  # 1) 每张卡 OOM 前最后一个成功阶段
  for f in mem-rank*-oom-run1.ndjson; do
      echo "=== $f ==="
      tail -1 "$f"
  done

  # 2) rank 0 各阶段水位时序（最后 50 条）
  jq -c '{stage,alloc_mb,reserved_mb,max_alloc_mb}' \
    mem-rank0-oom-run1.ndjson | tail -50

  # 3) 按阶段聚合峰值，找哪个阶段 alloc 跳得最高
  jq -s 'group_by(.stage)
         | map({stage: .[0].stage, peak_alloc: (map(.alloc_mb) | max)})
         | sort_by(.peak_alloc) | reverse | .[0:10]' \
    mem-rank0-oom-run1.ndjson

  # 4) 单个 stage 跨样本水位变化（看是否在 leak）
  jq -c 'select(.stage == "_merge_tensors/before_cat")
         | {ts,alloc_mb,reserved_mb,merged_n,feat_dim}' \
    mem-rank0-oom-run1.ndjson
  ```

## 下一步任务

1. **冒烟回归**：默认关闭 probe（不 export 环境变量）路径上跑一次完整 epoch 0，确认 generator refactor 没改动数值/性能。预期 reward 与上次 7 GPU 完整训练等价（前一 session 曾跑通 epoch 0）。
2. **debug 复现 OOM**：`export FF_TRELLIS2_MEM_PROBE=1` 后用同一 yaml 跑，等 OOM crash 把 NDJSON 贴回来。
3. **基于 NDJSON 决定下一步优化方向**：
   - 如果 `_merge_tensors/before_cat` 处仍是峰值：走"预分配 cat 输出 + 分块 `copy_`"方案，把 peak 从 2× 压到 1×。
   - 如果 `forward_chunked/level_done` 在某层后就跳 +30 GiB：定位到具体哪层 chunked 内部还有泄漏。
   - 如果 `render_latents/before_render` 之后才爆：renderer 本身的显存（不在本次 refactor 范围）。
   - 如果 baseline 81 GiB 占用是主因：考虑 ref 模型 offload 到 CPU（独立改动）。

## 初步方案

冒烟验证（不启用 probe）：

```bash
PATH=/home/zhiyuan_ma/anaconda3/envs/grpo3d_trellis2/bin:$PATH \
  ff-train examples/grpo/lora/trellis2/shape_unified_reward.yaml
```

期望：epoch 0 完整跑过、reward_mean / ratio_mean 与上一次成功 run 等价、无 OOM。

debug 复现：在上面命令前加 `FF_TRELLIS2_MEM_PROBE=1 FF_TRELLIS2_MEM_PROBE_TAG=oom-run1`，等 crash 取 `~/.cursor/projects/home-zhiyuan-ma-code-Flow-Factory/debug-mem/mem-rank*-oom-run1.ndjson`。

潜在风险：

- generator 的 `yield` 在某些 Python 边界条件（exception 中途）下析构时机比 list-comp 略晚——但这里没有 try/except 包围 `_merge_tensors`，正常完成路径下析构时机与 list-comp 完全一致。
- probe 文件 IO 走 `with f.open('a')` 每条 event open/close 一次，关闭状态下零开销，开启后每条 ~10us。覆盖 ~30 个 stage × K=14 样本/step × N step 大概不会成为瓶颈，但启用 profiling 时记得只在 OOM 复现 run 用。
- `dist.is_initialized()` 在单卡 dev 跑也安全：已经判过。
