# Plan: OPD contrastive — 白底渲染 + 命名统一

## 目标
1. 将渲染背景从黑色改为白色，使 `preprocess_image` 的 rembg 能正确去背裁切
2. 统一命名：`_pos_images`/`get_pos_images`/`pos_pil` → `_edited_images`/`get_edited_images`/`edited_pil`

## Global Constraints
- 不修改 `pipeline.preprocess_image()` 本身
- FlowEdit 服务器只接收/返回 RGB
- 当前 v13 正在线上跑，代码改动将在 v14 部署

## 关键发现（Explore 阶段）

### 上游参考
| 项目 | 渲染背景 | preprocess 处理 |
|------|---------|----------------|
| `flow_grpo_custom_v2`（Trellis 1） | 白色 `[1,1,1]` | RGB → rembg（白底效果好） |
| `flow_grpo_custom`（Trellis 2） | 白色 `(1,1,1)` | RGB → rembg（白底效果好） |
| **Flow-Factory**（当前） | **黑色 `[0,0,0]`** | RGB → rembg（**黑底误判**） |

### 数据流（改动后）
```
renderer (bg_color=[1,1,1])
  └─ sample.video = fg + white*(1-alpha)  → (T, 3, H, W) float32  （白色背景）

TargetImageBuffer.add_samples()
  ├─ raw_pil = to_pil_image(video[idx])     → RGB（白底）
  └─ edited_pil = _flowedit(raw_pil, cond)  → RGB（白底，FlowEdit prompt 也是白底）

_encode_c_tgt() → preprocess_image(img)
  └─ RGB → rembg → RGBA → crop（白底下 rembg 正常工作）
```

### 当前问题
- `raw_pil` 是黑背景 RGB → `preprocess_image` 跑 rembg → 黑底上 rembg 容易误判边界
- 改成白底后，与 FlowEdit prompt ("White background.") 一致，rembg 表现正常

## 相关代码
| 文件 | 函数/类 | 作用 |
|------|---------|------|
| `examples/opd/lora/trellis2/tex_contrastive.yaml` L44 | `render_bg_color` | 渲染背景色配置 |
| `src/.../trainers/trellis2_opd.py` L114 | `TargetImageBuffer._pos_images` | 命名统一对象 |
| `src/.../trainers/trellis2_opd.py` L150 | `get_pos_images()` | 命名统一对象 |
| `src/.../trainers/trellis2_opd.py` L464-489 | `_log_flowedit_comparison` | `pos_pil` 局部变量 |
| `src/.../models/trellis2/trellis2.py` L3186 | `render_latents` | 使用 `render_bg_color` |

## 实现步骤

- [ ] Step 1: 修改 config `render_bg_color: [1, 1, 1]`（黑 → 白）
- [ ] Step 2: 命名统一 — `_pos_images` → `_edited_images`，`get_pos_images` → `get_edited_images`，`pos_pil` → `edited_pil`
- [ ] Step 3: Commit + sync S3 + 提交 normal pod 训练（v14）
  - `git commit` 含 Step 1-2 改动 + 之前的 isinstance fix
  - `s5cmd sync` 到 S3
  - `koala submit -m normal -g 8 --s3-log --code "$S3:/data/work/run_codes" -c "..."` 提交 v14

## Code Diff

#### `examples/opd/lora/trellis2/tex_contrastive.yaml` (+1/-1)

```diff
@@ model — 渲染背景色改白
   render_mode: "shaded"
-  render_bg_color: [0, 0, 0]
+  render_bg_color: [1, 1, 1]
   envmap_path: null
```

#### `src/flow_factory/trainers/trellis2_opd.py` — Step 2: 命名统一 (rename)

```diff
@@ TargetImageBuffer — 字段和方法重命名
-        self._pos_images: List[Optional[Image.Image]] = []
+        self._edited_images: List[Optional[Image.Image]] = []

-        self._pos_images.clear()
+        self._edited_images.clear()

-                self._pos_images.append(None)
+                self._edited_images.append(None)

-                self._pos_images.append(edited_pil)
+                self._edited_images.append(edited_pil)

-                self._pos_images.append(raw_pil)
+                self._edited_images.append(raw_pil)

-    def get_pos_images(self) -> List[Optional[Image.Image]]:
-        return self._pos_images
+    def get_edited_images(self) -> List[Optional[Image.Image]]:
+        return self._edited_images

@@ prepare_feedback — 调用方重命名
-            self._encode_c_tgt(samples, self._tgt_buffer.get_pos_images(), key="image_cond_tgt")
+            self._encode_c_tgt(samples, self._tgt_buffer.get_edited_images(), key="image_cond_tgt")
             self._encode_c_tgt(samples, self._tgt_buffer.get_raw_images(), key="image_cond_neg")
         else:
-            self._encode_c_tgt(samples, self._tgt_buffer.get_pos_images(), key="image_cond_tgt")
+            self._encode_c_tgt(samples, self._tgt_buffer.get_edited_images(), key="image_cond_tgt")

@@ _log_flowedit_comparison — 局部变量重命名
-        pos_images = self._tgt_buffer.get_pos_images()
+        edited_images = self._tgt_buffer.get_edited_images()
         ...
-            pos_pil = pos_images[i] if i < len(pos_images) else None
-            if cond_pil is None or raw_pil is None or pos_pil is None:
+            edited_pil = edited_images[i] if i < len(edited_images) else None
+            if cond_pil is None or raw_pil is None or edited_pil is None:
                 continue
             ...
-            pos_pil = pos_pil.resize((w, h), Image.Resampling.LANCZOS)
+            edited_pil = edited_pil.resize((w, h), Image.Resampling.LANCZOS)
             ...
-            canvas.paste(pos_pil.convert("RGB"), (w * 2, 0))
+            canvas.paste(edited_pil.convert("RGB"), (w * 2, 0))
```

## 方案对比

| 方案 | 优点 | 缺点 |
|------|------|------|
| **A: 白底渲染（推荐）** | 最简单（1 行 config）；与上游项目一致；rembg 对白底稳定 | rembg 仍有微小误差（可忽略） |
| B: RGBA + mask（旧方案） | 精确 mask，零 rembg 误差 | 多 ~15 行代码；需要处理 RGBA→RGB 转换边界 |
| C: 保持黑底 | 零改动 | rembg 黑底误判，当前主要问题 |

**推荐方案 A**：一行 config 改动解决核心问题，与 FlowEdit prompt ("White background.") 天然一致，上游两个项目已验证此路径可靠。

## 风险评估
- **wandb 可视化**：渲染图从黑底变白底，视觉对比更清晰（正面影响）
- **FlowEdit 编辑质量**：source 和 target 都是白底，FlowEdit 不需要"换背景"，编辑质量可能更好
- **训练 loss**：neg target（raw render）也变白底 → preprocess crop 更精确 → 更好的负样本

## 部署命令（Step 3）

```bash
# 参考: scripts/launch_contrastive_train.sh (setup → vllm-venv → FlowEdit servers → ff-train)
S3=s3://arcwm-code-us-west-2/ericzyma/flow-factory
s5cmd sync --exclude '.git/*' --exclude '.venv/*' --exclude '*/__pycache__/*' \
           --exclude 'wandb/*' --exclude 'exp/*' \
           --exclude '.claude/*' --exclude '.agents/*' . "$S3/"

koala submit -m normal -g 8 --s3-log \
  --code "$S3:/data/work/run_codes" \
  -j "ericzyma-cf-whitebg-v14" \
  -c "cd /data/work/run_codes && bash scripts/launch_contrastive_train.sh"
```

## 状态
**当前阶段**: Planning — 等待确认
