"""Chunked PBR mesh renderer for Trellis2.

源自 ``flow_grpo_custom/edit4shape/renderers/pbr_peeled_trellis2.py``
中的 ``PbrMeshRenderer``，仅保留 chunk 切分 + 跨 chunk 深度合成核心逻辑，
删除 GRPO-specific 半透明 alpha 反传 / sRGB transfer 等额外功能，
与上游 ``trellis2.renderers.pbr_mesh_renderer.PbrMeshRenderer`` 输出语义对齐。

Why: nvdiffrast ``rasterize_fwd_cuda`` 在 mesh faces > ~16M (2^24) 时会触发
``RuntimeError: subtriangle count overflow``。本模块把 faces 按
``_MAX_FACES_PER_CHUNK`` 分 chunk 跑，跨 chunk 用 per-pixel 深度排序 +
front-to-back alpha 合成，等价于无上限单次渲染但绕开 nvdiffrast 内部缓冲限制。

Usage: 通过 :func:`render_frames_chunked` 替代上游
``trellis2.utils.render_utils.render_frames``。输出 stacked CUDA tensor，
供下游直接做背景合成与 reward 计算。
"""

from typing import *

import torch
import torch.nn.functional as F
import utils3d
from easydict import EasyDict as edict

from trellis2.representations.mesh import (
    Mesh,
    MeshWithVoxel,
    MeshWithPbrMaterial,
    TextureFilterMode,
    AlphaMode,
    TextureWrapMode,
)
from trellis2.renderers.pbr_mesh_renderer import (
    EnvMap,
    intrinsics_to_projection,
    screen_space_ambient_occlusion,
    aces_tonemapping,
    gamma_correction,
)


# DepthPeeler 单次最大面片数。nvdiffrast 内部限制 2^24 ≈ 16.7M，
# 这里取 4M 留足安全余量。面片数超过此值时自动分 chunk 并做 per-pixel 深度归并。
_MAX_FACES_PER_CHUNK = 4_000_000


class PbrMeshRendererChunked:
    """Chunked counterpart of :class:`trellis2.renderers.PbrMeshRenderer`.

    与上游 ``PbrMeshRenderer`` 等价，区别在内部 ``_peel_all_chunks``
    把 faces 按 ``_MAX_FACES_PER_CHUNK`` 切片后逐 chunk 跑 ``DepthPeeler``，
    跨 chunk 用 per-pixel 深度排序 + front-to-back alpha 合成。

    Note:
        与上游 ``PbrMeshRenderer(rendering_options={...})`` 不同，本类用
        显式 kwargs 注入选项，无 ``self.rendering_options`` edict 容器。

    Args:
        resolution: 输出分辨率（像素）。
        near / far: 透视投影的近 / 远裁剪面。
        ssaa: 超采样倍率（光栅化在 ``resolution * ssaa`` 上做，最后下采样）。
        peel_layers: ``dr.DepthPeeler`` 剥离层数。
        device: CUDA 设备。
    """

    def __init__(
        self,
        resolution: int = 512,
        near: float = 1.0,
        far: float = 100.0,
        ssaa: int = 2,
        peel_layers: int = 8,
        device: str = 'cuda',
    ):
        if 'dr' not in globals():
            import nvdiffrast.torch as dr

        self.resolution = resolution
        self.near = near
        self.far = far
        self.ssaa = ssaa
        self.peel_layers = peel_layers
        self.glctx = dr.RasterizeCudaContext(device=device)
        self.device = device

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def _empty_result(self, envmap: dict) -> edict:
        """mesh 为空时的默认返回"""
        resolution = self.resolution
        out_dict = edict(
            normal=torch.ones((3, resolution, resolution), dtype=torch.float32, device=self.device),
            mask=torch.zeros((resolution, resolution), dtype=torch.float32, device=self.device),
            base_color=torch.zeros((3, resolution, resolution), dtype=torch.float32, device=self.device),
            metallic=torch.zeros((resolution, resolution), dtype=torch.float32, device=self.device),
            roughness=torch.zeros((resolution, resolution), dtype=torch.float32, device=self.device),
            alpha=torch.zeros((resolution, resolution), dtype=torch.float32, device=self.device),
            clay=torch.zeros((resolution, resolution), dtype=torch.float32, device=self.device),
        )
        for k in envmap.keys():
            shaded_key = f"shaded_{k}" if k != '' else "shaded"
            out_dict[shaded_key] = torch.zeros((3, resolution, resolution), dtype=torch.float32, device=self.device)
        return out_dict

    # ------------------------------------------------------------------
    # Phase 1: 坐标变换
    # ------------------------------------------------------------------

    def _transform_vertices(self, mesh, extrinsics, intrinsics, transformation):
        """世界坐标 → clip/camera space，同时计算 rays。

        Returns:
            vertices_clip:  (1, V, 4)
            vertices_cam:   (1, V, 4)
            vertices_batch: (1, V, 3) 变换后世界坐标
            vertices_orig:  (1, V, 3) 原始世界坐标（PBR 采样用）
            rays_o:         (H, W, 3)
            rays_d:         (H, W, 3)
            perspective:    (4, 4)
            extrinsics_b:   (1, 4, 4)
        """
        resolution = self.resolution
        ssaa = self.ssaa
        near = self.near
        far = self.far
        rast_res = resolution * ssaa

        rays_o, rays_d = utils3d.torch.get_image_rays(
            extrinsics, intrinsics, rast_res, rast_res)

        perspective = intrinsics_to_projection(intrinsics, near, far)  # (4, 4)
        full_proj = (perspective @ extrinsics).unsqueeze(0)            # (1, 4, 4)
        extrinsics_b = extrinsics.unsqueeze(0)                        # (1, 4, 4)

        vertices = mesh.vertices.unsqueeze(0)                         # (1, V, 3)
        vertices_orig = vertices.clone()                              # (1, V, 3)
        vertices_homo = torch.cat([
            vertices, torch.ones_like(vertices[..., :1])
        ], dim=-1)                                                    # (1, V, 4)

        if transformation is not None:
            vertices_homo = torch.bmm(
                vertices_homo,
                transformation.unsqueeze(0).transpose(-1, -2))        # (1, V, 4)
            vertices = vertices_homo[..., :3].contiguous()            # (1, V, 3)

        vertices_cam = torch.bmm(
            vertices_homo, extrinsics_b.transpose(-1, -2))            # (1, V, 4)
        vertices_clip = torch.bmm(
            vertices_homo, full_proj.transpose(-1, -2))               # (1, V, 4)

        return (vertices_clip, vertices_cam, vertices,
                vertices_orig, rays_o, rays_d, perspective, extrinsics_b)

    # ------------------------------------------------------------------
    # Phase 2: Face normals
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_face_normals(vertices_batch, faces):
        """计算 per-face 法向量。

        Args:
            vertices_batch: (1, V, 3)
            faces: (F, 3)
        Returns:
            face_normal: (F, 3)
        """
        v0 = vertices_batch[0, faces[:, 0], :3]  # (F, 3)
        v1 = vertices_batch[0, faces[:, 1], :3]  # (F, 3)
        v2 = vertices_batch[0, faces[:, 2], :3]  # (F, 3)
        face_normal = torch.cross(v1 - v0, v2 - v0, dim=1)  # (F, 3)
        return F.normalize(face_normal, dim=1)                # (F, 3)

    # ------------------------------------------------------------------
    # Phase 3: DepthPeeler 多层渲染 + 排序合成
    # ------------------------------------------------------------------

    def _compute_pbr_attrs(self, rast, rast_db, faces_chunk, face_offset,
                           chunk_size, mesh, vertices_orig, rast_res):
        """从 chunk 光栅化结果提取 PBR 属性，处理 face 索引重映射。

        Args:
            rast:          (1, H, W, 4) 光栅化结果（face_id 相对 chunk 内 1-indexed）
            rast_db:       (1, H, W, 4) 光栅化微分
            faces_chunk:   (F_chunk, 3) chunk 内的局部 faces
            face_offset:   chunk 在全局 faces 中的起始偏移
            chunk_size:    chunk 内面片数
            mesh:          Mesh 对象
            vertices_orig: (1, V, 3) 原始世界坐标
            rast_res:      光栅化分辨率

        Returns:
            gb_basecolor: (H, W, 3)
            gb_metallic:  (H, W, 1)
            gb_roughness: (H, W, 1)
            gb_alpha:     (H, W, 1)
        """
        if 'dr' not in globals():
            import nvdiffrast.torch as dr

        if isinstance(mesh, MeshWithVoxel):
            if 'grid_sample_3d' not in globals():
                from flex_gemm.ops.grid_sample import grid_sample_3d
            mask = rast[..., -1:] > 0  # (1, H, W, 1)
            # ★ 用 faces_chunk，vertex 索引仍是全局的
            xyz = dr.interpolate(vertices_orig, rast, faces_chunk)[0]        # (1, H, W, 3)
            xyz = ((xyz - mesh.origin) / mesh.voxel_size).reshape(1, -1, 3) # (1, H*W, 3)
            img = grid_sample_3d(
                mesh.attrs,
                torch.cat([torch.zeros_like(mesh.coords[..., :1]), mesh.coords], dim=-1),
                mesh.voxel_shape,
                xyz,
                mode='trilinear'
            )  # (1, H*W, C)
            img = img.reshape(1, rast_res, rast_res, mesh.attrs.shape[-1]) * mask  # (1, H, W, C)
            gb_basecolor = img[0, ..., mesh.layout['base_color']]  # (H, W, 3)
            gb_metallic = img[0, ..., mesh.layout['metallic']]     # (H, W, 1)
            gb_roughness = img[0, ..., mesh.layout['roughness']]   # (H, W, 1)
            gb_alpha = img[0, ..., mesh.layout['alpha']]           # (H, W, 1)
            return gb_basecolor, gb_metallic, gb_roughness, gb_alpha

        elif isinstance(mesh, MeshWithPbrMaterial):
            tri_id = rast[0, :, :, -1:]  # (H, W, 1) chunk-local 1-indexed
            mask = tri_id > 0
            # ★ 全局 face 索引重映射
            global_tri_id = (tri_id.long() - 1 + face_offset).clamp(min=0)  # (H, W, 1)
            mid = mesh.material_ids[global_tri_id]

            # ★ UV coords 切片（chunk-local）
            uv_chunk = mesh.uv_coords[face_offset:face_offset + chunk_size]  # (F_chunk, 3, 2)
            uv_flat = uv_chunk.reshape(1, -1, 2)                            # (1, F_chunk*3, 2)
            uv_faces = torch.arange(
                chunk_size * 3, dtype=torch.int, device=self.device
            ).reshape(-1, 3)                                                 # (F_chunk, 3)
            texc, texd = dr.interpolate(
                uv_flat, rast, uv_faces,
                rast_db=rast_db, diff_attrs='all')
            # Fix problematic texture coordinates
            texc = torch.nan_to_num(texc, nan=0.0, posinf=1e3, neginf=-1e3)
            texc = torch.clamp(texc, min=-1e3, max=1e3)
            texd = torch.nan_to_num(texd, nan=0.0, posinf=1e3, neginf=-1e3)
            texd = torch.clamp(texd, min=-1e3, max=1e3)

            gb_basecolor = torch.zeros((rast_res, rast_res, 3), dtype=torch.float32, device=self.device)
            gb_metallic = torch.zeros((rast_res, rast_res, 1), dtype=torch.float32, device=self.device)
            gb_roughness = torch.zeros((rast_res, rast_res, 1), dtype=torch.float32, device=self.device)
            gb_alpha = torch.zeros((rast_res, rast_res, 1), dtype=torch.float32, device=self.device)
            for id, mat in enumerate(mesh.materials):
                mat_mask = (mid == id).float() * mask.float()
                mat_texc = texc * mat_mask
                mat_texd = texd * mat_mask

                if mat.base_color_texture is not None:
                    bc = dr.texture(
                        mat.base_color_texture.image.unsqueeze(0),
                        mat_texc, mat_texd,
                        filter_mode='linear-mipmap-linear' if mat.base_color_texture.filter_mode == TextureFilterMode.LINEAR else 'nearest',
                        boundary_mode='clamp' if mat.base_color_texture.wrap_mode == TextureWrapMode.CLAMP_TO_EDGE else 'wrap'
                    )[0]
                    gb_basecolor += bc * mat.base_color_factor * mat_mask
                else:
                    gb_basecolor += mat.base_color_factor * mat_mask

                if mat.metallic_texture is not None:
                    m = dr.texture(
                        mat.metallic_texture.image.unsqueeze(0),
                        mat_texc, mat_texd,
                        filter_mode='linear-mipmap-linear' if mat.metallic_texture.filter_mode == TextureFilterMode.LINEAR else 'nearest',
                        boundary_mode='clamp' if mat.metallic_texture.wrap_mode == TextureWrapMode.CLAMP_TO_EDGE else 'wrap'
                    )[0]
                    gb_metallic += m * mat.metallic_factor * mat_mask
                else:
                    gb_metallic += mat.metallic_factor * mat_mask

                if mat.roughness_texture is not None:
                    r = dr.texture(
                        mat.roughness_texture.image.unsqueeze(0),
                        mat_texc, mat_texd,
                        filter_mode='linear-mipmap-linear' if mat.roughness_texture.filter_mode == TextureFilterMode.LINEAR else 'nearest',
                        boundary_mode='clamp' if mat.roughness_texture.wrap_mode == TextureWrapMode.CLAMP_TO_EDGE else 'wrap'
                    )[0]
                    gb_roughness += r * mat.roughness_factor * mat_mask
                else:
                    gb_roughness += mat.roughness_factor * mat_mask

                if mat.alpha_mode == AlphaMode.OPAQUE:
                    gb_alpha += 1.0 * mat_mask
                else:
                    if mat.alpha_texture is not None:
                        a = dr.texture(
                            mat.alpha_texture.image.unsqueeze(0),
                            mat_texc, mat_texd,
                            filter_mode='linear-mipmap-linear' if mat.alpha_texture.filter_mode == TextureFilterMode.LINEAR else 'nearest',
                            boundary_mode='clamp' if mat.alpha_texture.wrap_mode == TextureWrapMode.CLAMP_TO_EDGE else 'wrap'
                        )[0]
                        if mat.alpha_mode == AlphaMode.MASK:
                            gb_alpha += (a * mat.alpha_factor > mat.alpha_cutoff).float() * mat_mask
                        elif mat.alpha_mode == AlphaMode.BLEND:
                            gb_alpha += a * mat.alpha_factor * mat_mask
                    else:
                        if mat.alpha_mode == AlphaMode.MASK:
                            gb_alpha += (mat.alpha_factor > mat.alpha_cutoff).float() * mat_mask
                        elif mat.alpha_mode == AlphaMode.BLEND:
                            gb_alpha += mat.alpha_factor * mat_mask

            return gb_basecolor, gb_metallic, gb_roughness, gb_alpha

    def _compute_one_layer(self, rast, rast_db, faces_chunk, face_offset,
                           chunk_size, face_normal, mesh,
                           vertices_batch, vertices_orig, vertices_cam,
                           rays_o, envmap, num_envmaps,
                           extrinsics_b, rast_res):
        """计算单个 chunk-layer 的所有属性 + shading。

        Args:
            rast:            (1, H, W, 4) 光栅化结果
            rast_db:         (1, H, W, 4) 光栅化微分
            faces_chunk:     (F_chunk, 3)
            face_offset:     chunk 偏移
            chunk_size:      chunk 面片数
            face_normal:     (F_total, 3) 全局 face normals
            mesh:            Mesh 对象
            vertices_batch:  (1, V, 3) 变换后世界坐标
            vertices_orig:   (1, V, 3) 原始世界坐标
            vertices_cam:    (1, V, 4) 相机空间坐标
            rays_o:          (H, W, 3)
            envmap:          dict of EnvMap
            num_envmaps:     int
            extrinsics_b:    (1, 4, 4)
            rast_res:        int

        Returns:
            dict with keys: gb_alpha, gb_shaded, gb_cam_normal, gb_depth,
                            out_normal, mask, base_color, metallic, roughness, alpha_attr
        """
        if 'dr' not in globals():
            import nvdiffrast.torch as dr

        # ---- Pos ----
        pos = dr.interpolate(vertices_batch, rast, faces_chunk)[0][0]  # (H, W, 3)

        # ---- Camera-space depth ----
        gb_depth = dr.interpolate(
            vertices_cam[..., 2:3].contiguous(), rast, faces_chunk
        )[0][0]                                                        # (H, W, 1)

        # ---- Face normal（chunk 切片）----
        face_normal_chunk = face_normal[face_offset:face_offset + chunk_size]  # (F_chunk, 3)
        chunk_fn_idx = torch.arange(
            chunk_size, dtype=torch.int, device=self.device
        ).unsqueeze(1).repeat(1, 3).contiguous()                       # (F_chunk, 3)
        gb_normal = dr.interpolate(
            face_normal_chunk.unsqueeze(0), rast, chunk_fn_idx
        )[0][0]                                                        # (H, W, 3)
        gb_normal = torch.where(
            torch.sum(gb_normal * (pos - rays_o), dim=-1, keepdim=True) > 0,
            -gb_normal, gb_normal)                                     # (H, W, 3)
        gb_cam_normal = (extrinsics_b[..., :3, :3].reshape(1, 1, 3, 3)
                         @ gb_normal.unsqueeze(-1)).squeeze(-1)        # (H, W, 3)

        # 首层输出用
        out_normal = -gb_cam_normal * 0.5 + 0.5                        # (H, W, 3)
        mask = (rast[0, ..., -1:] > 0).float()                         # (H, W, 1)
        out_normal = out_normal * mask + (1.0 - mask)                  # (H, W, 3)

        # ---- PBR attributes ----
        gb_basecolor, gb_metallic, gb_roughness, gb_alpha = \
            self._compute_pbr_attrs(
                rast, rast_db, faces_chunk, face_offset, chunk_size,
                mesh, vertices_orig, rast_res)

        # ---- Shading ----
        gb_basecolor_s = torch.clamp(gb_basecolor, 0.0, 1.0) ** 2.2   # (H, W, 3)
        gb_metallic_s = torch.clamp(gb_metallic, 0.0, 1.0)            # (H, W, 1)
        gb_roughness_s = torch.clamp(gb_roughness, 0.0, 1.0)          # (H, W, 1)
        gb_alpha_s = torch.clamp(gb_alpha, 0.0, 1.0)                  # (H, W, 1)
        gb_orm = torch.cat([
            torch.zeros_like(gb_metallic_s), gb_roughness_s, gb_metallic_s
        ], dim=-1)                                                     # (H, W, 3)
        gb_shaded = torch.stack([
            e.shade(
                pos.unsqueeze(0), gb_normal.unsqueeze(0),
                gb_basecolor_s.unsqueeze(0), gb_orm.unsqueeze(0),
                rays_o, specular=True,
            )[0]
            for e in envmap.values()
        ], dim=0)                                                      # (E, H, W, 3)

        return {
            'gb_alpha': gb_alpha_s,           # (H, W, 1)
            'gb_shaded': gb_shaded,           # (E, H, W, 3)
            'gb_cam_normal': gb_cam_normal,   # (H, W, 3)
            'gb_depth': gb_depth,             # (H, W, 1)
            'out_normal': out_normal,         # (H, W, 3) — 首层用
            'mask': mask,                     # (H, W, 1) — 首层用
            'base_color': gb_basecolor,       # (H, W, 3) — 首层用
            'metallic': gb_metallic,          # (H, W, 1) — 首层用
            'roughness': gb_roughness,        # (H, W, 1) — 首层用
            'alpha_attr': gb_alpha,           # (H, W, 1) — 首层用
        }

    def _peel_all_chunks(self, vertices_clip, vertices_cam, vertices_batch,
                         vertices_orig, faces, face_normal, mesh,
                         rays_o, envmap, num_envmaps,
                         extrinsics_b, rast_res, peel_layers):
        """Phase A: 将 faces 分 chunk，逐 chunk 逐层 peel，收集所有层数据。

        与 Normal peeled 路径共享同一分 chunk + 多层 peel 策略。

        Returns:
            all_depths:      List[Tensor(H, W)]       — detach，用于排序
            all_alphas:      List[Tensor(H, W, 1)]    — 材质 alpha
            all_shadeds:     List[Tensor(E, H, W, 3)] — E=num_envmaps
            all_cam_normals: List[Tensor(H, W, 3)]    — 相机空间法向量
            all_cam_depths:  List[Tensor(H, W, 1)]    — 相机空间深度
            fl_data_list:    List[dict]                — 每个 chunk 的首层属性
        """
        if 'dr' not in globals():
            import nvdiffrast.torch as dr

        num_faces = faces.shape[0]
        K = (num_faces + _MAX_FACES_PER_CHUNK - 1) // _MAX_FACES_PER_CHUNK

        all_depths: list = []       # List[Tensor(H, W)]       — 非可微，用于排序
        all_alphas: list = []       # List[Tensor(H, W, 1)]
        all_shadeds: list = []      # List[Tensor(E, H, W, 3)]
        all_cam_normals: list = []  # List[Tensor(H, W, 3)]
        all_cam_depths: list = []   # List[Tensor(H, W, 1)]
        fl_data_list: list = []     # List[dict] — 每个 chunk 的首层属性

        for chunk_idx in range(K):
            off = chunk_idx * _MAX_FACES_PER_CHUNK
            size = min(_MAX_FACES_PER_CHUNK, num_faces - off)
            faces_chunk = faces[off:off + size]  # (F_chunk, 3)

            with dr.DepthPeeler(self.glctx, vertices_clip, faces_chunk,
                                (rast_res, rast_res)) as peeler:
                for layer_idx in range(peel_layers):
                    rast, rast_db = peeler.rasterize_next_layer()  # (1, H, W, 4)

                    # 提前终止：仅 layer >= 1 且该层全空时跳出。
                    # layer 0 必须始终走完整路径，确保 fl_data 被记录、
                    # 进而 _merge_first_layer 能产出 alpha/normal/mask 等
                    # 字段——与上游 PbrMeshRenderer.render（无 break）
                    # 行为对齐，避免某帧 mesh 在视锥外时 res 字段缺失，
                    # 引发下游 stack 维度不一致。
                    if layer_idx > 0 and (rast[0, ..., -1] == 0).all():
                        break

                    # 排序用 depth（detach）
                    sort_depth = rast[0, ..., 2].detach()             # (H, W)
                    sort_depth[rast[0, ..., -1] == 0] = float('inf')  # 空像素 → inf

                    # 逐层计算
                    layer = self._compute_one_layer(
                        rast, rast_db, faces_chunk, off, size,
                        face_normal, mesh,
                        vertices_batch, vertices_orig, vertices_cam,
                        rays_o, envmap, num_envmaps,
                        extrinsics_b, rast_res)

                    all_depths.append(sort_depth)
                    all_alphas.append(layer['gb_alpha'])            # (H, W, 1)
                    all_shadeds.append(layer['gb_shaded'])          # (E, H, W, 3)
                    all_cam_normals.append(layer['gb_cam_normal'])  # (H, W, 3)
                    all_cam_depths.append(layer['gb_depth'])        # (H, W, 1)

                    # 记录各 chunk 首层属性（用于跨 chunk 归并）
                    if layer_idx == 0:
                        fl_data_list.append({
                            'sort_depth': sort_depth.clone(),
                            'normal': layer['out_normal'],
                            'mask': layer['mask'],
                            'base_color': layer['base_color'],
                            'metallic': layer['metallic'],
                            'roughness': layer['roughness'],
                            'alpha_attr': layer['alpha_attr'],
                        })

        return (all_depths, all_alphas, all_shadeds,
                all_cam_normals, all_cam_depths, fl_data_list)

    @staticmethod
    def _sort_and_composite(all_depths, all_alphas, all_shadeds,
                            all_cam_normals, all_cam_depths,
                            num_envmaps, rast_res, device):
        """Phase B: per-pixel 跨 chunk 深度排序 + front-to-back alpha composite

        与 Normal peeled 路径共享同一深度排序与 alpha 合成语义。

        Returns:
            shaded: (E, H, W, 3)
            depth:  (H, W, 1)
            normal: (H, W, 3)
            alpha:  (H, W, 1)
        """
        H = W = rast_res
        shaded = torch.zeros(num_envmaps, H, W, 3, device=device)  # (E, H, W, 3)
        depth = torch.full((H, W, 1), 1e10, device=device)         # (H, W, 1)
        normal = torch.zeros(H, W, 3, device=device)               # (H, W, 3)
        max_w = torch.zeros(H, W, 1, device=device)                # (H, W, 1)
        alpha = torch.zeros(H, W, 1, device=device)                # (H, W, 1)

        if not all_depths:
            return shaded, depth, normal, alpha

        T = len(all_depths)
        sort_idx = torch.stack(all_depths).argsort(dim=0)           # (T, H, W)

        # gather 重排（gather 对 input 可微、对 index 不可微）
        stacked_a = torch.stack(all_alphas)                         # (T, H, W, 1)
        stacked_s = torch.stack(all_shadeds)                        # (T, E, H, W, 3)
        stacked_cn = torch.stack(all_cam_normals)                   # (T, H, W, 3)
        stacked_cd = torch.stack(all_cam_depths)                    # (T, H, W, 1)

        idx_1 = sort_idx.unsqueeze(-1)                              # (T, H, W, 1)
        sorted_a = torch.gather(stacked_a, 0, idx_1)               # (T, H, W, 1)
        sorted_cn = torch.gather(
            stacked_cn, 0,
            idx_1.expand(-1, -1, -1, 3))                            # (T, H, W, 3)
        sorted_cd = torch.gather(stacked_cd, 0, idx_1)             # (T, H, W, 1)
        idx_s = sort_idx.unsqueeze(1).unsqueeze(-1).expand_as(
            stacked_s)                                               # (T, E, H, W, 3)
        sorted_s = torch.gather(stacked_s, 0, idx_s)               # (T, E, H, W, 3)

        # front-to-back compositing
        for rank in range(T):
            w = (1 - alpha) * sorted_a[rank]                        # (H, W, 1)
            depth = torch.where(w > max_w, sorted_cd[rank], depth)  # (H, W, 1)
            normal = torch.where(w > max_w, sorted_cn[rank], normal)# (H, W, 3)
            max_w = torch.maximum(max_w, w)                         # (H, W, 1)
            shaded = shaded + w * sorted_s[rank]                    # (E, H, W, 3)
            alpha = alpha + w                                       # (H, W, 1)

        return shaded, depth, normal, alpha

    @staticmethod
    def _merge_first_layer(fl_data_list, rast_res, device):
        """Phase C: 跨 chunk 首层属性归并（per-pixel 选最近 chunk）

        与 Normal peeled 路径共享同一首层归并语义。

        Returns:
            dict: normal, mask, base_color, metallic, roughness, alpha
        """
        result = {}
        if not fl_data_list:
            return result

        if len(fl_data_list) == 1:
            fl = fl_data_list[0]
            result['normal'] = fl['normal']
            result['mask'] = fl['mask']
            result['base_color'] = fl['base_color']
            result['metallic'] = fl['metallic']
            result['roughness'] = fl['roughness']
            result['alpha'] = fl['alpha_attr']
            return result

        # 多 chunk: per-pixel 选最近 chunk
        stacked_depth = torch.stack(
            [d['sort_depth'] for d in fl_data_list])              # (K, H, W)
        closest = stacked_depth.argmin(dim=0)                     # (H, W)

        for key in ['normal', 'mask', 'base_color', 'metallic', 'roughness', 'alpha_attr']:
            stacked = torch.stack(
                [d[key] for d in fl_data_list])                   # (K, H, W, C)
            C = stacked.shape[-1]
            idx = closest.unsqueeze(-1).expand(
                -1, -1, C).unsqueeze(0)                           # (1, H, W, C)
            merged = torch.gather(stacked, 0, idx).squeeze(0)    # (H, W, C)
            out_key = 'alpha' if key == 'alpha_attr' else key
            result[out_key] = merged

        return result

    @staticmethod
    def _assemble_output(shaded, depth, normal, alpha,
                         fl_attrs, envmap, rast_res):
        """Phase D: 组装 out_dict"""
        out_dict = edict()

        # 首层属性
        for k, v in fl_attrs.items():
            out_dict[k] = v

        # shaded（per envmap）
        for i, k in enumerate(envmap.keys()):
            shaded_key = f"shaded_{k}" if k != '' else "shaded"
            out_dict[shaded_key] = shaded[i]                      # (H, W, 3)

        # 内部字段（SSAO / background 用，后续清理）
        out_dict._depth = depth      # (H, W, 1)
        out_dict._normal = normal    # (H, W, 3)
        out_dict._alpha = alpha      # (H, W, 1)

        return out_dict

    # ------------------------------------------------------------------
    # Phase 4: 后处理（SSAO + Background）
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_post_effects(out_dict, perspective, rays_d, envmap, use_envmap_bg):
        """SSAO + 环境光背景"""
        # SSAO
        f_occ = screen_space_ambient_occlusion(
            out_dict._depth, out_dict._normal, perspective, intensity=1.5)
        for k in envmap.keys():
            key = f"shaded_{k}" if k != '' else "shaded"
            out_dict[key] = out_dict[key] * (1 - f_occ)
        out_dict.clay = (1 - f_occ)

        # Background
        if use_envmap_bg:
            bg = torch.stack([e.sample(rays_d) for e in envmap.values()], dim=0)
            for i, k in enumerate(envmap.keys()):
                key = f"shaded_{k}" if k != '' else "shaded"
                out_dict[key] = out_dict[key] + (1 - out_dict._alpha) * bg[i]

        # 清理内部临时字段
        del out_dict._depth, out_dict._normal, out_dict._alpha

    # ------------------------------------------------------------------
    # Phase 5: SSAA 下采样 + Tonemapping
    # ------------------------------------------------------------------

    def _downsample(self, out_dict, envmap):
        """SSAA 下采样 + tonemapping"""
        resolution = self.resolution
        ssaa = self.ssaa

        for k in list(out_dict.keys()):
            if ssaa > 1:
                out_dict[k] = F.interpolate(
                    out_dict[k].unsqueeze(0).permute(0, 3, 1, 2),
                    (resolution, resolution),
                    mode='bilinear', align_corners=False, antialias=True)
            else:
                out_dict[k] = out_dict[k].permute(2, 0, 1)
            out_dict[k] = out_dict[k].squeeze()

        # Tonemapping (与上游 PbrMeshRenderer 对齐：aces + gamma 2.2，不用 sRGB OETF)
        for k in envmap.keys():
            shaded_key = f"shaded_{k}" if k != '' else "shaded"
            out_dict[shaded_key] = aces_tonemapping(out_dict[shaded_key])
            out_dict[shaded_key] = gamma_correction(out_dict[shaded_key])

        return out_dict

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def render(
            self,
            mesh: Mesh,
            extrinsics: torch.Tensor,
            intrinsics: torch.Tensor,
            envmap: Union[EnvMap, Dict[str, EnvMap]],
            use_envmap_bg: bool = False,
            transformation: Optional[torch.Tensor] = None
        ) -> edict:
        """渲染 PBR mesh（DepthPeeler 多层渲染 + 自动分 chunk）。

        5 Phase 流水线:
          1. _transform_vertices    → 坐标变换
          2. _compute_face_normals  → face normals
          3. _depth_peel_render     → DepthPeeler 多层渲染 + 排序合成
          4. _apply_post_effects    → SSAO + Background
          5. _downsample            → SSAA 下采样 + Tonemapping

        Args:
            mesh:           Mesh 对象
            extrinsics:     (4, 4) 相机外参
            intrinsics:     (3, 3) 相机内参
            envmap:         EnvMap 或 dict of EnvMap
            use_envmap_bg:  是否用环境光作背景
            transformation: (4, 4) 可选变换矩阵

        Returns:
            edict: shaded, normal, base_color, metallic, roughness, alpha, mask, clay
        """
        if not isinstance(envmap, dict):
            envmap = {'': envmap}

        if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
            return self._empty_result(envmap)

        rast_res = self.resolution * self.ssaa
        peel_layers = self.peel_layers
        num_envmaps = len(envmap)

        # ============ Phase 1: 坐标变换 ============
        (vertices_clip, vertices_cam, vertices_batch, vertices_orig,
         rays_o, rays_d, perspective, extrinsics_b) = \
            self._transform_vertices(mesh, extrinsics, intrinsics, transformation)

        # ============ Phase 2: Face normals ============
        face_normal = self._compute_face_normals(vertices_batch, mesh.faces)

        # ============ Phase 3: DepthPeeler 多层渲染 + 排序合成 ============
        (all_depths, all_alphas, all_shadeds,
         all_cam_normals, all_cam_depths, fl_data_list) = \
            self._peel_all_chunks(
                vertices_clip, vertices_cam, vertices_batch, vertices_orig,
                mesh.faces, face_normal, mesh,
                rays_o, envmap, num_envmaps,
                extrinsics_b, rast_res, peel_layers)

        shaded, depth, normal, alpha = self._sort_and_composite(
            all_depths, all_alphas, all_shadeds,
            all_cam_normals, all_cam_depths,
            num_envmaps, rast_res, self.device)
        del all_depths, all_alphas, all_shadeds, all_cam_normals, all_cam_depths

        fl_attrs = self._merge_first_layer(fl_data_list, rast_res, self.device)
        del fl_data_list

        out_dict = self._assemble_output(
            shaded, depth, normal, alpha, fl_attrs, envmap, rast_res)

        # ============ Phase 4: 后处理（SSAO + Background） ============
        self._apply_post_effects(out_dict, perspective, rays_d, envmap, use_envmap_bg)

        # ============ Phase 5: SSAA 下采样 + Tonemapping ============
        return self._downsample(out_dict, envmap)


def render_frames_chunked(
    mesh: Mesh,
    extrinsics: List[torch.Tensor],
    intrinsics: List[torch.Tensor],
    options: Optional[Dict[str, Any]] = None,
    envmap: Optional[Union[EnvMap, Dict[str, EnvMap]]] = None,
    verbose: bool = False,
    **render_kwargs,
) -> Dict[str, torch.Tensor]:
    """Multi-frame PBR rendering with chunked rasterization.

    与上游 ``trellis2.utils.render_utils.render_frames`` 等价，但内部用
    :class:`PbrMeshRendererChunked` 绕开 nvdiffrast subtriangle overflow，
    且**直接返回 stacked cuda tensor**（而非 list-of-uint8），消除下游
    numpy ↔ tensor 的 round-trip。仅支持 PBR mesh 路径
    （``MeshWithVoxel`` / ``MeshWithPbrMaterial``）。

    Args:
        mesh: ``MeshWithVoxel`` 或 ``MeshWithPbrMaterial`` 实例。
        extrinsics: 每帧 (4, 4) 相机外参列表。
        intrinsics: 每帧 (3, 3) 相机内参列表。
        options: 透传给 :class:`PbrMeshRendererChunked` 构造器的 kwargs，
            支持 keys ``resolution``, ``near``, ``far``, ``ssaa``,
            ``peel_layers``；为 ``None`` 时全部用类默认值。
        envmap: ``EnvMap`` 或 ``{name: EnvMap}``。
        verbose: 是否打印 tqdm 进度。
        **render_kwargs: 透传给 ``PbrMeshRendererChunked.render``。

    Returns:
        ``{key: Tensor (T, C, H, W) float32 [0, 1] cuda}``。
        - 3 通道字段（``shaded`` / ``shaded_<k>`` / ``normal`` / ``base_color``）：
          ``(T, 3, H, W)``；
        - 单通道字段（``alpha`` / ``mask`` / ``metallic`` / ``roughness`` / ``clay``）：
          ``(T, 1, H, W)``（不再 broadcast 成 3 通道，节省 3× 显存）。
    """
    if not isinstance(mesh, (MeshWithVoxel, MeshWithPbrMaterial)):
        raise TypeError(
            f"render_frames_chunked only supports PBR mesh "
            f"(MeshWithVoxel | MeshWithPbrMaterial), got {type(mesh).__name__}. "
            f"For Mesh / Voxel, fall back to upstream render_utils.render_frames."
        )

    renderer = PbrMeshRendererChunked(**(options or {}))

    if envmap is not None:
        render_kwargs.setdefault('envmap', envmap)

    if verbose:
        from tqdm import tqdm
        iterator = tqdm(
            list(zip(extrinsics, intrinsics)),
            total=len(extrinsics), desc='Rendering (chunked)',
        )
    else:
        iterator = zip(extrinsics, intrinsics)

    rets: Dict[str, List[torch.Tensor]] = {}
    for extr, intr in iterator:
        res = renderer.render(mesh, extr, intr, **render_kwargs)  # edict[(C,H,W) | (H,W)]
        for k, v in res.items():
            if v.dim() == 2:
                v = v.unsqueeze(0)                                  # (H, W) → (1, H, W)
            rets.setdefault(k, []).append(v)                        # List[(C, H, W)]

    return {k: torch.stack(vs, dim=0) for k, vs in rets.items()}    # {key: (T, C, H, W)}
