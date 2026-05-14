# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# src/flow_factory/trainers/trellis2_mixin.py
"""
Trellis2 trainer mixin: cross-GPU upstream stage sharing and multi-stage
rollout logic shared between Trellis2GRPOTrainer and Trellis2NFTTrainer.

This mixin assumes the host class has `self.adapter`, `self.accelerator`,
`self.training_args`, `self.model_args`, and `self.config` (all provided
by ``BaseTrainer``).
"""
import gc
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.distributed as dist

from ..models.trellis2 import Trellis2Sample
from ..samples import BaseSample
from ..utils.base import filter_kwargs
from ..utils.logger_utils import setup_logger

logger = setup_logger(__name__)


# OOM recovery design:
# Distributed sampling can hang if only one rank catches an OOM while other
# ranks are already waiting in NCCL collectives (dist.broadcast or
# accelerator.gather). Each OOM-prone section must clear CUDA memory, use a
# collective check so all ranks agree to skip, and raise _WindowOOMSkipped for
# the concrete sample() implementation to catch.
# Layer 1: _run_owned_pilots handles owner-rank upstream inference.
# Layer 2: _broadcast_tensor handles receiver-rank broadcast-buffer allocation.
# Layer 3: _rollout_group handles training-stage adapter.inference.


class _WindowOOMSkipped(RuntimeError):
    """Raised when a rollout window must be skipped due to OOM on any rank."""


class Trellis2TrainerMixin:
    """Shared Trellis2 multi-stage rollout and upstream stage sharing.

    Provides:
      * ``_init_trellis2`` — stage detection, render kwargs, DDP static graph
      * Cross-GPU upstream broadcast protocol
      * ``_rollout_group`` — upstream sharing → ``adapter.inference``
      * Static helpers (``_infer_batch_size``, ``_merge_batches``)
    """

    @staticmethod
    def _recover_oom() -> None:
        """Release CUDA memory after an OOM exception."""
        gc.collect()
        torch.cuda.empty_cache()

    def _dist_any_error(self, local_error: bool) -> bool:
        """Return True if *any* rank reports an error (all_reduce MAX)."""
        flag = torch.tensor(
            [int(local_error)],
            dtype=torch.int32,
            device=self.accelerator.device,
        )  # (1,)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return flag.item() > 0

    def _init_trellis2(self) -> None:
        """Initialise Trellis2-specific state.  Call from ``__init__``."""
        train_stage = self.adapter.pipeline._target_flow_model.split("_")[0]
        self._training_stage = train_stage
        if train_stage == "shape":
            self._upstream_stages = ["dense"]
            self._inference_stages = ["shape", "tex"]
        elif train_stage == "tex":
            self._upstream_stages = ["dense", "shape"]
            self._inference_stages = ["tex"]
        else:
            self._upstream_stages = []
            self._inference_stages = ["dense"]

        K = self.training_args.group_size
        bs = self.training_args.per_device_batch_size
        if K % bs != 0:
            raise ValueError(
                f"group_size ({K}) must be divisible by "
                f"per_device_batch_size ({bs}) for batch accumulation."
            )
        if self.config.data_args.sampler_type == "group_contiguous":
            self._batches_to_merge = K // bs
        else:
            self._batches_to_merge = 1

        extra = self.model_args.extra_kwargs
        self._render_kwargs = {
            "decode_output": extra.get("decode_output", False),
            "render_num_frames": extra.get("render_num_frames", 24),
            "render_resolution": extra.get("render_resolution", 512),
            "render_mode": extra.get("render_mode", "shaded"),
            "render_bg_color": tuple(extra.get("render_bg_color", (0, 0, 0))),
            "envmap_path": extra.get("envmap_path", None),
        }

        # Trellis2's sparse transformer uses gradient checkpointing with
        # use_reentrant=False, which re-enters autograd during backward and
        # can fire DDP's AccumulateGrad hooks twice for the same parameter,
        # causing "Expected to mark a variable ready only once".
        # _set_static_graph() switches DDP to a graph-order reduction strategy
        # that tolerates multiple hook invocations.
        for name in self.adapter.target_module_map:
            module = getattr(self.adapter, name, None)
            if module is not None and hasattr(module, "_set_static_graph"):
                module._set_static_graph()
                logger.info(f"Called _set_static_graph() on DDP-wrapped '{name}'")

        # FlowEdit config
        fe_enabled = extra.get("flowedit_enabled", False)
        if fe_enabled:
            self._flowedit_cfg = {
                "fe_steps": extra.get("flowedit_steps", 20),
                "cfg_scale_src": extra.get("flowedit_cfg_scale_src", -3.0),
                "cfg_scale_tgt": extra.get("flowedit_cfg_scale_tgt", 5.0),
                "noise_update": extra.get("flowedit_noise_update", True),
                "flowedit_use_ref": extra.get("flowedit_use_ref", True),
            }
            K = self.training_args.group_size
            self.advantage_processor.group_size = K * 2
            self.reward_buffer.group_size = K * 2
            logger.info(
                "FlowEdit enabled: reward and advantage group_size doubled %d -> %d",
                K,
                K * 2,
            )
        else:
            self._flowedit_cfg = None

    # Rollout helpers

    def _rollout_group(
        self,
        merged_batch: Dict[str, Any],
        trajectory_indices,
        compute_log_prob: bool = True,
    ) -> List[BaseSample]:
        """Run upstream stages (cross-GPU shared) then training stage.

        Raises ``_WindowOOMSkipped`` if any rank hits OOM.

        Args:
            merged_batch: Merged dataloader batch dict.
            trajectory_indices: Trajectory index spec for inference.
            compute_log_prob: Whether to compute log-probabilities (GRPO
                requires True, NFT uses False).
        """
        pre_samples = self._distributed_upstream_stages(merged_batch, trajectory_indices)

        sample_kwargs: Dict[str, Any] = {
            **self.training_args,
            "compute_log_prob": compute_log_prob,
            "trajectory_indices": trajectory_indices,
            "stages": self._inference_stages,
            "training_stage": self._training_stage,
            **self._render_kwargs,
            **merged_batch,
        }
        if pre_samples is not None:
            sample_kwargs["samples"] = pre_samples

        sample_kwargs = filter_kwargs(self.adapter.inference, **sample_kwargs)

        # Layer 3: adapter.inference has no NCCL collectives; OOM only affects
        # the current rank. Later accelerator.gather calls require every rank to
        # participate, so all ranks must agree to skip before continuing.
        local_error = False
        result = None
        try:
            result = self.adapter.inference(**sample_kwargs)
        except torch.cuda.OutOfMemoryError:
            self._recover_oom()
            logger.warning("CUDA OOM during adapter.inference, will skip window")
            local_error = True

        if self._dist_any_error(local_error):
            del result  # Release any partially computed samples before raising.
            raise _WindowOOMSkipped("OOM during inference stage")
        return result

    def _flowedit_group(
        self,
        group: List[BaseSample],
        fe_cfg: Dict[str, Any],
    ) -> List[BaseSample]:
        """FlowEdit a group of already-completed samples.  Peer of ``_rollout_group``.

        Raises ``_WindowOOMSkipped`` if any rank hits OOM.
        """
        fe_kwargs: Dict[str, Any] = {
            "stage": self._training_stage,
            "resolution": self._resolve_rollout_resolution(),
            "num_inference_steps": self.training_args.num_inference_steps,
            **self._render_kwargs,
            **fe_cfg,
        }
        fe_kwargs = filter_kwargs(self.adapter.flowedit_inference, **fe_kwargs)

        local_error = False
        result = None
        try:
            result = self.adapter.flowedit_inference(samples=group, **fe_kwargs)
        except torch.cuda.OutOfMemoryError:
            self._recover_oom()
            logger.warning("CUDA OOM during FlowEdit, will skip group")
            local_error = True

        if self._dist_any_error(local_error):
            del result
            raise _WindowOOMSkipped("OOM during FlowEdit")
        return result

    # Upstream stage sharing

    def _distributed_upstream_stages(
        self,
        merged_batch: Dict[str, Any],
        trajectory_indices,
    ) -> Optional[List[Trellis2Sample]]:
        """Run upstream stages with cross-GPU owner-broadcast protocol.

        Returns pre-filled samples, or ``None`` when there are no upstream
        stages (i.e. the training target is the dense stage itself).
        Raises ``_WindowOOMSkipped`` if any rank fails.
        """
        if not self._upstream_stages:
            return None

        resolution = self._resolve_rollout_resolution()
        samples = self._create_sample_stubs(merged_batch, resolution)
        uid_to_owner = self._elect_uid_owners(samples)

        pilot_results, failed_uids = self._run_owned_pilots(
            samples,
            merged_batch,
            resolution,
            trajectory_indices,
            uid_to_owner,
        )

        # Even failed uids must run the status broadcast; otherwise peer ranks
        # block forever in dist.broadcast(status).
        any_failed = False
        for uid in sorted(uid_to_owner):
            ok = self._broadcast_upstream_for_uid(
                uid,
                uid_to_owner[uid],
                pilot_results,
                samples,
                failed_uids,
            )
            if not ok:
                any_failed = True

        if any_failed:
            raise _WindowOOMSkipped("OOM during upstream pilot stage")
        return samples

    def _resolve_rollout_resolution(self) -> int:
        """Resolve scalar Trellis2 resolution from training args."""
        return self.adapter._as_scalar_resolution(self.training_args.resolution)

    def _create_sample_stubs(
        self,
        merged_batch: Dict[str, Any],
        resolution: int,
    ) -> List[Trellis2Sample]:
        """Create local Trellis2Sample stubs from a merged dataloader batch."""
        batch_size = self._infer_batch_size(merged_batch)
        cond_images = merged_batch.get("condition_images")
        prompts = merged_batch.get("prompt")
        return [
            Trellis2Sample(
                condition_images=(cond_images[b] if cond_images is not None else None),
                resolution=resolution,
                prompt=(prompts[b] if prompts is not None else None),
            )
            for b in range(batch_size)
        ]

    def _elect_uid_owners(
        self,
        samples: List[Trellis2Sample],
    ) -> Dict[int, int]:
        """All-gather uids and elect the owner rank for each (round-robin)."""
        device = self.accelerator.device
        local_uids = torch.tensor(
            [s.unique_id for s in samples],
            dtype=torch.int64,
            device=device,
        )  # (local_B,)
        all_uids = self.accelerator.gather(local_uids)  # (W * local_B,)
        local_B = len(samples)

        uid_to_owner: Dict[int, int] = {}
        for i, uid in enumerate(sorted(torch.unique(all_uids).tolist())):
            mask = (all_uids == uid).nonzero(as_tuple=True)[0]
            candidates = (mask // local_B).unique().tolist()
            uid_to_owner[uid] = candidates[i % len(candidates)]
        return uid_to_owner

    def _run_owned_pilots(
        self,
        samples: List[Trellis2Sample],
        merged_batch: Dict[str, Any],
        resolution: int,
        trajectory_indices,
        uid_to_owner: Dict[int, int],
    ) -> Tuple[Dict[int, Trellis2Sample], Set[int]]:
        """Run upstream stages for uids owned by this rank.

        Returns ``(uid_to_pilot, failed_uids)``.
        Breaks early after the first OOM.
        """
        my_rank = self.accelerator.process_index
        my_owned_uids = [uid for uid, owner in uid_to_owner.items() if owner == my_rank]
        pilot_results: Dict[int, Trellis2Sample] = {}
        failed_uids: Set[int] = set()
        # Layer 1: after the first OOM, mark the remaining owned uids as failed
        # instead of retrying under the same memory pressure.
        oom_hit = False
        for uid in my_owned_uids:
            if oom_hit:
                failed_uids.add(uid)
                continue
            try:
                local_idx = next(i for i, s in enumerate(samples) if s.unique_id == uid)
                pilot = [
                    Trellis2Sample(
                        condition_images=samples[local_idx].condition_images,
                        resolution=resolution,
                    )
                ]
                for stage in self._upstream_stages:
                    cond_key = "image_cond_512" if stage == "dense" else "image_cond_1024"
                    neg_key = "neg_image_cond_512" if stage == "dense" else "neg_image_cond_1024"
                    self.adapter._run_stage_inference(
                        stage,
                        pilot,
                        image_cond=[merged_batch[cond_key][local_idx]],
                        neg_image_cond=[merged_batch[neg_key][local_idx]],
                        resolution=resolution,
                        num_inference_steps=self.training_args.num_inference_steps,
                        generator=None,
                        trajectory_indices=trajectory_indices,
                        extra_call_back_kwargs=[],
                        ss_resolution=32 if resolution <= 512 else 64,
                        is_training_stage=False,
                        compute_log_prob=False,
                    )
                pilot_results[uid] = pilot[0]
            except torch.cuda.OutOfMemoryError:
                self._recover_oom()
                logger.warning(
                    "CUDA OOM in upstream pilot for uid %d, skipping remaining uids",
                    uid,
                )
                oom_hit = True
                failed_uids.add(uid)
        return pilot_results, failed_uids

    # ── Broadcast helpers ────────────────────────────────────────────

    def _broadcast_tensor(
        self,
        tensor: Optional[torch.Tensor],
        src: int,
        dtype: Optional[torch.dtype] = None,
    ) -> Optional[torch.Tensor]:
        """Broadcast a tensor with runtime-unknown shape from *src* to all ranks.

        Returns ``None`` if any rank fails to allocate the receive buffer (OOM).
        """
        device = self.accelerator.device
        my_rank = self.accelerator.process_index

        ndim_t = torch.zeros(1, dtype=torch.long, device=device)  # (1,)
        if my_rank == src:
            t = (
                tensor.to(device=device, dtype=dtype).contiguous()
                if dtype
                else tensor.to(device=device).contiguous()
            )
            ndim_t[0] = t.ndim
        dist.broadcast(ndim_t, src=src)

        shape_t = torch.zeros(ndim_t.item(), dtype=torch.long, device=device)  # (ndim,)
        if my_rank == src:
            shape_t[:] = torch.tensor(t.shape, dtype=torch.long, device=device)  # (ndim,)
        dist.broadcast(shape_t, src=src)

        # Layer 2: receiver-side buffer allocation can OOM. The allocation check
        # must happen before the data broadcast, because a rank that failed to
        # allocate cannot safely participate in dist.broadcast(t).
        alloc_ok = torch.ones(1, dtype=torch.int32, device=device)  # (1,)
        if my_rank != src:
            try:
                t = torch.zeros(
                    *shape_t.tolist(),
                    dtype=dtype or torch.float32,
                    device=device,
                )  # (*shape_t,)
            except torch.cuda.OutOfMemoryError:
                self._recover_oom()
                logger.warning(
                    "CUDA OOM allocating broadcast buffer %s",
                    shape_t.tolist(),
                )
                alloc_ok[0] = 0
                t = torch.empty(0, device=device)  # (0,), placeholder only

        dist.all_reduce(alloc_ok, op=dist.ReduceOp.MIN)
        if alloc_ok.item() == 0:
            return None

        dist.broadcast(t, src=src)
        return t

    def _broadcast_upstream_for_uid(
        self,
        uid: int,
        owner: int,
        pilot_results: Dict[int, Trellis2Sample],
        local_samples: List[Trellis2Sample],
        failed_uids: Set[int],
    ) -> bool:
        """Broadcast upstream stage outputs from *owner* to all ranks.

        Returns ``False`` if this uid must be skipped (owner OOM or receiver
        allocation failure).
        """
        device = self.accelerator.device
        my_rank = self.accelerator.process_index
        is_owner = my_rank == owner

        # Broadcast the owner's failure status before the data tensors so
        # receiver ranks can skip buffer allocation for failed uids.
        status = torch.zeros(1, dtype=torch.int32, device=device)  # (1,)
        if is_owner and uid in failed_uids:
            status[0] = 1
        dist.broadcast(status, src=owner)
        if status.item() == 1:
            return False

        pilot = pilot_results.get(uid)

        coords = self._broadcast_tensor(
            pilot.sparse_coords if is_owner else None,
            owner,
            dtype=torch.int32,
        )
        if coords is None:
            return False
        dense_latent = self._broadcast_tensor(
            pilot.dense_final_latent if is_owner else None,
            owner,
        )
        if dense_latent is None:
            return False

        shape_latent = None
        if "shape" in self._upstream_stages:
            shape_latent = self._broadcast_tensor(
                pilot.shape_final_latent if is_owner else None,
                owner,
            )
            if shape_latent is None:
                return False

        for s in local_samples:
            if s.unique_id != uid:
                continue
            s.sparse_coords = coords.clone()
            s.sparse_coords[:, 0] = 0
            s.dense_final_latent = dense_latent.clone()
            if shape_latent is not None:
                s.shape_final_latent = shape_latent.clone()
        return True

    # ── Feedback guard ─────────────────────────────────────────────

    def prepare_feedback(self, samples: List[BaseSample]) -> None:
        """Guard against empty samples from OOM-skipped windows.

        In the extreme case where every rollout window is skipped, ``samples``
        is empty and the base reward_buffer.finalize() path would fail.
        optimize() is naturally a no-op for empty samples (num_batches=0), while
        EMA and epoch advancement still happen unconditionally in start().
        """
        if not samples:
            logger.warning(
                "Epoch %d: all sample windows skipped (OOM), "
                "skipping feedback (optimize will be a no-op)",
                self.epoch,
            )
            return
        super().prepare_feedback(samples)

    # ── Eval / static helpers ────────────────────────────────────────

    def _extra_eval_inference_kwargs(self) -> dict:
        """Extra kwargs for evaluation inference (full 3-stage pipeline)."""
        return {**self._render_kwargs, "stages": ["dense", "shape", "tex"]}

    @staticmethod
    def _infer_batch_size(batch: dict) -> int:
        """Infer local batch size from a dataloader batch dict."""
        for key in ("prompt", "image_cond_512", "image_cond_1024", "condition_images"):
            val = batch.get(key)
            if val is not None:
                return len(val)
        raise ValueError(
            "Cannot infer batch size: none of prompt, image_cond_512, "
            "image_cond_1024, condition_images found in batch."
        )

    @staticmethod
    def _merge_batches(batches: list) -> dict:
        """Merge consecutive dataloader batches into one.

        List-typed values are concatenated; tensor values are cat-ed along
        dim 0; scalar/other values are kept from the first batch.
        """
        merged = {}
        for key in batches[0]:
            values = [b[key] for b in batches]
            if isinstance(values[0], list):
                merged[key] = [item for sublist in values for item in sublist]
            elif isinstance(values[0], torch.Tensor):
                merged[key] = torch.cat(values, dim=0)
            else:
                merged[key] = values[0]
        return merged
