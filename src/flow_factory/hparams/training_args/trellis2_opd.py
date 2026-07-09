from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Tuple, Union

from ..abc import ArgABC
from ._base import TrainingArguments, _standardize_timestep_range
from .opd import resolve_distill_step_band


@dataclass(kw_only=True)
class FlowEditConfig(ArgABC):
    """Configuration for contrastive FlowEdit distillation."""

    server_url_template: str = field(
        default="http://localhost:809{rank}",
        metadata={"help": "URL template for FlowEdit servers. {rank} is replaced per DDP rank."},
    )
    prompt: str = field(
        default="Rotate the camera. White background.",
        metadata={"help": "Text prompt sent to the FlowEdit model."},
    )
    cfg_tgt: float = field(
        default=7.5,
        metadata={"help": "Classifier-free guidance scale for target (edited) image."},
    )
    cfg_src: float = field(
        default=-7.5,
        metadata={"help": "Classifier-free guidance scale for source (raw) image."},
    )
    n_max: int = field(
        default=28,
        metadata={"help": "Maximum number of FlowEdit denoising steps to use."},
    )
    steps: int = field(
        default=28,
        metadata={"help": "Number of FlowEdit inference steps."},
    )
    timeout: float = field(
        default=120.0,
        metadata={"help": "HTTP timeout in seconds for a single FlowEdit call."},
    )


@dataclass
class Trellis2OPDTrainingArguments(TrainingArguments):
    r"""Training arguments for Trellis2 OPD self-distillation.

    Teacher = pretrained base (LoRA disabled via ``adapter.use_ref_parameters``).
    Student = base + LoRA (default context).
    Loss: ``0.5 * ||mu_S(x_j, c_ref) - mu_T(x_j, c_tgt)||^2 / denom``.
    """

    timestep_range: Union[float, Tuple[float, float]] = field(
        default=0.99,
        metadata={
            "help": (
                "Fraction band of denoising transitions to distill. "
                "A float ``f`` means ``[0, f]``; a tuple is ``[lo, hi]``. "
                "Default 0.99 skips the near-clean tail."
            )
        },
    )
    num_inner_epochs: int = field(
        default=1,
        metadata={"help": "Reuse epochs over the on-policy trajectories."},
    )
    teacher_frame_strategy: Literal["random"] = field(
        default="random",
        metadata={"help": "How to select c_tgt frame from student rollout video."},
    )
    ref_kl_beta: float = field(
        default=0.0,
        metadata={
            "help": (
                "Coefficient for reference KL regularization. "
                "Penalizes student divergence from the ref model under c_ref. "
                "0.0 disables (backward compatible)."
            )
        },
    )
    use_visibility_mask: bool = field(
        default=False,
        metadata={
            "help": "Mask OPD loss to voxels visible from c_tgt viewpoint.",
        },
    )
    visibility_mask_mode: Literal["any", "all", "soft"] = field(
        default="any",
        metadata={
            "help": (
                "Aggregation from decoded voxels (512³) to latent (16³). "
                "'any': binary OR; 'all': binary AND; 'soft': visible fraction."
            ),
        },
    )
    flowedit: Optional[FlowEditConfig] = field(
        default=None,
        metadata={
            "help": (
                "FlowEdit contrastive distillation config. "
                "None disables (standard self-distill). "
                "Pass a dict or FlowEditConfig to enable."
            ),
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.timestep_range = _standardize_timestep_range(self.timestep_range)
        if isinstance(self.flowedit, dict):
            self.flowedit = FlowEditConfig.from_dict(self.flowedit)

    def get_num_train_timesteps(self, args: Any) -> int:
        lo, hi = resolve_distill_step_band(self.num_inference_steps, self.timestep_range)
        return hi - lo
