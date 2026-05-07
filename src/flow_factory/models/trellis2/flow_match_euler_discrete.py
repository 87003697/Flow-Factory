"""
Sparse-aware flow matching scheduler for Trellis2 shape/tex stages.

Unlike FlowMatchEulerDiscreteSDEScheduler (in flow_factory.scheduler) which
operates on dense tensors via to_broadcast_tensor(), this scheduler works
directly on SparseTensor latents, preserving _spatial_cache and avoiding the
N_points-as-batch broadcasting error.

Supports both scalar (GRPO, all samples share one timestep) and per-sample
(NFT, (B,) tensor) timesteps via ``_expand_to_points``.
"""

import math
from typing import List, Literal, Optional, Union

import numpy as np
import torch
from trellis2.modules.sparse import SparseTensor

from ...scheduler.abc import SDESchedulerMixin, SDESchedulerOutput


class SparseFlowMatchEulerSDEScheduler(SDESchedulerMixin):
    """
    Sparse-aware flow matching scheduler for Trellis2 shape/tex stages.

    Provides two step entry points:
      - step_by_index(): for inference loops (index-based, float64 precision)
      - step(): for training forward() calls (t/t_next float values)
    """

    def __init__(
        self,
        noise_level: float = 0.7,
        sde_steps: Optional[Union[int, list, torch.Tensor]] = None,
        num_sde_steps: Optional[int] = None,
        seed: int = 42,
        dynamics_type: Literal["Flow-SDE", "Dance-SDE", "CPS", "ODE"] = "Flow-SDE",
        sigma_min: float = 0.0,
        rescale_t: float = 1.0,
    ):
        self.noise_level = noise_level
        self._sde_steps = (
            torch.tensor(sde_steps, dtype=torch.int64) if sde_steps is not None else None
        )
        self._num_sde_steps = num_sde_steps
        self.seed = seed
        self.dynamics_type = dynamics_type
        self.sigma_min = sigma_min
        self.rescale_t = rescale_t
        self._is_eval = False

        self._timesteps_np: np.ndarray = np.array([])  # float64
        self._timesteps_native: torch.Tensor = torch.tensor([])  # [0, 1]
        self.timesteps: torch.Tensor = torch.tensor([])  # [0, 1000]
        self.sigmas: torch.Tensor = torch.tensor([])  # [0, 1]

    # ── Time grid ────────────────────────────────────────────────────

    def set_timesteps(self, num_steps: int, device: torch.device) -> None:
        """Build float64 time sequence, aligned with official FlowEulerSampler.sample()."""
        t_seq = np.linspace(1, 0, num_steps + 1)  # float64
        t_seq = self.rescale_t * t_seq / (1 + (self.rescale_t - 1) * t_seq)
        self._timesteps_np = t_seq
        self._timesteps_native = torch.from_numpy(t_seq).to(device=device, dtype=torch.float32)
        self.timesteps = self._timesteps_native * 1000
        self.sigmas = self._timesteps_native.clone()

    def get_timesteps_for_loop(self) -> List[int]:
        """Return index list [0, ..., num_steps-1] for inference loop."""
        return list(range(len(self._timesteps_np) - 1))

    def get_precise_t(self, idx: int) -> float:
        """Return numpy float64-precision timestep value."""
        return float(self._timesteps_np[idx])

    # ── Mode management (SDESchedulerMixin abstract methods) ─────────

    @property
    def is_eval(self) -> bool:
        return self._is_eval

    def eval(self) -> None:
        self._is_eval = True

    def train(self, mode: bool = True) -> None:
        self._is_eval = not mode

    def rollout(self, mode: bool = True) -> None:
        self.train(mode=mode)

    def set_seed(self, seed: int) -> None:
        self.seed = seed

    # ── SDE step selection (SDESchedulerMixin abstract methods) ──────

    @property
    def sde_steps(self) -> torch.Tensor:
        if self._sde_steps is not None:
            return self._sde_steps
        return torch.arange(0, max(len(self.timesteps) - 1, 0), dtype=torch.int64)

    @property
    def num_sde_steps(self) -> int:
        """
        Returns the number of training steps with SDE noise.
        """
        if self._num_sde_steps is not None:
            return self._num_sde_steps

        # Default: all train steps
        return len(self.sde_steps)

    @property
    def current_sde_steps(self) -> torch.Tensor:
        """
        Returns the current SDE step indices under the self.seed.
        Randomly select self.num_train_steps from self.train_steps.
        """
        if self.num_sde_steps >= len(self.sde_steps):
            return self.sde_steps
        generator = torch.Generator().manual_seed(self.seed)
        idx = torch.randperm(len(self.sde_steps), generator=generator)[: self.num_sde_steps]
        return self.sde_steps[idx]

    @property
    def train_timesteps(self) -> torch.Tensor:
        """
        Returns timestep **indices** that to train on.
        """
        return self.current_sde_steps

    def get_train_timesteps(self) -> torch.Tensor:
        """
        Returns timesteps [0, 1000] within the current window.
        """
        return self.timesteps[self.train_timesteps]

    def get_train_sigmas(self) -> torch.Tensor:
        """
        Returns sigmas within the current window.
        """
        return self.sigmas[self.train_timesteps]

    def get_noise_levels(self) -> torch.Tensor:
        nl = torch.zeros_like(self.timesteps)
        nl[self.current_sde_steps] = self.noise_level
        return nl

    def get_noise_level_for_timestep(self, timestep):
        t = timestep.item() if isinstance(timestep, torch.Tensor) else timestep
        idx = int((self._timesteps_native - t).abs().argmin().item())
        return self.noise_level if idx in self.current_sde_steps.tolist() else 0.0

    def get_noise_level_for_sigma(self, sigma):
        return self.get_noise_level_for_timestep(sigma)

    # ── Per-point expansion ─────────────────────────────────────────

    @staticmethod
    def _expand_to_points(
        value: Union[float, torch.Tensor],
        latents: SparseTensor,
    ) -> torch.Tensor:
        """Expand a scalar or ``(B,)`` tensor to per-point ``(N_total, 1)``.

        Args:
            value: Python float, 0-d tensor, or ``(B,)`` tensor.
            latents: SparseTensor whose ``coords[:, 0]`` gives the batch index.

        Returns:
            Float32 tensor of shape ``(N_total, 1)``.
        """
        N = latents.feats.shape[0]
        device = latents.feats.device

        if isinstance(value, (int, float)):
            return torch.full((N, 1), value, device=device, dtype=torch.float32)

        value = value.to(device=device, dtype=torch.float32)
        if value.ndim == 0:
            return value.view(1, 1).expand(N, 1)

        batch_idx = latents.coords[:, 0].long()  # (N_total,)
        return value[batch_idx].unsqueeze(-1)  # (N_total, 1)

    @staticmethod
    def _scalar_repr(value: Union[float, torch.Tensor]) -> float:
        """Return a representative Python float for interval checks."""
        if isinstance(value, (int, float)):
            return float(value)
        t = value.float()
        return float(t.mean().item()) if t.numel() > 1 else float(t.item())

    # ── Step entry points ────────────────────────────────────────────

    def step_by_index(
        self,
        velocity: SparseTensor,
        idx: int,
        latents: SparseTensor,
        next_latents: Optional[SparseTensor] = None,
        generator: Optional[torch.Generator] = None,
        noise_level: Optional[float] = None,
        compute_log_prob: bool = True,
    ) -> SDESchedulerOutput:
        """Inference loop entry. Uses _timesteps_np[idx] for float64 precision."""
        assert idx + 1 < len(self._timesteps_np), f"idx={idx} has no successor"
        t_val = float(self._timesteps_np[idx])
        t_next = float(self._timesteps_np[idx + 1])
        return self._step_impl(
            velocity,
            t_val,
            t_next,
            latents,
            next_latents,
            generator,
            noise_level,
            compute_log_prob,
        )

    def step(
        self,
        velocity: SparseTensor,
        t_val: Union[float, torch.Tensor],
        t_next_val: Union[float, torch.Tensor],
        latents: SparseTensor,
        next_latents: Optional[SparseTensor] = None,
        generator: Optional[torch.Generator] = None,
        noise_level: Optional[float] = None,
        compute_log_prob: bool = True,
    ) -> SDESchedulerOutput:
        """Training forward() entry.

        ``t_val`` / ``t_next_val`` may be Python floats (GRPO, shared
        timestep) or ``(B,)`` tensors (NFT, per-sample timestep).  Both
        forms are expanded to ``(N_total, 1)`` internally via
        ``_expand_to_points``.
        """
        return self._step_impl(
            velocity,
            t_val,
            t_next_val,
            latents,
            next_latents,
            generator,
            noise_level,
            compute_log_prob,
        )

    # ── Internal dispatch ────────────────────────────────────────────

    def _step_impl(
        self,
        velocity: SparseTensor,
        t_val: Union[float, torch.Tensor],
        t_next_val: Union[float, torch.Tensor],
        latents: SparseTensor,
        next_latents: Optional[SparseTensor],
        generator: Optional[torch.Generator],
        noise_level: Optional[float],
        compute_log_prob: bool,
    ) -> SDESchedulerOutput:
        dynamics_type = self.dynamics_type
        if self.is_eval or dynamics_type == "ODE":
            return self._step_ode(velocity, t_val, t_next_val, latents)
        elif dynamics_type == "Flow-SDE":
            return self._step_flow_sde(
                velocity,
                t_val,
                t_next_val,
                latents,
                next_latents,
                generator,
                noise_level,
                compute_log_prob,
            )
        elif dynamics_type == "CPS":
            return self._step_cps(
                velocity,
                t_val,
                t_next_val,
                latents,
                next_latents,
                generator,
                noise_level,
                compute_log_prob,
            )
        else:
            raise NotImplementedError(
                f"dynamics_type='{dynamics_type}' not yet implemented for sparse latents. "
                "Use 'ODE', 'Flow-SDE', or 'CPS'."
            )

    # ── ODE step ─────────────────────────────────────────────────────

    def _step_ode(
        self,
        velocity: SparseTensor,
        t_val: Union[float, torch.Tensor],
        t_next_val: Union[float, torch.Tensor],
        latents: SparseTensor,
    ) -> SDESchedulerOutput:
        """ODE Euler step: ``x_{t-1} = x_t - (t - t_prev) * v``.

        All arithmetic is per-point via ``_expand_to_points``.  When
        ``t_val`` is a scalar the expanded tensor has uniform values,
        producing the same result as the previous scalar path.
        """
        t_pts = self._expand_to_points(t_val, latents)  # (N, 1)
        t_next_pts = self._expand_to_points(t_next_val, latents)  # (N, 1)
        delta = t_pts - t_next_pts  # (N, 1)
        prev_feats = latents.feats - delta * velocity.feats  # (N, C)
        prev_sample = latents.replace(feats=prev_feats)

        return SDESchedulerOutput.from_dict(
            {
                "next_latents": prev_sample,
                "noise_pred": velocity.feats,
                "log_prob": None,
            }
        )

    # ── Flow-SDE step ────────────────────────────────────────────────

    def _step_flow_sde(
        self,
        velocity: SparseTensor,
        t_val: Union[float, torch.Tensor],
        t_next_val: Union[float, torch.Tensor],
        latents: SparseTensor,
        next_latents: Optional[SparseTensor],
        generator: Optional[torch.Generator],
        noise_level: Optional[float],
        compute_log_prob: bool,
    ) -> SDESchedulerOutput:
        """Flow-SDE step with per-point ``(N, 1)`` arithmetic.

        Math matches ``FlowMatchEulerDiscreteSDEScheduler.step()`` Flow-SDE
        branch.  When called with scalar ``t_val`` the expanded tensors are
        uniform, reproducing the original scalar path exactly.
        """
        sigma = self._expand_to_points(t_val, latents)  # (N, 1)
        dt = self._expand_to_points(t_next_val, latents) - sigma  # (N, 1)

        if noise_level is None:
            noise_level = self.get_noise_level_for_timestep(self._scalar_repr(t_val))

        sigma_max = (
            float(self._timesteps_np[1])
            if len(self._timesteps_np) > 1
            else self._scalar_repr(t_val)
        )
        sigma_safe = torch.where(sigma >= 1.0, sigma_max, sigma)  # (N, 1)
        std_dev_t = torch.sqrt(sigma / (1.0 - sigma_safe)) * noise_level  # (N, 1)

        x_feats = latents.feats  # (N, C)
        v_feats = velocity.feats  # (N, C)

        mean_feats = (
            x_feats * (1.0 + std_dev_t**2 / (2.0 * sigma) * dt)
            + v_feats * (1.0 + std_dev_t**2 * (1.0 - sigma) / (2.0 * sigma)) * dt
        )  # (N, C)
        next_latents_mean = latents.replace(feats=mean_feats)

        _input_dtype = x_feats.dtype

        if next_latents is None:
            noise_feats = torch.randn(
                x_feats.shape,
                generator=generator,
                device=x_feats.device,
                dtype=torch.float32,
            )  # (N, C)
            nl_feats = mean_feats + noise_feats * (std_dev_t * torch.sqrt(-dt))  # (N, C)
            next_latents_st = latents.replace(feats=nl_feats.to(_input_dtype).float())
        else:
            next_latents_st = next_latents

        log_prob = None
        if compute_log_prob:
            std_variance = std_dev_t * torch.sqrt(-dt)  # (N, 1)
            diff = next_latents_st.feats.detach() - mean_feats  # (N, C)
            log_prob = (
                -(diff**2) / (2.0 * std_variance**2)
                - torch.log(std_variance)
                - 0.5 * math.log(2.0 * math.pi)
            ).mean(
                dim=-1
            )  # (N,)

        std_dev_scalar = float(std_dev_t.mean().item())
        dt_scalar = float(dt.mean().item())

        return SDESchedulerOutput.from_dict(
            {
                "next_latents": next_latents_st,
                "next_latents_mean": next_latents_mean,
                "noise_pred": velocity.feats,
                "log_prob": log_prob,
                "std_dev_t": torch.tensor(std_dev_scalar),
                "dt": torch.tensor(dt_scalar),
            }
        )

    # ── CPS step ──────────────────────────────────────────────────────

    def _step_cps(
        self,
        velocity: SparseTensor,
        t_val: Union[float, torch.Tensor],
        t_next_val: Union[float, torch.Tensor],
        latents: SparseTensor,
        next_latents: Optional[SparseTensor],
        generator: Optional[torch.Generator],
        noise_level: Optional[float],
        compute_log_prob: bool,
    ) -> SDESchedulerOutput:
        """CPS (Consistency Policy Sampling) step with per-point arithmetic.

        Math matches ``FlowMatchEulerDiscreteSDEScheduler.step()`` CPS branch.
        """
        sigma = self._expand_to_points(t_val, latents)          # (N, 1)
        sigma_prev = self._expand_to_points(t_next_val, latents)  # (N, 1)

        if noise_level is None:
            noise_level = self.get_noise_level_for_timestep(self._scalar_repr(t_val))

        std_dev_t = sigma_prev * math.sin(noise_level * math.pi / 2)  # (N, 1)

        x_feats = latents.feats    # (N, C)
        v_feats = velocity.feats   # (N, C)

        x0_feats = x_feats - sigma * v_feats                    # (N, C)
        x1_feats = x_feats + v_feats * (1.0 - sigma)            # (N, C)
        mean_feats = (
            x0_feats * (1.0 - sigma_prev)
            + x1_feats * torch.sqrt(sigma_prev ** 2 - std_dev_t ** 2)
        )  # (N, C)

        _input_dtype = x_feats.dtype

        if next_latents is None:
            noise_feats = torch.randn(
                x_feats.shape,
                generator=generator,
                device=x_feats.device,
                dtype=torch.float32,
            )  # (N, C)
            nl_feats = mean_feats + std_dev_t * noise_feats      # (N, C)
            next_latents_st = latents.replace(feats=nl_feats.to(_input_dtype).float())
        else:
            next_latents_st = next_latents

        log_prob = None
        if compute_log_prob:
            diff = next_latents_st.feats.detach() - mean_feats   # (N, C)
            log_prob = -(diff ** 2).mean(dim=-1)                  # (N,)

        return SDESchedulerOutput.from_dict(
            {
                "next_latents": next_latents_st,
                "next_latents_mean": latents.replace(feats=mean_feats),
                "noise_pred": velocity.feats,
                "log_prob": log_prob,
                "std_dev_t": torch.tensor(float(std_dev_t.mean().item())),
                "dt": torch.tensor(float((sigma_prev - sigma).mean().item())),
            }
        )
