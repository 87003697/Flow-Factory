"""
Sparse-aware flow matching scheduler for Trellis2 shape/tex stages.

Unlike FlowMatchEulerDiscreteSDEScheduler (in flow_factory.scheduler) which
operates on dense tensors via to_broadcast_tensor(), this scheduler works
directly on SparseTensor latents, preserving _spatial_cache and avoiding the
N_points-as-batch broadcasting error.
"""

from typing import List, Optional, Union, Literal
import math

import torch
import numpy as np

from trellis2.modules.sparse import SparseTensor
from ...scheduler.abc import SDESchedulerOutput, SDESchedulerMixin


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
            torch.tensor(sde_steps, dtype=torch.int64)
            if sde_steps is not None else None
        )
        self._num_sde_steps = num_sde_steps
        self.seed = seed
        self.dynamics_type = dynamics_type
        self.sigma_min = sigma_min
        self.rescale_t = rescale_t
        self._is_eval = False

        self._timesteps_np: np.ndarray = np.array([])  # float64
        self.timesteps: torch.Tensor = torch.tensor([])
        self.sigmas: torch.Tensor = torch.tensor([])

    # ── Time grid ────────────────────────────────────────────────────

    def set_timesteps(self, num_steps: int, device: torch.device) -> None:
        """Build float64 time sequence, aligned with official FlowEulerSampler.sample()."""
        t_seq = np.linspace(1, 0, num_steps + 1)  # float64
        t_seq = self.rescale_t * t_seq / (1 + (self.rescale_t - 1) * t_seq)
        self._timesteps_np = t_seq
        self.timesteps = torch.from_numpy(t_seq).to(device=device, dtype=torch.float32)
        self.sigmas = self.timesteps.clone()

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
        idx = torch.randperm(len(self.sde_steps), generator=generator)[:self.num_sde_steps]
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
        idx = int((self.timesteps - t).abs().argmin().item())
        return self.noise_level if idx in self.current_sde_steps.tolist() else 0.0

    def get_noise_level_for_sigma(self, sigma):
        return self.get_noise_level_for_timestep(sigma)

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
        t_val  = float(self._timesteps_np[idx])
        t_next = float(self._timesteps_np[idx + 1])
        return self._step_impl(
            velocity, t_val, t_next, latents,
            next_latents, generator, noise_level, compute_log_prob,
        )

    def step(
        self,
        velocity: SparseTensor,
        t_val: float,
        t_next_val: float,
        latents: SparseTensor,
        next_latents: Optional[SparseTensor] = None,
        generator: Optional[torch.Generator] = None,
        noise_level: Optional[float] = None,
        compute_log_prob: bool = True,
    ) -> SDESchedulerOutput:
        """Training forward() entry. t_val / t_next_val are Python floats."""
        return self._step_impl(
            velocity, t_val, t_next_val, latents,
            next_latents, generator, noise_level, compute_log_prob,
        )

    # ── Internal dispatch ────────────────────────────────────────────

    def _step_impl(
        self,
        velocity: SparseTensor,
        t_val: float,
        t_next_val: float,
        latents: SparseTensor,
        next_latents: Optional[SparseTensor],
        generator: Optional[torch.Generator],
        noise_level: Optional[float],
        compute_log_prob: bool,
    ) -> SDESchedulerOutput:
        dynamics_type = self.dynamics_type
        if self.is_eval or dynamics_type == 'ODE':
            return self._step_ode(velocity, t_val, t_next_val, latents)
        elif dynamics_type == 'Flow-SDE':
            return self._step_flow_sde(
                velocity, t_val, t_next_val, latents,
                next_latents, generator, noise_level, compute_log_prob,
            )
        else:
            raise NotImplementedError(
                f"dynamics_type='{dynamics_type}' not yet implemented for sparse latents. "
                "Use 'ODE' or 'Flow-SDE'."
            )

    # ── ODE step ─────────────────────────────────────────────────────

    def _step_ode(
        self,
        velocity: SparseTensor,
        t_val: float,
        t_next_val: float,
        latents: SparseTensor,
    ) -> SDESchedulerOutput:
        """
        ODE Euler step aligned with official flow_euler.py:
            pred_x_prev = x_t - (t - t_prev) * pred_v
        delta is Python float (float64 precision), arithmetic on SparseTensor.
        Returns SparseTensor in next_latents to preserve _spatial_cache.
        """
        delta = t_val - t_next_val  # Python float, float64 precision
        prev_sample = latents - delta * velocity  # SparseTensor, preserves _spatial_cache
        return SDESchedulerOutput.from_dict({
            'next_latents': prev_sample,      # SparseTensor — preserves _spatial_cache
            'noise_pred':   velocity.feats,
            'log_prob':     None,
        })

    # ── Flow-SDE step ────────────────────────────────────────────────

    def _step_flow_sde(
        self,
        velocity: SparseTensor,
        t_val: float,
        t_next_val: float,
        latents: SparseTensor,
        next_latents: Optional[SparseTensor],
        generator: Optional[torch.Generator],
        noise_level: Optional[float],
        compute_log_prob: bool,
    ) -> SDESchedulerOutput:
        """
        Flow-SDE step. Math matches FlowMatchEulerDiscreteSDEScheduler.step()
        Flow-SDE branch, but uses Python float for sigma/dt instead of
        to_broadcast_tensor.
        """
        sigma = t_val
        dt = t_next_val - t_val  # < 0

        if noise_level is None:
            noise_level = self.get_noise_level_for_timestep(t_val)

        sigma_max = float(self._timesteps_np[1]) if len(self._timesteps_np) > 1 else sigma
        sigma_safe = sigma_max if sigma >= 1.0 else sigma
        std_dev_t = math.sqrt(sigma / (1.0 - sigma_safe)) * noise_level

        # Mean term (scalar arithmetic on SparseTensor)
        next_latents_mean = (
            latents  * (1.0 + std_dev_t**2 / (2.0 * sigma) * dt)
            + velocity * (1.0 + std_dev_t**2 * (1.0 - sigma) / (2.0 * sigma)) * dt
        )

        _input_dtype = latents.feats.dtype

        if next_latents is None:
            noise_feats = torch.randn(
                latents.feats.shape,
                generator=generator,
                device=latents.feats.device,
                dtype=torch.float32,
            )
            noise_st = latents.replace(feats=noise_feats)
            next_latents_st = next_latents_mean + noise_st * (std_dev_t * math.sqrt(-dt))
            next_latents_st = next_latents_st.replace(
                feats=next_latents_st.feats.to(_input_dtype).float()
            )
        else:
            next_latents_st = next_latents

        log_prob = None
        if compute_log_prob:
            std_variance = std_dev_t * math.sqrt(-dt)
            diff = next_latents_st.feats.detach() - next_latents_mean.feats
            log_prob = (
                -(diff ** 2) / (2.0 * std_variance**2)
                - math.log(std_variance)
                - 0.5 * math.log(2.0 * math.pi)
            ).mean(dim=-1)  # (N,) token-level

        return SDESchedulerOutput.from_dict({
            'next_latents':      next_latents_st,       # SparseTensor — preserves _spatial_cache
            'next_latents_mean': next_latents_mean,     # SparseTensor
            'noise_pred':        velocity.feats,
            'log_prob':          log_prob,
            'std_dev_t':         torch.tensor(std_dev_t),
            'dt':                torch.tensor(dt),
        })
