r"""RollFlow: Flow Matching + Central-Difference Lagrangian Self-Distillation.

Core training algorithm
-----------------------
Displacement-loss pseudocode for a batch containing active LSD tokens.
Context/state and padding cleanup are omitted here for readability; the actual
implementation packs their batch dimensions consistently with the actions.

    s, t = sample_training_times()
    x_s = (1-s) * x0 + s * x1
    x_t = (1-t) * x0 + t * x1
    v_gt = x1 - x0

    # Only valid, non-padding tokens with s < t-delta and t+delta <= 1 use LSD.
    active = valid_lsd_tokens(s, t, delta, pad)
    rows = active.flatten(1).any(dim=1)
    x_s_lsd, s_lsd, t_lsd = x_s[rows], s[rows], t[rows]
    active_lsd = active[rows]
    pad_lsd = None if pad is None else pad[rows]
    B, B_lsd = x_s.shape[0], x_s_lsd.shape[0]
    t_minus = where(active_lsd, t_lsd - delta, t_lsd)
    t_plus  = where(active_lsd, t_lsd + delta, t_lsd)

    # Teacher: predict the middle position, then its instantaneous velocity.
    with no_grad():
        V_st = model(x_s_lsd, s_lsd, t_lsd)
        X_t_hat = x_s_lsd + (t_lsd - s_lsd) * V_st
        v_teacher = model(X_t_hat, t_lsd, t_lsd)

    # One student forward: two LSD endpoint predictions + full-batch FM.
    V_minus, V_plus, v_local = model(
        cat((x_s_lsd, x_s_lsd, x_t), dim=0),
        cat((s_lsd, s_lsd, t), dim=0),
        cat((t_minus, t_plus, t), dim=0),
    ).split([B_lsd, B_lsd, B], dim=0)
    X_minus = x_s_lsd + (t_minus - s_lsd) * V_minus
    X_plus  = x_s_lsd + (t_plus  - s_lsd) * V_plus
    delta_x_student = X_plus - X_minus
    delta_x_teacher = 2 * delta * v_teacher  # Already detached above.

    loss_fm  = masked_mse(v_local - v_gt, pad=pad)
    loss_lsd = masked_mse(delta_x_student - delta_x_teacher,
                          valid=active_lsd, pad=pad_lsd)
    loss = w_fm * loss_fm + w_lsd * loss_lsd

The teacher displacement is a local approximation over the interval 2*delta.
The implementation cancels the shared x_s before subtracting, using FP32
displacements instead of adding and subtracting positions. If w_lsd=0 or no
tokens are active, it runs only local FM. The default legacy ``velocity`` loss
additionally divides loss_lsd by (2*delta)^2 before the weighted sum; select
``lsd_loss_space='displacement'`` to use the formula above directly.

Time convention
---------------
    x0 = Gaussian prior at flow time 0
    x1 = clean action trajectory at flow time 1
    s = source time
    t = terminal time
    X_{s,t}(z) = z + (t-s) * u_theta(z,s,t),  s <= t

Time sampling
-------------
    Input: horizon H, chunk size C, probabilities p_k1, p_fm

    1. Temporal refinement:
        M = H / C, K ∈ Div(M) = {k ∈ N : k divides M}
        P(K=1)=p_k1; each other divisor has probability (1-p_k1)/(len(Div(M))-1)
        Sample one K for the batch. If M=1, always use K=1.

    When M is a power of two, the default candidate set is
    Div(M) = {1, 2, 4, ..., M}, i.e. powers of two in a geometric sequence
    with common ratio 2 (for example, M=8 gives {1, 2, 4, 8}). For other
    values of M, the sampler uses all divisors of M.

    2. Ordered time sampling:
        Sample: t_i ~ U(0,1)
        Sort: t_1 > t_2 > ... > t_K

    3. Interval sampling:
        Sample: ratio ~ p_fm·δ(1)+(1-p_fm)·U(0,1)

    4. Construct:
        s_i = ratio * t_i

Training samples K from the divisors of M=H/C, with p_k1 controlling K=1.
The remaining choices share the residual probability. K independently sorted target
times are expanded over equal action blocks, and one source ratio per sample gives
s=ratio*t. An FM-only curriculum starts with s=t and anneals to the configured
mixture probability. Specifically, ``fm_curriculum_steps`` keeps p_fm=1, then
p_fm decays linearly to its floor over the remaining ``total_train_steps``.
LSD uses a conditional mean over active tokens, so this sampling curriculum
does not ramp its coefficient. Optional ``lsd_warmup_steps`` independently
ramps the LSD weight from zero after FM warmup; zero preserves old weighting.
The LSD residual is computed in displacement units without dividing by the
finite-difference interval. ``lsd_loss_space='velocity'`` (default) restores
historical loss units by rescaling the scalar MSE; ``'displacement'`` uses the
raw increment MSE. With fixed delta, L_displacement=(2*delta)^2*L_velocity.
Deployment still uses the exact rolling staircase needed for cache alignment.

There is no JVP, CSF, Split loss, persistent training cache, or adjacent-window
training sampler. Rolling state exists only at inference.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import Tensor, nn

_VELOCITY_MODES = ("average", "instant")
_LSD_LOSS_SPACES = ("velocity", "displacement")


# =============================================================================
# 1. Configuration
# =============================================================================

@dataclass
class RollFlowConfig:
    horizon: int
    action_dim: int
    chunk_size: int
    finite_difference_delta: float = 0.01
    # Optional relative uniform jitter; avoids training at only one FD scale.
    finite_difference_delta_jitter: float = 0.0
    inference_steps: Optional[int] = None
    velocity_mode: str = "average"
    p_k1: float = 0.7
    p_fm: float = 0.7

    # Pure-FM warmup; after this, p_fm decays linearly to its floor.
    fm_curriculum_steps: int = 20_000
    total_train_steps: Optional[int] = None
    w_fm: float = 1.0
    w_lsd: float = 0.5
    # Both use one displacement residual; velocity preserves old loss units.
    lsd_loss_space: str = "velocity"
    # Independent weight ramp after FM warmup; 0 preserves legacy weighting.
    lsd_warmup_steps: int = 0
    use_ot: bool = True
    clip_velocity: float = 0.0
    iterative_cold_start: bool = False
    reset_cache_each_step: bool = False

    def __post_init__(self) -> None:
        if self.horizon <= 0 or self.action_dim <= 0:
            raise ValueError("horizon/action_dim must be positive")
        if self.chunk_size <= 0 or self.horizon % self.chunk_size:
            raise ValueError("chunk_size must be positive and divide horizon")
        if not 0.0 < self.finite_difference_delta < 0.5:
            raise ValueError("finite_difference_delta must be in (0, 0.5)")
        if not 0.0 <= self.finite_difference_delta_jitter < 1.0:
            raise ValueError("finite_difference_delta_jitter must be in [0, 1)")
        if self.finite_difference_delta * (1 + self.finite_difference_delta_jitter) >= .5:
            raise ValueError("jittered finite_difference_delta must stay below 0.5")
        if self.w_fm < 0 or self.w_lsd < 0:
            raise ValueError("loss weights must be non-negative")
        if self.lsd_loss_space not in _LSD_LOSS_SPACES:
            raise ValueError("lsd_loss_space must be 'velocity' or 'displacement'")
        if self.clip_velocity < 0:
            raise ValueError("clip_velocity must be non-negative")
        if not 0.0 <= self.p_k1 <= 1.0:
            raise ValueError("p_k1 must lie in [0, 1]")
        if not 0.0 <= self.p_fm <= 1.0:
            raise ValueError("p_fm must lie in [0, 1]")
        if self.velocity_mode not in _VELOCITY_MODES:
            raise ValueError("velocity_mode must be 'average' or 'instant'")
        if self.fm_curriculum_steps < 0:
            raise ValueError("fm_curriculum_steps must be non-negative")
        if self.lsd_warmup_steps < 0:
            raise ValueError("lsd_warmup_steps must be non-negative")
        if self.total_train_steps is not None and self.total_train_steps < 0:
            raise ValueError("total_train_steps must be non-negative or None")
        if not self.valid_steps(self.resolved_inference_steps):
            raise ValueError(
                f"inference_steps must lie in [1,{self.num_action_chunks}]"
            )

    @property
    def num_action_chunks(self) -> int:
        return self.horizon // self.chunk_size

    @property
    def resolved_inference_steps(self) -> int:
        return (
            self.num_action_chunks
            if self.inference_steps is None
            else int(self.inference_steps)
        )

    def valid_steps(self, k: int) -> bool:
        return 1 <= int(k) <= self.num_action_chunks


@dataclass(frozen=True)
class TrainingTimes:
    s: Tensor
    t: Tensor
    active: Tensor
    num_time_groups: int
    block_size: int
    ratio: Tensor
    fm_mask: Tensor


@dataclass(frozen=True)
class StaircaseTimes:
    s: Tensor
    t: Tensor
    active: Tensor
    refinement_steps: int
    lower: Tensor
    upper: Tensor


@dataclass(frozen=True)
class RollFlowCacheInfo:
    shape: tuple[int, int, int]
    refinement_steps: int
    device: torch.device
    dtype: torch.dtype


# =============================================================================
# 2. Core algorithm: exactly FM + central-difference LSD
# =============================================================================

def lerp_path(x0: Tensor, x1: Tensor, time: Tensor) -> Tensor:
    return (1.0 - time.float()) * x0.float() + time.float() * x1.float()


def flow_map(x_s: Tensor, s: Tensor, t: Tensor, velocity: Tensor) -> Tensor:
    return x_s.float() + (t - s).float() * velocity.float()


def zero_padded(x: Tensor, pad: Optional[Tensor]) -> Tensor:
    """Remove meaningless values from padded action tokens.

    Loss masking alone is not enough for a Transformer: padded tokens still enter
    self-attention.  Zeroing them here keeps long-horizon batches from injecting
    fresh Gaussian noise through padded positions.  A model-specific attention
    mask can still be supplied through ``model_kwargs`` when available.
    """
    if pad is None:
        return x
    return x.masked_fill(pad.bool().unsqueeze(-1), 0)


def _flow_matching_objective(
    model: nn.Module,
    x_t: Tensor,
    t: Tensor,
    v_gt: Tensor,
    *,
    context: Optional[Tensor],
    pad: Optional[Tensor],
    weight: float,
    model_kwargs: dict[str, Any],
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute the ordinary FM objective without constructing LSD branches."""
    v_local = _predict(model, x_t, t, t, context, model_kwargs)
    fm_loss = masked_mse(v_local - v_gt, pad=pad)
    zero = fm_loss.detach().new_zeros(())
    valid = None if pad is None else (~pad.bool()).unsqueeze(-1)
    v_local_abs = (
        v_local.detach().abs().mean()
        if valid is None else masked_mean(v_local.detach().abs(), valid)
    )
    v_gt_abs = (
        v_gt.detach().abs().mean()
        if valid is None else masked_mean(v_gt.detach().abs(), valid)
    )
    return weight * fm_loss, {
        "fm_loss": fm_loss.detach(),
        "lsd_loss": zero,
        "active_lsd_frac": zero,
        "v_local_abs": v_local_abs,
        "v_gt_abs": v_gt_abs,
        "v_tangent_abs": zero,
        "v_teacher_abs": zero,
    }


def flow_map_increment(v_minus: Tensor, v_plus: Tensor, s: Tensor, t: Tensor,
                       active: Tensor, delta: float) -> Tensor:
    """Difference between the predicted displacements from s to t+ and t-.

    Displacement = duration * average velocity. Both predictions share x_s,
    so we need not add that position before subtracting. Keep arithmetic FP32.
    """
    duration = t.float() - s.float()
    time_offset = delta * active.float()  # Only active terminal times move.
    displacement_minus = (duration - time_offset) * v_minus.float()
    displacement_plus = (duration + time_offset) * v_plus.float()
    return displacement_plus - displacement_minus


def flow_map_tangent(v_minus: Tensor, v_plus: Tensor, s: Tensor, t: Tensor,
                     active: Tensor, delta: float) -> Tensor:
    """Central difference along the *joint* active clock, in FP32.

    Algebraically equals (F(t+delta*a)-F(t-delta*a))/(2*delta),
    without adding and then subtracting the shared source state. Inactive
    outputs may still vary through cross-token coupling. This does not recover
    precision already lost in BF16 network predictions.
    """
    return flow_map_increment(v_minus, v_plus, s, t, active, delta) / (2.0 * delta)


def central_difference_lsd(
    model: nn.Module,
    x0: Tensor,
    x1: Tensor,
    s: Tensor,
    t: Tensor,
    *,
    delta: float,
    active: Optional[Tensor] = None,
    context: Optional[Tensor] = None,
    pad: Optional[Tensor] = None,
    w_fm: float = 1.0,
    w_lsd: float = 0.5,
    lsd_loss_space: str = "velocity",
    model_kwargs: Optional[dict[str, Any]] = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Memory-efficient FM + central-difference LSD objective.

    ``active`` marks tokens carrying a finite off-diagonal interval. Diagonal
    tokens always receive local FM supervision and are excluded from LSD.
    Teacher branches are detached; the student endpoints and local FM predictions
    share one packed forward.

    The student residual is an unnormalized displacement increment minus
    ``2*delta`` times the detached teacher velocity. Legacy ``velocity`` units
    rescale its MSE by ``1/(2*delta)**2``; ``displacement`` leaves it untouched.
    Thus identical w_lsd values do not imply identical training strength.
    """
    if lsd_loss_space not in _LSD_LOSS_SPACES:
        raise ValueError("lsd_loss_space must be 'velocity' or 'displacement'")
    kwargs = {} if model_kwargs is None else model_kwargs
    delta = float(delta)
    active = _lsd_mask(s, t, delta) if active is None else active.bool()
    _validate_training_inputs(x0, x1, s, t, active, pad)
    _validate_times(s, t, delta, active)

    # Padding has no dynamics.  Besides excluding it from the loss, remove both
    # action/noise and time signals so padded tokens cannot pollute attention.
    x0 = zero_padded(x0, pad)
    x1 = zero_padded(x1, pad)
    if pad is not None:
        pad3 = pad.bool().unsqueeze(-1)
        s = s.masked_fill(pad3, 0)
        t = t.masked_fill(pad3, 0)
        active = active & ~pad3
    x_t = lerp_path(x0, x1, t)
    v_gt = x1.float() - x0.float()

    # w_lsd=0 is a true FM fast path: no teacher/tangent graph is constructed.
    if w_lsd <= 0:
        return _flow_matching_objective(
            model, x_t, t, v_gt,
            context=context, pad=pad, weight=w_fm, model_kwargs=kwargs,
        )
    loss_active = active
    if not bool(loss_active.any()):
        return _flow_matching_objective(
            model, x_t, t, v_gt,
            context=context, pad=pad, weight=w_fm, model_kwargs=kwargs,
        )
    x_s = lerp_path(x0, x1, s)

    # Exclude FM-only rows from teacher/tangent forwards; local FM keeps B rows.
    lsd_rows = loss_active.flatten(1).any(dim=1)
    x_s_lsd = x_s[lsd_rows]
    s_lsd = s[lsd_rows]
    t_lsd = t[lsd_rows]
    active_lsd = loss_active[lsd_rows]
    context_lsd = select_batch(context, lsd_rows, x0.shape[0])
    kwargs_lsd = select_batch(kwargs, lsd_rows, x0.shape[0])

    # Freeze inactive terminal times; boundary-excluded tokens need not be diagonal.
    t_minus = torch.where(active_lsd, t_lsd - delta, t_lsd)
    t_plus = torch.where(active_lsd, t_lsd + delta, t_lsd)

    # Autocast can cache a detached low-precision weight when the no-grad
    # teacher runs first, and then silently reuse it for the student. Disable
    # only the cache for these forwards; retain the caller's autocast dtype.
    device_type = x_s.device.type
    autocast_context = (
        torch.autocast(device_type=device_type, dtype=torch.get_autocast_dtype(device_type),
                       cache_enabled=False)
        if torch.is_autocast_enabled(device_type) else nullcontext()
    )
    with autocast_context:
        # 1) Detached self-teacher.
        with torch.no_grad():
            V_st = _predict(model, x_s_lsd, s_lsd, t_lsd, context_lsd, kwargs_lsd)
            X_t_hat = flow_map(x_s_lsd, s_lsd, t_lsd, V_st)
            v_teacher = _predict(model, X_t_hat, t_lsd, t_lsd, context_lsd, kwargs_lsd)

        # 2) One packed student forward: tangent endpoints + local FM.
        V_minus, V_plus, v_local = _parallel_student_predictions(
            model, x_s, x_t, s, t, t_minus, t_plus, context, kwargs, lsd_rows
        )
    # Student: displacement between the two predicted endpoints.
    student_displacement = flow_map_increment(
        V_minus, V_plus, s_lsd, t_lsd, active_lsd, delta
    )
    # Teacher: local velocity * elapsed time between those endpoints.
    # v_teacher was computed under no_grad above; it remains a fixed target.
    teacher_displacement = (2.0 * delta) * v_teacher.float()
    loss_fm = masked_mse(v_local - v_gt, pad=pad)
    pad_lsd = None if pad is None else pad[lsd_rows]
    loss_lsd = masked_mse(
        student_displacement - teacher_displacement,
        pad=pad_lsd,
        valid=active_lsd,
    )
    if lsd_loss_space == "velocity":
        loss_lsd = loss_lsd / (2.0 * delta) ** 2
    loss = w_fm * loss_fm + w_lsd * loss_lsd
    valid_local = None if pad is None else (~pad.bool()).unsqueeze(-1)
    v_local_abs = (
        v_local.detach().abs().mean()
        if valid_local is None else masked_mean(v_local.detach().abs(), valid_local)
    )
    v_gt_abs = (
        v_gt.detach().abs().mean()
        if valid_local is None else masked_mean(v_gt.detach().abs(), valid_local)
    )
    if pad is None:
        active_frac = loss_active.float().mean()
    else:
        valid_tokens = (~pad.bool()).sum().clamp_min(1)
        active_frac = loss_active.squeeze(-1).sum().to(torch.float32) / valid_tokens
    return loss, {
        "fm_loss": loss_fm.detach(),
        "lsd_loss": loss_lsd.detach(),
        "active_lsd_frac": active_frac.detach(),
        "v_local_abs": v_local_abs,
        "v_gt_abs": v_gt_abs,
        # Velocity-unit diagnostic only; no division in the displacement loss graph.
        "v_tangent_abs": masked_mean(student_displacement.detach().abs(), active_lsd) / (2.0 * delta),
        "v_teacher_abs": masked_mean(v_teacher.abs(), active_lsd),
    }


# =============================================================================
# 3. Staircase time sampling
# =============================================================================

class StaircaseTimeSampler:
    """Random grouped training times + exact deployment staircase.
    Training samples K from the divisors of M=H/C.  K=1 has probability
    ``p_k1`` and every other divisor shares the remaining mass.  The K sorted
    target times are expanded over equal contiguous action blocks.  Each sample
    uses one ratio for all groups, s=ratio*t; FM-only samples use ratio=1.
    When M is a power of two, these default candidates are {1, 2, 4, ..., M},
    a geometric sequence with common ratio 2; otherwise all divisors of M are
    candidates.
    Deployment retains the K-level layout required for cache alignment:
    For a sampled interval [R,T] and K refinement levels, chunk j uses
        t_j = T - min(j,   K)/K * (T-R)
        s_j = T - min(j+1, K)/K * (T-R)
    so for j < K:
        t_{j+1} = s_j,
    while j >= K is diagonal at R.  The front chunk is closest to the clean
    endpoint; future chunks are progressively noisier.
    """

    def __init__(self, cfg: RollFlowConfig):
        self.cfg = cfg
        self._train_ks = torch.tensor(
            [k for k in range(1, cfg.num_action_chunks + 1) if cfg.num_action_chunks % k == 0],
            dtype=torch.long,
        )

    def sample_training(
        self,
        batch: int,
        *,
        device,
        dtype=torch.float32,
        generator=None,
        p_fm: Optional[float] = None,
        num_time_groups: Optional[int] = None,
    ) -> TrainingTimes:
        if batch <= 0:
            raise ValueError("batch must be positive")
        device = torch.device(device)
        p_fm = self.cfg.p_fm if p_fm is None else float(p_fm)
        if not 0.0 <= p_fm <= 1.0:
            raise ValueError("p_fm must lie in [0, 1]")
        if num_time_groups is None:
            K = self._sample_k(device, generator)
        else:
            K = int(num_time_groups)
            if K not in self._train_ks.tolist():
                raise ValueError(f"num_time_groups must divide {self.cfg.num_action_chunks}")
        t_group = torch.sort(
            torch.rand(batch, K, 1, device=device, dtype=dtype, generator=generator),
            dim=1,
            descending=True,
        ).values
        ratio = torch.rand(batch, 1, 1, device=device, dtype=dtype, generator=generator)
        fm_mask = torch.rand(batch, 1, 1, device=device, generator=generator) < p_fm
        ratio = torch.where(fm_mask, torch.ones_like(ratio), ratio)
        s_group = ratio * t_group
        repeats = (self.cfg.num_action_chunks // K) * self.cfg.chunk_size
        s = s_group.repeat_interleave(repeats, dim=1)
        t = t_group.repeat_interleave(repeats, dim=1)
        active = _lsd_mask(s, t, self.cfg.finite_difference_delta)
        return TrainingTimes(
            s,
            t,
            active,
            K,
            self.cfg.num_action_chunks // K,
            ratio,
            fm_mask,
        )

    def deployment(
        self,
        batch: int,
        *,
        refinement_steps: int,
        cold: bool,
        device,
        dtype=torch.float32,
    ) -> StaircaseTimes:
        K = int(refinement_steps)
        if not self.cfg.valid_steps(K):
            raise ValueError(f"refinement_steps must lie in [1,{self.cfg.num_action_chunks}]")
        lower = torch.zeros(batch, device=device, dtype=dtype)
        upper = torch.ones(batch, device=device, dtype=dtype)
        s, t, active = self._build(lower, upper, K)
        if cold:
            # One-pass cold start: every token begins at the prior time 0 and
            # jumps directly to its staircase target t.
            s = torch.zeros_like(s)
            active = t > 0
        return StaircaseTimes(s, t, active, K, lower, upper)

    def _sample_k(self, device, generator) -> int:
        ks = self._train_ks.to(device=device)
        if len(ks) == 1:
            return 1
        probs = torch.full(
            (len(ks),),
            (1.0 - self.cfg.p_k1) / (len(ks) - 1),
            device=device,
        )
        probs[0] = self.cfg.p_k1
        index = torch.multinomial(probs, 1, generator=generator)
        return int(ks[index].item())

    def _build(
        self, lower: Tensor, upper: Tensor, K: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        cfg = self.cfg
        M = cfg.num_action_chunks
        j = torch.arange(M, device=lower.device, dtype=lower.dtype)
        gap = (upper - lower)[:, None]
        alpha_t = torch.clamp(j / K, max=1.0)[None, :]
        alpha_s = torch.clamp((j + 1.0) / K, max=1.0)[None, :]
        t_chunk = upper[:, None] - gap * alpha_t
        s_chunk = upper[:, None] - gap * alpha_s
        t = t_chunk.repeat_interleave(cfg.chunk_size, dim=1).unsqueeze(-1)
        s = s_chunk.repeat_interleave(cfg.chunk_size, dim=1).unsqueeze(-1)
        active = (t - s) > self.cfg.finite_difference_delta
        return s, t, active


# =============================================================================
# 4. Engineering helpers
# =============================================================================

def _lsd_mask(s: Tensor, t: Tensor, delta: float) -> Tensor:
    delta = float(delta)
    return ((t - s) > delta + 1e-6) & (t + delta <= 1.0 + 1e-6)


def _validate_times(s: Tensor, t: Tensor, delta: float, active: Tensor) -> None:
    tol = 1e-6
    if s.shape != t.shape:
        raise ValueError("s and t must have identical shapes")
    if bool((s < -tol).any()) or bool((t > 1.0 + tol).any()) or bool((s > t + tol).any()):
        raise ValueError("require 0 <= s <= t <= 1")
    # Use the same subtraction and threshold as _lsd_mask. Reassociating this
    # as s >= t-delta-tol can disagree by one FP32 ULP at the boundary and
    # reject samples which the sampler itself marked active.
    if bool((active & ((t - s) <= delta + tol)).any()):
        raise ValueError("active LSD tokens require s < t-delta")
    if bool((active & (t + delta > 1.0 + tol)).any()):
        raise ValueError("active LSD tokens require t+delta <= 1")


def _validate_training_inputs(
    x0: Tensor,
    x1: Tensor,
    s: Tensor,
    t: Tensor,
    active: Tensor,
    pad: Optional[Tensor],
) -> None:
    if x0.shape != x1.shape or x0.ndim != 3:
        raise ValueError("x0 and x1 must have identical [B,H,A] shapes")
    expected_times = (*x0.shape[:2], 1)
    if s.shape != expected_times or t.shape != expected_times:
        raise ValueError(f"s and t must have shape {expected_times}")
    if active.shape != expected_times:
        raise ValueError(f"active must have shape {expected_times}")
    if pad is not None and pad.shape != x0.shape[:2]:
        raise ValueError(f"pad must have shape {tuple(x0.shape[:2])}")


def _predict(
    model: nn.Module,
    z: Tensor,
    s: Tensor,
    t: Tensor,
    context: Optional[Tensor],
    kwargs: dict[str, Any],
) -> Tensor:
    out = model(z, s, t, context, **kwargs)
    if out.shape != z.shape:
        raise ValueError(f"model output {tuple(out.shape)} must match z {tuple(z.shape)}")
    return out


def _parallel_student_predictions(
    model,
    x_s,
    x_t,
    s,
    t,
    t_minus,
    t_plus,
    context,
    kwargs,
    lsd_rows,
):
    B = x_s.shape[0]
    B_lsd = t_minus.shape[0]
    out = _predict(
        model,
        torch.cat((x_s[lsd_rows], x_s[lsd_rows], x_t), dim=0),
        torch.cat((s[lsd_rows], s[lsd_rows], t), dim=0),
        torch.cat((t_minus, t_plus, t), dim=0),
        merge_lsd_and_full_batch(context, lsd_rows, B),
        merge_lsd_and_full_batch(kwargs, lsd_rows, B),
    )
    return out.split((B_lsd, B_lsd, B), dim=0)


def select_batch(value: Any, rows: Tensor, batch: int) -> Any:
    """Select batch-aligned tensors recursively; leave constants unchanged."""
    if value is None:
        return None
    if torch.is_tensor(value):
        return value[rows] if value.ndim and value.shape[0] == batch else value
    if isinstance(value, dict):
        return {key: select_batch(item, rows, batch) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(select_batch(item, rows, batch) for item in value)
    if isinstance(value, list):
        return [select_batch(item, rows, batch) for item in value]
    return value


def merge_lsd_and_full_batch(value: Any, rows: Tensor, batch: int) -> Any:
    """Build the [LSD-, LSD+, full-FM] batch recursively."""
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.ndim and value.shape[0] == batch:
            selected = value[rows]
            return torch.cat((selected, selected, value), dim=0)
        return value
    if isinstance(value, dict):
        return {
            key: merge_lsd_and_full_batch(item, rows, batch)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(merge_lsd_and_full_batch(item, rows, batch) for item in value)
    if isinstance(value, list):
        return [merge_lsd_and_full_batch(item, rows, batch) for item in value]
    return value


def masked_mse(
    error: Tensor,
    *,
    pad: Optional[Tensor] = None,
    valid: Optional[Tensor] = None,
) -> Tensor:
    """Mean squared error over valid action tokens, robust to invalid NaNs.
    Mask *before* squaring.  Multiplying an already-computed NaN loss by zero
    would still produce NaN and can poison long-horizon batches containing
    invalid/padded positions.
    """
    mask = torch.ones(error.shape[:2], device=error.device, dtype=torch.bool)
    if pad is not None:
        mask &= ~pad.bool()
    if valid is not None:
        valid2d = valid.squeeze(-1) if valid.ndim == 3 else valid
        mask &= valid2d.bool()
    mask3 = mask.unsqueeze(-1)
    safe_error = torch.where(mask3, error.float(), torch.zeros((), device=error.device))
    per_token = safe_error.square().mean(dim=-1)
    denom = mask.sum().to(per_token.dtype).clamp_min(1.0)
    return per_token.sum() / denom


def masked_mean(value: Tensor, valid: Tensor) -> Tensor:
    """Mean over valid entries without host syncs; ignore invalid NaNs."""
    mask = valid.expand_as(value).bool()
    safe_value = torch.where(mask, value, torch.zeros((), device=value.device, dtype=value.dtype))
    denom = mask.sum().to(value.dtype).clamp_min(1.0)
    return safe_value.sum() / denom


@torch.no_grad()
def ot_match(x1: Tensor, x0: Tensor, pad: Optional[Tensor] = None) -> Tensor:
    """Match noise to targets, excluding padded target timesteps from cost."""
    if x1.shape != x0.shape or x1.ndim != 3:
        raise ValueError("x1 and x0 must have identical [B,H,A] shapes")
    if pad is not None and pad.shape != x1.shape[:2]:
        raise ValueError(f"pad must have shape {tuple(x1.shape[:2])}")
    if x1.shape[0] <= 1:
        return x0
    from scipy.optimize import linear_sum_assignment
    if pad is None or not bool(pad.any()):
        cost = torch.cdist(x1.float().flatten(1), x0.float().flatten(1))
    else:
        # Masked squared distance without allocating [B,B,H,A].  This matters
        # for long horizons where the previous broadcast can become expensive.
        x1f = x1.float().flatten(1)
        x0f = x0.float().flatten(1)
        valid = (~pad.bool()).unsqueeze(-1).expand_as(x1).float().flatten(1)
        denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        target_sq = (valid * x1f.square()).sum(dim=1, keepdim=True)
        noise_sq = valid @ x0f.square().T
        cross = (valid * x1f) @ x0f.T
        cost = (target_sq + noise_sq - 2.0 * cross).clamp_min_(0.0) / denom
    cost = cost.cpu()
    row, col = linear_sum_assignment(cost.numpy())
    row = torch.as_tensor(row, device=x0.device)
    col = torch.as_tensor(col, device=x0.device)
    matched = torch.empty_like(x0)
    matched[row] = x0[col]
    return matched


def infer_device_dtype(model, context=None, device=None, dtype=None):
    """Resolve action-cache placement from the model, with context fallback."""
    # RollFlow is also called with a bound prediction method (for example
    # ``action_head._predict_velocity``), not only with an ``nn.Module``.
    # In that case resolve placement from the owning module.
    candidates = (model, getattr(model, "__self__", None))
    p = None
    for candidate in candidates:
        parameters = getattr(candidate, "parameters", None)
        if callable(parameters):
            p = next(parameters(), None)
            if p is not None:
                break
    if p is not None:
        model_device, model_dtype = p.device, p.dtype
    else:
        model_device = context.device if context is not None else torch.device("cpu")
        model_dtype = context.dtype if context is not None else torch.float32
    return (
        model_device if device is None else torch.device(device),
        model_dtype if dtype is None else dtype,
    )


# =============================================================================
# 5. Public wrapper: ordinary training samples + rolling inference
# =============================================================================

class RollFlow:

    def __init__(self, cfg: RollFlowConfig):
        self.cfg = cfg
        self.time = StaircaseTimeSampler(cfg)
        self._cache: Optional[Tensor] = None
        self._cache_steps: Optional[int] = None
        self._cache_mode: Optional[str] = None

    def loss(
        self,
        model: nn.Module,
        x: Tensor,
        step: int = 0,
        context: Optional[Tensor] = None,
        pad: Optional[Tensor] = None,
        **model_kwargs,
    ) -> tuple[Tensor, dict[str, Any]]:
        self._check_actions(x)
        cfg = self.cfg
        B = x.shape[0]
        if pad is not None and pad.shape != (B, cfg.horizon):
            raise ValueError(f"pad must be [B,{cfg.horizon}]")
        x1 = x
        x0 = torch.randn_like(x1)
        if cfg.use_ot:
            x0 = ot_match(x1, x0, pad=pad)

        # Padded tokens have no physical meaning.  Sanitizing both endpoints
        # prevents long-horizon padding from injecting random noise into attention.
        x1 = zero_padded(x1, pad)
        x0 = zero_padded(x0, pad)
        warmup = cfg.fm_curriculum_steps
        total = cfg.total_train_steps
        if step < warmup:
            p_fm = 1.0
        elif total is None or total <= warmup:
            p_fm = cfg.p_fm
        else:
            progress = min(max((step - warmup) / (total - warmup), 0.0), 1.0)
            p_fm = 1.0 - progress * (1.0 - cfg.p_fm)
        times = self.time.sample_training(B, device=x.device, p_fm=p_fm)
        lsd_weight = cfg.w_lsd
        if cfg.lsd_warmup_steps:
            lsd_weight *= min(max((step - warmup) / cfg.lsd_warmup_steps, 0.0), 1.0)
        delta = cfg.finite_difference_delta
        if lsd_weight > 0 and p_fm < 1 and cfg.finite_difference_delta_jitter:
            # Draw AFTER noise/time sampling so a fixed per-step seed keeps
            # the FM input and time samples matched to the non-jittered arm.
            delta *= 1 + cfg.finite_difference_delta_jitter * (
                2 * float(torch.rand((), device=x.device)) - 1
            )
        active = _lsd_mask(times.s, times.t, delta)

        # Pure-FM is intentionally a direct one-forward path.  It keeps the
        # exact staircase t sampling while avoiding all LSD allocations/syncs.
        if lsd_weight <= 0 or p_fm >= 1.0:
            t = times.t
            if pad is not None:
                t = t.masked_fill(pad.bool().unsqueeze(-1), 0)
            x_t = lerp_path(x0, x1, t)
            v_gt = x1.float() - x0.float()
            loss, stats = _flow_matching_objective(
                model,
                x_t,
                t,
                v_gt,
                context=context,
                pad=pad,
                weight=cfg.w_fm,
                model_kwargs=model_kwargs,
            )
        else:
            loss, stats = central_difference_lsd(
                model,
                x0,
                x1,
                times.s,
                times.t,
                delta=delta,
                active=active,
                context=context,
                pad=pad,
                w_fm=cfg.w_fm,
                w_lsd=lsd_weight,
                lsd_loss_space=cfg.lsd_loss_space,
                model_kwargs=model_kwargs,
            )

        # Keep diagnostics small but sufficient to catch long-horizon collapse.
        pad_frac = 0.0 if pad is None else float(pad.float().mean().detach().cpu())
        return loss, {
            "loss": float(loss.detach().cpu()),
            "fm_loss": float(stats["fm_loss"].cpu()),
            "lsd_loss": float(stats["lsd_loss"].cpu()),
            "lsd_weight": lsd_weight,
            "finite_difference_delta": delta,
            "weighted_lsd_loss": lsd_weight * float(stats["lsd_loss"].cpu()),
            "active_lsd_frac": float(stats["active_lsd_frac"].cpu()),
            "v_local_abs": float(stats["v_local_abs"].cpu()),
            "v_gt_abs": float(stats["v_gt_abs"].cpu()),
            "v_tangent_abs": float(stats["v_tangent_abs"].cpu()),
            "v_teacher_abs": float(stats["v_teacher_abs"].cpu()),
            "num_time_groups": times.num_time_groups,
            "train_block_size": times.block_size,
            "p_fm": p_fm,
            "fm_only_frac": float(times.fm_mask.float().mean().cpu()),
            "source_ratio": float(times.ratio.mean().cpu()),
            "s_min": float(times.s.min().cpu()),
            "s_max": float(times.s.max().cpu()),
            "s_mean": float(times.s.mean().cpu()),
            "t_min": float(times.t.min().cpu()),
            "t_max": float(times.t.max().cpu()),
            "t_mean": float(times.t.mean().cpu()),
            "pad_frac": pad_frac,
            "use_ot": bool(cfg.use_ot),
        }

    def reset(self) -> None:
        self._cache = None
        self._cache_steps = None
        self._cache_mode = None

    @property
    def cache_info(self) -> Optional[RollFlowCacheInfo]:
        if self._cache is None or self._cache_steps is None:
            return None
        return RollFlowCacheInfo(
            tuple(self._cache.shape),
            self._cache_steps,
            self._cache.device,
            self._cache.dtype,
        )

    @torch.no_grad()
    def step(
        self,
        model: nn.Module,
        batch: int,
        context: Optional[Tensor] = None,
        *,
        device=None,
        dtype=None,
        refinement_steps: Optional[int] = None,
        velocity_mode: Optional[str] = None,
        **model_kwargs,
    ) -> Tensor:
        """Advance the rolling cache and return the next execution chunk.
        ``horizon`` is the cache length, ``chunk_size`` is the number of
        actions returned per call, and ``refinement_steps`` is the number of
        flow refinements.  For example, ``H=128, C=32, K=4`` keeps four
        32-action blocks in the cache and returns ``[B, 32, action_dim]``.
        """
        velocity_mode = self.cfg.velocity_mode if velocity_mode is None else velocity_mode
        if velocity_mode not in _VELOCITY_MODES:
            raise ValueError("velocity_mode must be 'average' or 'instant'")
        cfg = self.cfg
        K = cfg.resolved_inference_steps if refinement_steps is None else int(refinement_steps)
        if not cfg.valid_steps(K):
            raise ValueError(f"refinement_steps must lie in [1,{cfg.num_action_chunks}]")
        device, dtype = infer_device_dtype(model, context, device, dtype)
        if cfg.reset_cache_each_step:
            self.reset()
        elif self._cache is not None and self._cache_mode != velocity_mode:
            self.reset()
        z, cold = self._rolling_input(batch, device, dtype, K)
        times = self.time.deployment(
            batch,
            refinement_steps=K,
            cold=cold,
            device=device,
        )
        if cold and cfg.iterative_cold_start:
            cache = self._cold_iterative(
                model, z, times.t, K, context, model_kwargs, dtype, velocity_mode
            )
        else:
            query_t = times.s if velocity_mode == "instant" else times.t
            velocity = self._predict_inference(
                model, z, times.s, query_t, context, model_kwargs
            )
            cache = flow_map(z, times.s, times.t, velocity).to(dtype)
        self._cache = cache.detach()
        self._cache_steps = K
        self._cache_mode = velocity_mode
        return self._cache[:, : cfg.chunk_size].clone()

    def _rolling_input(self, batch, device, dtype, K):
        cfg = self.cfg
        valid = (
            self._cache is not None
            and self._cache_steps == K
            and self._cache.shape == (batch, cfg.horizon, cfg.action_dim)
            and self._cache.device == device
            and self._cache.dtype == dtype
        )
        if not valid:
            return torch.randn(batch, cfg.horizon, cfg.action_dim, device=device, dtype=dtype), True
        keep = self._cache[:, cfg.chunk_size :]
        tail = torch.randn(batch, cfg.chunk_size, cfg.action_dim, device=device, dtype=dtype)
        return torch.cat((keep, tail), dim=1), False

    def _cold_iterative(self, model, z, final_t, K, context, kwargs, dtype, velocity_mode):
        current_s = torch.zeros_like(final_t)
        for level in range(1, K + 1):
            next_t = torch.minimum(final_t, torch.full_like(final_t, level / K))
            query_t = current_s if velocity_mode == "instant" else next_t
            v = self._predict_inference(model, z, current_s, query_t, context, kwargs)
            z = flow_map(z, current_s, next_t, v).to(dtype)
            current_s = next_t
        return z

    def _predict_inference(self, model, z, s, t, context, kwargs):
        v = _predict(model, z, s, t, context, kwargs)
        clip = self.cfg.clip_velocity
        return v if clip <= 0 else clip * torch.tanh(v / clip)

    def _check_actions(self, x: Tensor) -> None:
        expected = (self.cfg.horizon, self.cfg.action_dim)
        if x.ndim != 3 or tuple(x.shape[1:]) != expected:
            raise ValueError(f"x must be [B,{expected[0]},{expected[1]}]")
