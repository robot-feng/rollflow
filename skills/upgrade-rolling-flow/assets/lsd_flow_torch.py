"""Optional, explicit straight-path finite-difference Lagrangian distillation.

This is a path-explicit extension of the user's RollFlow clean FD objective,
not a verbatim copy, MeanFlow identity loss, or a generic diffusion converter.
The instantaneous asset remains independent. Canonical progress is 0=noise,
1=data; the public callback uses the policy's original time/sign convention.

Only affine zero-endpoint interpolation is implemented. Preserve the native
noise sampler, local FM reduction and context preparation outside this file.
Teacher and students must evaluate a deterministic field of the same model;
prepare the observation once, detach it only inside teacher no_grad scopes.
No guessed recursive batch packing or automatic OT is performed.
"""

from __future__ import annotations

from collections.abc import Callable
import math

import torch
from torch import Tensor, nn

from rolling_flow_torch import FlowConvention, RollingFlow, _check_convention, _positive_int


def progress(time: Tensor, convention: FlowConvention) -> Tensor:
    _check_convention(convention)
    return time.float() if convention is FlowConvention.NOISE_AT_ZERO else 1 - time.float()


def native_time(time: Tensor, convention: FlowConvention) -> Tensor:
    return progress(time, convention)


def interior_warm_endpoints(
    scale: Tensor,
    *,
    convention: FlowConvention,
    horizon: int = 64,
    execution_horizon: int = 16,
    refinements: int = 2,
    fd_fraction: float = 0.01,
) -> tuple[Tensor, Tensor]:
    """Explicit optional near-warm training pairs, without choosing their law.

    ``scale`` is caller-sampled FP32/FP64 [B] canonical progress q. Warm
    progress blocks are S_j=max(1-(j+1)/R,0), T_j=max(1-j/R,0).
    Return native endpoints (q*S,q*T). R>1 yields nonradial endpoint pairs
    that a shared S=rho*T sampler cannot produce. q<=R/(R+fd_fraction)
    supplies room for the symmetric radial FD stencil. q=1 is deliberately
    rejected: exact warm endpoints touch the data boundary and are not
    eligible for this central FD loss. The caller owns mixture probabilities,
    q distribution, FM sampling, prefix overrides and quality evaluation.
    """
    _check_convention(convention)
    for value, name in (
        (horizon, "horizon"),
        (execution_horizon, "execution_horizon"),
        (refinements, "refinements"),
    ):
        _positive_int(value, name)
    if horizon % execution_horizon or refinements > horizon // execution_horizon:
        raise ValueError("require C dividing H and 1 <= refinements <= H/C")
    if not math.isfinite(fd_fraction) or not 0 < fd_fraction < 1:
        raise ValueError("fd_fraction must be finite and in (0,1)")
    if (
        not isinstance(scale, Tensor)
        or scale.ndim != 1
        or scale.numel() == 0
        or scale.dtype not in (torch.float32, torch.float64)
    ):
        raise ValueError("scale must be nonempty FP32/FP64 [B]")
    upper = refinements / (refinements + fd_fraction)
    if not torch.isfinite(scale).all() or not ((scale > 0) & (scale <= upper)).all():
        raise ValueError("scale must be positive with enough room for the radial FD stencil")
    blocks = torch.arange(horizon // execution_horizon, device=scale.device, dtype=scale.dtype)
    target = (1 - blocks / refinements).clamp_min(0).repeat_interleave(execution_horizon)
    source = (1 - (blocks + 1) / refinements).clamp_min(0).repeat_interleave(execution_horizon)
    source = scale[:, None] * source[None]
    target = scale[:, None] * target[None]
    return native_time(source, convention), native_time(target, convention)


def masked_mse(prediction: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    """Example reducer, NOT a replacement for a policy's original reduction."""
    error = torch.where(valid, prediction.float() - target.float(), 0.0)
    return error.square().sum() / valid.sum().clamp_min(1)


def require_deterministic_modules(model: nn.Module) -> None:
    """Catch common stochastic/stateful fields; custom RNG needs separate audit.

    Does not mutate train/eval flags. no_grad() alone does not remove dropout.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.modules.dropout._DropoutNd) and module.training and module.p:
            raise ValueError(f"LSD requires deterministic field evaluation: active dropout at {name}")
        if isinstance(module, nn.modules.batchnorm._BatchNorm) and module.training:
            raise ValueError(f"LSD requires fixed batch statistics: training BatchNorm at {name}")
        if isinstance(module, nn.MultiheadAttention) and module.training and module.dropout:
            raise ValueError(f"LSD requires deterministic attention: dropout at {name}")
        if isinstance(module, nn.RNNBase) and module.training and module.dropout and module.num_layers > 1:
            raise ValueError(f"LSD requires deterministic recurrent layers: dropout at {name}")


class EndpointResidual(nn.Module):
    """Add a gap-gated residual after the original completed time condition.

    Accept original conditions [B,E] or [B,H,E]. Source/target times must have
    the same leading shape. Original parameter names/shapes stay in the host.
    The host's checkpoint loader must allow only this new branch's keys.

    Multiplying by the native endpoint gap guarantees exact diagonal identity
    after training, not only at initialization. Zero ONLY the new last
    projection initially. Store ``conditioning_schema`` in the host config;
    existing trained ungated interval weights are not function compatible.
    """

    conditioning_schema = "native_gap_gated_v1"

    def __init__(self, width: int, features: int = 32):
        super().__init__()
        _positive_int(width, "width")
        _positive_int(features, "features")
        if features % 2:
            raise ValueError("features must be even")
        self.register_buffer("frequencies", torch.logspace(0, 2, features // 2), persistent=False)
        self.input = nn.Linear(features, width)
        self.output = nn.Linear(width, width)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, condition: Tensor, source: Tensor, target: Tensor) -> Tensor:
        if source.shape != target.shape or source.shape != condition.shape[:-1]:
            raise ValueError("endpoint shapes must match the condition's leading dimensions")
        gap = (target.float() - source.float())[..., None]
        phase = gap * self.frequencies.float()
        features = torch.cat((phase.sin(), phase.cos()), -1).to(self.input.weight.dtype)
        residual = self.output(torch.nn.functional.silu(self.input(features)))
        return condition + (gap * residual.float()).to(condition.dtype)


def lsd_loss(
    predict: Callable[[Tensor, Tensor, Tensor], Tensor],
    *,
    noise: Tensor,
    actions: Tensor,
    source_time: Tensor,
    target_time: Tensor,
    convention: FlowConvention,
    fm_reduce: Callable[[Tensor, Tensor, Tensor], Tensor],
    lsd_reduce: Callable[[Tensor, Tensor, Tensor], Tensor],
    fm_time: Tensor | None = None,
    valid_mask: Tensor | None = None,
    dynamics_mask: Tensor | None = None,
    fm_weight: float = 1.0,
    lsd_weight: float = 0.5,
    fd_fraction: float = 0.01,
    min_fd_fraction: float = 1e-4,
    deterministic: bool = False,
) -> tuple[Tensor, dict]:
    """Local native FM plus a straight-path, stop-gradient self-teacher.

    All times are [B,H] in the NATIVE convention. Source is noisier than target.
    predict returns native signed interval velocity [B,H,D]. Both reducers are
    explicit so native masks/weights/reduction cannot be silently replaced.
    valid_mask is LOSS validity per ELEMENT [B,H,D]. dynamics_mask separately
    marks physically present dimensions/tokens; only it sanitizes inputs and
    removes motion. Loss-only masking must not change attention inputs or the
    endpoint direction. Both default to all-present/all-supervised. Padding
    attention masks belong to predict and must follow the native model.

    For canonical delta=t-s, perturb the WHOLE endpoint along delta:
    t_plus/minus=t +/- h*delta. The normalized tangent is
    (U_plus+U_minus)/2 + (U_plus-U_minus)/(2*h).
    h is shared across every token in a row and capped by endpoint room.
    Rows with insufficient room are excluded from LSD, never from local FM.

    Five sequential B-sized field calls when active: two no_grad teacher calls,
    two gradient-bearing tangent calls, and one local FM call. They share the
    caller's same prepared context; teachers do not detach the student context.
    w_lsd=0 or no eligible row executes exactly one local FM call. Returning
    statistics does not retain the student graph beyond the returned loss.
    """
    _check_convention(convention)
    for name, value in (("fm_weight", fm_weight), ("lsd_weight", lsd_weight)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not (0 < min_fd_fraction <= fd_fraction < 1):
        raise ValueError("require 0 < min_fd_fraction <= fd_fraction < 1")
    if actions.ndim != 3 or noise.shape != actions.shape or actions.shape[0] == 0:
        raise ValueError("actions and noise must be matching nonempty [B,H,D]")
    if actions.device != noise.device or not actions.is_floating_point() or not noise.is_floating_point():
        raise ValueError("actions and noise must be floating tensors on the same device")
    if not deterministic and lsd_weight:
        raise ValueError("verify deterministic field evaluation before enabling LSD")
    times_shape = actions.shape[:2]
    for time in (source_time, target_time, target_time if fm_time is None else fm_time):
        if time.shape != times_shape or time.device != actions.device or not time.is_floating_point():
            raise ValueError("each time must be a floating [B,H] tensor on the action device")
        if lsd_weight and time.dtype not in (torch.float32, torch.float64):
            raise ValueError("LSD endpoints must be sampled/stored in FP32 or FP64 before finite differences")
        if not torch.isfinite(time).all() or not ((time >= 0) & (time <= 1)).all():
            raise ValueError("reference times must be finite in [0,1]")
    present = torch.ones_like(actions, dtype=torch.bool) if dynamics_mask is None else dynamics_mask
    if present.shape != actions.shape or present.dtype != torch.bool or present.device != actions.device:
        raise ValueError("dynamics_mask must be bool [B,H,D] on the action device")
    valid = present if valid_mask is None else valid_mask
    if valid.shape != actions.shape or valid.dtype != torch.bool or valid.device != actions.device:
        raise ValueError("valid_mask must be bool [B,H,D] on the action device")
    if (valid & ~present).any():
        raise ValueError("loss cannot supervise physically absent coordinates")
    token_valid = present.any(-1)
    data = torch.where(present, actions.float(), 0.0)
    prior = torch.where(present, noise.float(), 0.0)
    if not torch.isfinite(data).all() or not torch.isfinite(prior).all():
        raise ValueError("nonfinite valid action/noise")
    s = torch.where(token_valid, progress(source_time, convention), 0.0)
    t = torch.where(token_valid, progress(target_time, convention), 0.0)
    if (s > t).any():
        raise ValueError("source must be no cleaner than target in the native convention")
    f = torch.where(token_valid, progress(target_time if fm_time is None else fm_time, convention), 0.0)
    sign = 1 if convention is FlowConvention.NOISE_AT_ZERO else -1
    local_target = sign * (data - prior)
    batch_calls: list[dict] = []

    def field(x, start, end, role):
        out = predict(x, native_time(start, convention), native_time(end, convention))
        if not isinstance(out, Tensor) or out.shape != actions.shape or out.device != actions.device:
            raise ValueError("predict must return a floating velocity of the action shape/device")
        if not out.is_floating_point():
            raise TypeError("predict must return floating velocity")
        batch_calls.append({"role": role, "batch": actions.shape[0], "grad_enabled": torch.is_grad_enabled()})
        return torch.where(present, out.float(), 0.0)

    delta = t - s
    moving = (delta > 0) & token_valid
    room = torch.where(moving, (1 - t) / delta.clamp_min(torch.finfo(torch.float32).tiny), torch.inf)
    h = room.amin(dim=1).clamp(max=fd_fraction)
    supervised_moving = (valid & moving[..., None]).flatten(1).any(1)
    eligible = supervised_moving & (h >= min_fd_fraction)
    lsd_valid = valid & moving[..., None] & eligible[:, None, None]
    # Ineligible ROWS remain exactly unchanged; never clip tokenwise direction.
    h = torch.where(eligible, h, fd_fraction)
    perturb = torch.where(eligible[:, None], delta * h[:, None], 0.0)
    x_fm = prior + f[..., None] * (data - prior)
    lsd_value = None
    if lsd_weight and bool(eligible.any()):
        x_s = prior + s[..., None] * (data - prior)
        with torch.no_grad():
            u_teacher = sign * field(x_s, s, t, "teacher_map")
            x_hat = torch.where(moving[..., None], x_s + delta[..., None] * u_teacher, x_s)
            teacher = sign * field(x_hat, t, t, "teacher_local")
        minus = sign * field(x_s, s, t - perturb, "student_minus")
        plus = sign * field(x_s, s, t + perturb, "student_plus")
        tangent = 0.5 * (plus + minus) + (plus - minus) / (2 * h[:, None, None])
        lsd_value = lsd_reduce(tangent, teacher, lsd_valid)
    local = field(x_fm, f, f, "student_local")
    fm_value = fm_reduce(local, local_target, valid)
    if lsd_value is None:
        lsd_value = fm_value.new_zeros(())
    loss = fm_weight * fm_value + lsd_weight * lsd_value
    for name, value in (("fm_loss", fm_value), ("lsd_loss", lsd_value), ("loss", loss)):
        if value.ndim != 0 or not torch.isfinite(value):
            raise FloatingPointError(f"{name} must be a finite scalar")
    return loss, {
        "fm_loss": fm_value.detach(),
        "lsd_loss": lsd_value.detach(),
        "eligible_rows": int(eligible.sum()) if lsd_weight else 0,
        "skipped_lsd_rows": int((supervised_moving & ~eligible).sum()) if lsd_weight else 0,
        "calls": batch_calls,
        "batch_equivalent_forward_calls": len(batch_calls),
        "gradient_batch_equivalent_forward_calls": sum(c["grad_enabled"] for c in batch_calls),
        "path": "straight",
    }


class RollingLSD(RollingFlow):
    """Interval-velocity rolling with explicit refinement/cold-NFE budgets.

    Default H64/C16, R2 lifecycle refinements and 2 cold serial calls. Warm
    always one call. R may be 1..H/C; future idle tokens remain at the prior.
    Cold calls subdivide the straight path from zero to the final staircase.
    Unlike the instantaneous sampler, each model call receives BOTH endpoints.
    Assumptions and state/consumption ownership match RollingFlow.
    """

    def __init__(
        self,
        *,
        action_dim: int,
        convention: FlowConvention,
        horizon: int = 64,
        execution_horizon: int = 16,
        refinements: int = 2,
        cold_steps: int = 2,
    ):
        super().__init__(
            action_dim=action_dim, convention=convention, horizon=horizon, execution_horizon=execution_horizon
        )
        _positive_int(refinements, "refinements")
        _positive_int(cold_steps, "cold_steps")
        if refinements > self.steps:
            raise ValueError("refinements must not exceed H/C")
        self.refinements = refinements
        self.cold_steps = cold_steps

    @torch.no_grad()
    def step(
        self, predict, *, noise, consumed_actions=0, action_dim_mask=None, stream_ids=None, context_key=None
    ):
        return self._interval_transaction(
            predict,
            noise=noise,
            consumed_actions=consumed_actions,
            action_dim_mask=action_dim_mask,
            stream_ids=stream_ids,
            context_key=context_key,
        )

    def reset(self):
        super().reset()
        self._cache_budget = None

    def _interval_transaction(
        self, predict, *, noise, consumed_actions, action_dim_mask, stream_ids, context_key
    ):
        # State validation intentionally mirrors the standalone instantaneous
        # contract; no artificial callback is passed through its Euler loop.
        self._validate_configuration()
        _positive_int(self.refinements, "refinements")
        _positive_int(self.cold_steps, "cold_steps")
        if self.refinements > self.steps:
            raise ValueError("refinements must not exceed H/C")
        if (
            not isinstance(noise, Tensor)
            or not noise.is_floating_point()
            or noise.ndim != 3
            or noise.shape[0] == 0
        ):
            raise ValueError("noise must be a floating nonempty [B,N,D] tensor")
        batch, device = noise.shape[0], noise.device
        if noise.shape != (batch, self.noise_horizon, self.action_dim):
            raise ValueError("noise has the wrong batch/horizon/action shape")
        cold = self._cache is None
        if not cold and self._cache_budget != (self.refinements, self.cold_steps):
            raise ValueError("interval budgets changed; reset before changing the schedule")
        if type(consumed_actions) is not int or consumed_actions != (0 if cold else self.execution_horizon):
            raise ValueError("consume exactly C actions between calls or reset")
        if stream_ids is None:
            if batch != 1:
                raise ValueError("batched rolling requires stable ordered stream_ids")
            stream_ids = (0,)
        if not isinstance(stream_ids, (tuple, list)) or len(stream_ids) != batch:
            raise ValueError("one stable stream ID per row is required")
        stream_ids = tuple(stream_ids)
        if any(type(s) not in (str, int) for s in stream_ids) or len(set(stream_ids)) != batch:
            raise ValueError("stream IDs must be unique strings/integers")
        if context_key is not None and not isinstance(context_key, str):
            raise TypeError("context_key must be a string or None")
        if not cold and (stream_ids != self._stream_ids or context_key != self._context_key):
            raise ValueError("stream/context identity changed; reset first")
        shape = (batch, self.horizon, self.action_dim)
        if not cold and (self._cache.shape != shape or self._cache.device != device):
            raise ValueError("batch/device changed; reset first")
        if action_dim_mask is None:
            mask = torch.ones(batch, self.action_dim, dtype=torch.bool, device=device)
        else:
            if not isinstance(action_dim_mask, Tensor) or action_dim_mask.dtype != torch.bool:
                raise TypeError("action_dim_mask must be a boolean tensor, not inferred from numbers")
            if action_dim_mask.shape != (batch, self.action_dim):
                raise ValueError("action_dim_mask must have shape [B,D]")
            mask = action_dim_mask.to(device=device, dtype=torch.bool)
        if not cold and not torch.equal(mask, self._action_dim_mask):
            raise ValueError("action dimension mask changed; reset first")
        x = (
            noise.float().clone()
            if cold
            else torch.cat((self._cache[:, self.execution_horizon :], noise.float()), 1)
        )
        invalid = ~mask[:, None]
        x = x.masked_fill(invalid, 0)
        if not torch.isfinite(x).all():
            raise ValueError("nonfinite valid noise/cache")
        block = torch.arange(self.steps, device=device, dtype=torch.float32)
        final = (1 - block / self.refinements).clamp_min(0)
        final = final.repeat_interleave(self.execution_horizon)[None].expand(batch, -1)
        source = torch.zeros_like(final) if cold else (final - 1 / self.refinements).clamp_min(0)
        nfe = self.cold_steps if cold else 1
        for index in range(nfe):
            target = final * ((index + 1) / nfe) if cold else final
            raw = predict(x, native_time(source, self.convention), native_time(target, self.convention))
            if (
                not isinstance(raw, Tensor)
                or not raw.is_floating_point()
                or raw.shape != shape
                or raw.device != device
            ):
                raise ValueError("interval predictor must preserve floating shape/device")
            velocity = raw.float().masked_fill(invalid, 0)
            signed_dt = native_time(target, self.convention) - native_time(source, self.convention)
            x = torch.where(signed_dt[..., None] != 0, x + signed_dt[..., None] * velocity, x)
            x = x.masked_fill(invalid, 0)
            if not torch.isfinite(x).all():
                raise FloatingPointError("nonfinite active interval update; state not committed")
            source = target
        self._cache = x.detach()
        self._stream_ids = stream_ids
        self._context_key = context_key
        self._action_dim_mask = mask.clone()
        self._cache_budget = (self.refinements, self.cold_steps)
        self.last_nfe = nfe
        return x[:, : self.execution_horizon].clone()
