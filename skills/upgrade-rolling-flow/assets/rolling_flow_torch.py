"""Grouped-time training and rolling Euler inference for existing FM policies.

Use only for the affine, zero-endpoint path with a temporally independent
noise prior. Keep each policy's original time/noise
samplers, velocity target, action processing and FlowConvention. The policy's
predictor must genuinely support [B,H] time conditioning in one forward.
"""

from collections.abc import Callable
from enum import Enum
import math

import torch
from torch import Tensor


class FlowConvention(Enum):
    NOISE_AT_ONE = "noise_at_one"
    NOISE_AT_ZERO = "noise_at_zero"


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _check_convention(convention: FlowConvention) -> None:
    if not isinstance(convention, FlowConvention):
        raise TypeError("convention must be the policy's FlowConvention")


def training_group_choices(horizon: int) -> tuple[int, ...]:
    """All equal-size partitions; H=64 gives (1,2,4,8,16,32,64)."""
    _positive_int(horizon, "horizon")
    return tuple(groups for groups in range(1, horizon + 1) if horizon % groups == 0)


def sample_grouped_time(
    sample_time: Callable[[int], Tensor],
    *,
    batch_size: int,
    horizon: int = 64,
    num_groups: int | None = None,
    convention: FlowConvention,
    group_generator: torch.Generator | None = None,
    time_bounds: tuple[float, float] | None = None,
    group_unit: int = 1,
    single_group_probability: float | None = None,
) -> Tensor:
    """Return [B,H] model times, preserving the original sampler and dtype.

    Choose one G per batch, independently of inference's execution horizon.
    Each example receives G original sampler draws: near actions are cleaner
    and distant actions noisier. For G=1, expand the untouched B original
    draws; do not add a new sampling distribution, cast, or transform.
    The native sampler owns its support: some trained policies have small
    excursions outside [0,1]. Set time_bounds only when the native contract
    requires them. This does not extend the normalized rolling/LSD schedule.
    An optional CPU generator separates group selection from native RNG draws.
    group_unit=1 uses Div(H); group_unit=C uses Div(H/C). The latter only
    changes instant grouping granularity, and is NOT the full original LSD
    sampler. None gives uniform group selection; p gives extra mass to G=1.
    """
    _check_convention(convention)
    _positive_int(batch_size, "batch_size")
    _positive_int(horizon, "horizon")
    _positive_int(group_unit, "group_unit")
    if horizon % group_unit:
        raise ValueError("group_unit must divide horizon")
    choices = training_group_choices(horizon // group_unit)
    if single_group_probability is not None:
        if isinstance(single_group_probability, bool) or not isinstance(single_group_probability, (int, float)):
            raise TypeError("single_group_probability must be a finite probability")
        if not math.isfinite(single_group_probability) or not 0 <= single_group_probability <= 1:
            raise ValueError("single_group_probability must lie in [0,1]")
    if num_groups is None:
        if len(choices) == 1:
            num_groups = 1
        elif single_group_probability is None:
            num_groups = choices[torch.randint(len(choices), (), device="cpu", generator=group_generator).item()]
        else:
            p = single_group_probability
            weights = torch.tensor([p] + [(1-p)/(len(choices)-1)]*(len(choices)-1), device="cpu")
            num_groups = choices[torch.multinomial(weights, 1, generator=group_generator).item()]
    _positive_int(num_groups, "num_groups")
    if num_groups not in choices:
        raise ValueError(f"num_groups must divide horizon/group_unit={horizon // group_unit}: {choices}")

    sampled = sample_time(batch_size * num_groups)
    if not isinstance(sampled, Tensor) or not sampled.is_floating_point():
        raise TypeError("sample_time must return a floating tensor")
    if sampled.shape != (batch_size * num_groups,):
        raise ValueError("sample_time must return shape [B*G]")
    if not torch.isfinite(sampled).all():
        raise ValueError("native time sampler must return finite values")
    if time_bounds is not None:
        low, high = time_bounds
        if not math.isfinite(low) or not math.isfinite(high) or low >= high:
            raise ValueError("time_bounds must be finite and strictly increasing")
        if not ((sampled >= low) & (sampled <= high)).all():
            raise ValueError("native time sample is outside its declared time_bounds")
    if num_groups == 1:
        return sampled[:, None].expand(batch_size, horizon)
    grouped = sampled.reshape(batch_size, num_groups)
    descending = convention is FlowConvention.NOISE_AT_ZERO
    grouped = grouped.sort(dim=1, descending=descending).values
    return grouped.repeat_interleave(horizon // num_groups, dim=1)


class RollingFlow:
    """One synchronous stream's FP32 latent cache, without model parameters.

    H is prediction horizon; C is the complete chunk consumed per call;
    K=H/C is the refinements per action. Cold start uses K forwards; a warm
    call uses one. Only the returned [B,C,D] tensor is clean.

    Supply noise from the original policy sampler using ``noise_horizon``.
    Reset for a new episode/task, changed batch-row identities, checkpoint,
    normalization, or action frame. All batch rows advance together. Partial
    consumption, action prefill and RTC are outside this first implementation.
    Fresh observation conditioning belongs in the caller's predict closure.
    """

    def __init__(
        self,
        *,
        action_dim: int,
        convention: FlowConvention,
        horizon: int = 64,
        execution_horizon: int = 16,
    ):
        _check_convention(convention)
        for value, name in (
            (horizon, "horizon"),
            (execution_horizon, "execution_horizon"),
            (action_dim, "action_dim"),
        ):
            _positive_int(value, name)
        if horizon % execution_horizon:
            raise ValueError("execution_horizon must divide horizon")
        self.horizon = horizon
        self.execution_horizon = execution_horizon
        self.action_dim = action_dim
        self.convention = convention
        self.steps = horizon // execution_horizon
        self._geometry = (horizon, execution_horizon, action_dim, convention, self.steps)
        self.reset()

    def _validate_configuration(self) -> None:
        # Geometry determines the meaning of every retained latent. Construct
        # a new state for geometry/direction changes, even after reset.
        _check_convention(self.convention)
        for value, name in (
            (self.horizon, "horizon"),
            (self.execution_horizon, "execution_horizon"),
            (self.action_dim, "action_dim"),
            (self.steps, "steps"),
        ):
            _positive_int(value, name)
        current = (self.horizon, self.execution_horizon, self.action_dim, self.convention, self.steps)
        if current != self._geometry:
            raise ValueError("rolling geometry/direction changed; construct a new state")

    def reset(self) -> None:
        self._cache: Tensor | None = None
        self._action_dim_mask: Tensor | None = None
        self.last_nfe = 0
        self._stream_ids = None
        self._context_key = None

    @property
    def noise_horizon(self) -> int:
        """Number of new noise tokens required by the next step."""
        return self.horizon if self._cache is None else self.execution_horizon

    @torch.no_grad()
    def step(
        self,
        predict: Callable[[Tensor, Tensor], Tensor],
        *,
        noise: Tensor,
        consumed_actions: int = 0,
        action_dim_mask: Tensor | None = None,
        stream_ids: tuple[str | int, ...] | None = None,
        context_key: str | None = None,
    ) -> Tensor:
        """Advance using external noise; commit state only after success.

        ``noise`` is [B,H,D] initially and [B,C,D] subsequently. ``predict``
        receives FP32 latent [B,H,D] and model time [B,H]; cast at the model
        boundary if required, and return velocity in the policy's convention.
        """
        self._validate_configuration()
        if not isinstance(noise, Tensor) or not noise.is_floating_point():
            raise TypeError("noise must be the policy's floating tensor sample")
        if noise.ndim != 3 or noise.shape[0] <= 0:
            raise ValueError("noise must have shape [B,noise_horizon,D] with B>0")
        batch, device = noise.shape[0], noise.device
        expected_noise = (batch, self.noise_horizon, self.action_dim)
        if noise.shape != expected_noise:
            raise ValueError(f"noise must have shape {expected_noise}")
        cold = self._cache is None
        expected_consumed = 0 if cold else self.execution_horizon
        if (
            isinstance(consumed_actions, bool)
            or not isinstance(consumed_actions, int)
            or consumed_actions != expected_consumed
        ):
            raise ValueError(
                f"expected consumed_actions={expected_consumed}; "
                "consume the complete output chunk or reset after interruption"
            )
        if stream_ids is None:
            if batch != 1:
                raise ValueError("batched rolling requires stable ordered stream_ids")
            stream_ids = (0,)
        if not isinstance(stream_ids, (tuple, list)) or len(stream_ids) != batch:
            raise ValueError("stream_ids must contain one stable ID per row")
        stream_ids = tuple(stream_ids)
        if any(isinstance(s, bool) or not isinstance(s, (str, int)) for s in stream_ids):
            raise ValueError("stream IDs must be strings or integers")
        if len(set(stream_ids)) != batch:
            raise ValueError("stream_ids must be unique")
        if not cold and stream_ids != self._stream_ids:
            raise ValueError("stream order/identity changed; reset or use per-stream states")
        if context_key is not None and not isinstance(context_key, str):
            raise TypeError("context_key must be a string or None")
        if not cold and context_key != self._context_key:
            raise ValueError("episode/model/normalization/frame key changed; reset first")
        shape = (batch, self.horizon, self.action_dim)
        if not cold and (self._cache.shape != shape or self._cache.device != device):
            raise ValueError("batch/device changed; reset the rolling state first")
        if action_dim_mask is None:
            mask = torch.ones(batch, self.action_dim, device=device, dtype=torch.bool)
        else:
            if not isinstance(action_dim_mask, Tensor) or action_dim_mask.dtype != torch.bool:
                raise TypeError("action_dim_mask must be a boolean tensor, not inferred from numbers")
            if action_dim_mask.shape != (batch, self.action_dim):
                raise ValueError("action_dim_mask must have shape [B,D]")
            mask = action_dim_mask.to(device=device, dtype=torch.bool)
        if not cold and not torch.equal(mask, self._action_dim_mask):
            raise ValueError("action_dim_mask changed; reset the rolling state first")

        new_noise = noise.float().clone()
        x = new_noise if cold else torch.cat((self._cache[:, self.execution_horizon :], new_noise), 1)
        invalid = ~mask[:, None, :]
        x = x.masked_fill(invalid, 0)
        if not torch.isfinite(x).all():
            raise ValueError("nonfinite active noise/latent")
        offset = torch.arange(self.steps, device=device, dtype=torch.float32)
        offset = offset.repeat_interleave(self.execution_horizon)[None]
        levels = range(self.steps, 0, -1) if cold else (1,)
        noise_at_one = self.convention is FlowConvention.NOISE_AT_ONE
        for level in levels:
            source = ((offset + level) / self.steps).clamp(max=1).expand(batch, -1)
            target = ((offset + level - 1) / self.steps).clamp(max=1).expand(batch, -1)
            if not noise_at_one:
                source, target = 1 - source, 1 - target
            velocity = predict(x, source)
            if not isinstance(velocity, Tensor) or not velocity.is_floating_point():
                raise TypeError("predict must return a floating velocity tensor")
            if velocity.shape != shape or velocity.device != device:
                raise ValueError("predict must preserve latent shape [B,H,D] and device")
            velocity = velocity.float().masked_fill(invalid, 0)
            dt = (target - source)[..., None]
            # Cold-start inactive blocks must stay untouched, even if their
            # unused predictions contain NaN/Inf (0 * NaN is still NaN).
            x = torch.where(dt != 0, x + dt * velocity, x)
            x = x.masked_fill(invalid, 0)
            if not torch.isfinite(x).all():
                raise FloatingPointError("nonfinite active Euler update; state not committed")

        self._stream_ids = stream_ids
        self._context_key = context_key
        self._cache = x.detach()
        self._action_dim_mask = mask.clone()
        self.last_nfe = self.steps if cold else 1
        return self._cache[:, : self.execution_horizon].clone()
