"""Small tensor operations for action-aligned time and clean-prefix adapters.

These operations do not select a probability path, a loss reduction, a time
sampler, an attention mask, or an asynchronous execution policy. Keep those in
the native policy. Validate numeric domains at its public boundary; the shape
operations themselves can be used inside a compiled action field.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def as_action_time(time: Tensor, *, batch_size: int, horizon: int) -> Tensor:
    """Expand [B] or retain [B,H], without averaging, casting, or quantizing.

    The returned view may share storage with the input. Do not modify it in
    place. Native time buckets are also supported after the native conversion.
    """
    if batch_size < 1 or horizon < 1:
        raise ValueError("batch_size and horizon must be positive")
    if not isinstance(time, Tensor):
        raise TypeError("time must be a tensor")
    if time.shape == (batch_size,):
        return time[:, None].expand(batch_size, horizon)
    if time.shape == (batch_size, horizon):
        return time
    raise ValueError(f"time must have shape {(batch_size,)} or {(batch_size, horizon)}")


def prefix_mask_from_lengths(lengths: Tensor, *, horizon: int) -> Tensor:
    """Validate integer [B] lengths before a compiled model call.

    Zero and H are valid tensor cases. A deployment may impose a stricter
    latency/overlap limit. Fractional lengths are never silently truncated.
    """
    if horizon < 1 or lengths.ndim != 1 or lengths.numel() == 0:
        raise ValueError("lengths must be nonempty [B] and horizon must be positive")
    if lengths.dtype not in (torch.int32, torch.int64):
        raise TypeError("prefix lengths must be int32 or int64")
    if bool(((lengths < 0) | (lengths > horizon)).any()):
        raise ValueError("prefix length must be between zero and horizon")
    return torch.arange(horizon, device=lengths.device)[None, :] < lengths[:, None]


def map_action_time_to_tokens(
    action_time: Tensor,
    action_index: Tensor,
    *,
    auxiliary_time: Tensor | None = None,
) -> Tensor:
    """Map [B,H] times to a packed sequence using explicit [S] ownership.

    Each index is an action index in [0,H), or -1 for an auxiliary token. A
    repeated index is allowed when one action owns multiple tokens. Auxiliary
    times must be explicitly supplied as [B,S]; there is no universal state,
    register, image, or history-token rule. This helper does not pool actions
    with unequal times into one token. Validate the static map outside compile.
    """
    if action_time.ndim != 2 or not all(action_time.shape):
        raise ValueError("action_time must have nonempty shape [B,H]")
    if action_index.ndim != 1 or action_index.numel() == 0:
        raise ValueError("action_index must have nonempty shape [S]")
    if action_index.dtype != torch.long or action_index.device != action_time.device:
        raise ValueError("action_index must be int64 on the action time device")
    if bool(((action_index < -1) | (action_index >= action_time.shape[1])).any()):
        raise ValueError("action_index contains an invalid action reference")
    needs_auxiliary = bool((action_index == -1).any())
    if auxiliary_time is None:
        if needs_auxiliary:
            raise ValueError("non-action tokens need an explicit time mapping")
        return action_time[:, action_index]
    shape = (action_time.shape[0], action_index.shape[0])
    if auxiliary_time.shape != shape:
        raise ValueError(f"auxiliary_time must have shape {shape}")
    if auxiliary_time.device != action_time.device or auxiliary_time.dtype != action_time.dtype:
        raise ValueError("auxiliary times must preserve native dtype and device")
    return torch.where(
        action_index[None, :] >= 0,
        action_time[:, action_index.clamp_min(0)],
        auxiliary_time,
    )


def apply_clean_prefix(
    x: Tensor,
    time: Tensor,
    *,
    prefix_values: Tensor,
    prefix_mask: Tensor,
    clean_time: float,
) -> tuple[Tensor, Tensor]:
    """Clamp known actions and their native time; retain scalar semantics.

    Input x is already constructed by the native noising path. Only a real
    action prefix is frozen; future clean targets must not be supplied as
    conditioning. Padding/dimension sanitation and loss masks are separate.
    """
    if x.ndim != 3 or prefix_values.shape != x.shape:
        raise ValueError("x and prefix_values must have matching [B,H,D] shapes")
    if prefix_values.device != x.device or prefix_values.dtype != x.dtype:
        raise ValueError("prefix values must match the native action dtype and device")
    if prefix_mask.shape != x.shape[:2] or prefix_mask.dtype != torch.bool:
        raise ValueError("prefix_mask must be bool [B,H]")
    if prefix_mask.device != x.device or time.device != x.device:
        raise ValueError("time, mask and actions must share a device")
    if not time.is_floating_point() or not math.isfinite(clean_time):
        raise ValueError("apply the prefix in native floating flow time, before bucket encoding")
    token_time = as_action_time(time, batch_size=x.shape[0], horizon=x.shape[1])
    conditioned = torch.where(prefix_mask[..., None], prefix_values, x)
    conditioned_time = torch.where(prefix_mask, token_time.new_full((), clean_time), token_time)
    return conditioned, conditioned_time


def prefix_safe_update(
    x: Tensor,
    velocity: Tensor,
    delta_time: Tensor,
    *,
    prefix_values: Tensor,
    prefix_mask: Tensor,
) -> Tensor:
    """One native Euler update with exact frozen prefix, including the last step.

    delta_time is [B,H]; set it from native source/target times. This operation
    supports a conditional sampler, not the persistent rolling/RTC queue
    transition. A prefix may be visible in attention while having zero motion
    and zero supervised loss. Keep it present in the model input.
    """
    if x.ndim != 3 or velocity.shape != x.shape or prefix_values.shape != x.shape:
        raise ValueError("x, velocity and prefix_values must have matching [B,H,D] shapes")
    if delta_time.shape != x.shape[:2] or prefix_mask.shape != x.shape[:2]:
        raise ValueError("delta_time and prefix_mask must have shape [B,H]")
    if prefix_mask.dtype != torch.bool:
        raise TypeError("prefix_mask must be boolean")
    if any(t.device != x.device for t in (velocity, delta_time, prefix_values, prefix_mask)):
        raise ValueError("all update tensors must share a device")
    if any(t.dtype != x.dtype for t in (velocity, prefix_values)):
        raise ValueError("actions, velocity and prefix values must share update dtype")
    moving = (delta_time != 0) & ~prefix_mask
    # where prevents an unused NaN velocity from changing a frozen position.
    updated = torch.where(moving[..., None], x + delta_time[..., None] * velocity, x)
    return torch.where(prefix_mask[..., None], prefix_values, updated)
