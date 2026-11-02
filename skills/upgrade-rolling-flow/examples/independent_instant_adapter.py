"""Minimal independent registration pattern; DOES NOT patch the upstream policy.

This is an interface example. Fill native_velocity and original noise sampling
from the exact host policy. It is NOT an automatic registry implementation.
"""

from __future__ import annotations
from collections.abc import Callable
from typing import Any
from rolling_flow_torch import FlowConvention, RollingFlow
from torch import Tensor


class RollFlowInstantAdapter:
    def __init__(
        self,
        native_velocity: Callable[[Tensor, Tensor, Any], Tensor],
        *,
        action_dim: int,
        convention: FlowConvention,
        horizon: int = 64,
        execution_horizon: int = 16,
    ) -> None:
        self.native_velocity = native_velocity
        self.state = RollingFlow(
            action_dim=action_dim,
            convention=convention,
            horizon=horizon,
            execution_horizon=execution_horizon,
        )

    def next_chunk(
        self,
        observation: Any,
        native_noise: Tensor,
        *,
        consumed_actions: int,
        stream_ids: tuple[str | int, ...],
        session_key: str,
        action_dim_mask: Tensor | None = None,
    ) -> Tensor:
        # Build observation conditioning once per request in a real VLA. Recompute
        # prefix for each new observation; never reuse suffix hidden/KV as latent.
        # Caller owns execution confirmation, normalizer, physical action mapping,
        # preprocessing, and original policy registry wiring.
        def predict(latent: Tensor, native_time: Tensor) -> Tensor:
            return self.native_velocity(latent, native_time, observation)

        return self.state.step(
            predict,
            noise=native_noise,
            consumed_actions=consumed_actions,
            action_dim_mask=action_dim_mask,
            stream_ids=stream_ids,
            context_key=session_key,
        )

    def reset(self) -> None:
        self.state.reset()


# Example naming only: do not silently overwrite upstream registry entries.
SUGGESTED_POLICY_NAMES = ('original_policy', 'original_policy_rollflow_instant', 'original_policy_rollflow_lsd')
