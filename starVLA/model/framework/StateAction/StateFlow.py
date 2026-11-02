"""State-only assembly of the existing GR00T FM and RollFlow heads.

No loss, sampler, or rolling integrator is reimplemented here. Inputs are
normalized by the dataset adapter; outputs remain in normalized action units.
"""
import numpy as np
import torch

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead
from starVLA.model.modules.action_model.GR00T_RollFlow_ActionHeader import RollFlowActionHead


@FRAMEWORK_REGISTRY.register("StateFlow")
class StateFlow(baseframework):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.variant = config.framework.variant
        if self.variant not in ("fm", "rolling_fm", "rolling_lsd"):
            raise ValueError("variant must be fm, rolling_fm, or rolling_lsd")
        cfg = config.framework.action_model
        self.repeated_diffusion_steps = int(getattr(cfg, "repeated_diffusion_steps", 1))
        if self.repeated_diffusion_steps < 1:
            raise ValueError("repeated_diffusion_steps must be positive")
        self.execution_horizon = int(cfg.execution_horizon)
        if not 0 < self.execution_horizon <= cfg.action_horizon:
            raise ValueError("execution_horizon must be in [1, action_horizon]")
        if cfg.diffusion_model_cfg["cross_attention_dim"] != cfg.state_dim:
            raise ValueError("State context requires cross_attention_dim == state_dim")
        if self.variant == "rolling_fm" and cfg.w_lsd != 0:
            raise ValueError("rolling_fm requires w_lsd=0")
        if self.variant == "rolling_lsd" and cfg.w_lsd <= 0:
            raise ValueError("rolling_lsd requires positive w_lsd")
        cls = FlowmatchingActionHead if self.variant == "fm" else RollFlowActionHead
        self.action_model = cls(config)

    @property
    def last_loss_stats(self):
        return self.action_model.last_loss_stats or {}

    def loss(self, state, actions, *, pad=None, training_step=0):
        if state.ndim == 2:
            state = state[:, None]
        # Same batch-major replication as QwenGR00T. Heads sample noise and
        # times AFTER replication, giving each copy independent draws (within
        # RollFlow's shared-per-batch choice of the number of time groups).
        repeats = self.repeated_diffusion_steps
        if repeats > 1:
            state = state.repeat(repeats, 1, 1)
            actions = actions.repeat(repeats, 1, 1)
            if pad is not None:
                pad = pad.repeat(repeats, 1)
        if self.variant == "fm":
            if pad is not None and bool(pad.any()):
                raise ValueError("Original FM has no padding loss mask; use complete windows")
            return self.action_model(state, actions, state)
        return self.action_model(
            state, actions, state, action_padding_mask=pad, training_step=training_step
        )

    @torch.no_grad()
    def sample_chunk(self, state, *, refinement_steps=4, velocity_mode="instant"):
        if state.ndim == 2:
            state = state[:, None]
        if self.variant == "fm":
            if velocity_mode != "instant":
                raise ValueError("Original FM supports instantaneous velocity only")
            previous = self.action_model.num_inference_timesteps
            try:
                self.action_model.num_inference_timesteps = int(refinement_steps)
                actions = self.action_model.predict_action(state, state)
                return actions[:, :self.execution_horizon]
            finally:
                self.action_model.num_inference_timesteps = previous
        return self.action_model.predict_action(
            state, state, refinement_steps=refinement_steps, velocity_mode=velocity_mode
        )

    def reset(self):
        if self.variant != "fm":
            self.action_model.reset()

    reset_cache = reset

    def _tensor(self, examples, key):
        parameter = next(self.parameters())
        return torch.as_tensor(np.asarray([e[key] for e in examples]),
                               device=parameter.device, dtype=parameter.dtype)

    def forward(self, examples, **kwargs):
        pad = None
        if any("action_padding_mask" in e for e in examples):
            if not all("action_padding_mask" in e for e in examples):
                raise ValueError("Padding mask must be present for every example")
            pad = self._tensor(examples, "action_padding_mask").bool()
        loss = self.loss(self._tensor(examples, "state"), self._tensor(examples, "action"),
                         pad=pad, training_step=int(kwargs.get("training_step", 0)))
        return {"action_loss": loss, "rollflow_metrics": self.last_loss_stats}

    @torch.inference_mode()
    def predict_action(self, examples, **kwargs):
        mode = kwargs.get("velocity_mode", "average" if self.variant == "rolling_lsd" else "instant")
        actions = self.sample_chunk(
            self._tensor(examples, "state"), velocity_mode=mode,
            refinement_steps=kwargs.get("refinement_steps", self.config.framework.action_model.inference_steps),
        )
        return {"normalized_actions": actions.float().cpu().numpy()}
