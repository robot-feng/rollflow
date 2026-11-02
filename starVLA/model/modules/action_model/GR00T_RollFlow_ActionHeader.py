# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified for starVLA RollFlow integration.

from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from starVLA.model.modules.action_model.rolling_meanflow_matching_head.action_encoder import (
    ActionEncoder,
)
from starVLA.model.modules.action_model.rolling_meanflow_matching_head.cross_attention_dit import DiT
from starVLA.model.modules.action_model.rolling_meanflow_matching_head.rolling_meanflow import (
    RollFlow,
    RollFlowConfig,
)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(F.relu(self.layer1(x)))


class RollFlowActionHeadConfig(PretrainedConfig):
    """Standalone schema for the RollFlow-specific action-head fields."""

    model_type = "starvla_rollflow_action_head"

    def __init__(
        self,
        action_model_type: str = "DiT-B",
        add_pos_embed: bool = True,
        diffusion_model_cfg: Optional[dict] = None,
        hidden_size: int = 1024,
        max_seq_len: int = 1024,
        action_dim: Optional[int] = None,
        state_dim: Optional[int] = None,
        action_horizon: Optional[int] = None,
        execution_horizon: int = 1,
        num_timestep_buckets: int = 1000,
        num_target_vision_tokens: int = 32,
        finite_difference_delta: float = 0.01,
        finite_difference_delta_jitter: float = 0.0,
        inference_steps: Optional[int] = None,
        velocity_mode: str = "average",
        p_k1: float = 0.7,
        p_fm: float = 0.3,
        fm_curriculum_steps: int = 5000,
        w_fm: float = 1.0,
        w_lsd: float = 0.5,
        lsd_loss_space: str = "velocity",
        lsd_warmup_steps: int = 0,
        clock_monitor_interval: int = 0,
        use_ot: bool = True,
        clip_velocity: float = 0.0,
        iterative_cold_start: bool = False,
        reset_cache_each_step: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.action_model_type = action_model_type
        self.add_pos_embed = add_pos_embed
        self.diffusion_model_cfg = {} if diffusion_model_cfg is None else diffusion_model_cfg
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.action_horizon = action_horizon
        self.execution_horizon = execution_horizon
        self.num_timestep_buckets = num_timestep_buckets
        self.num_target_vision_tokens = num_target_vision_tokens
        self.finite_difference_delta = finite_difference_delta
        self.finite_difference_delta_jitter = finite_difference_delta_jitter
        self.inference_steps = inference_steps
        self.velocity_mode = velocity_mode
        self.p_k1 = p_k1
        self.p_fm = p_fm
        self.fm_curriculum_steps = fm_curriculum_steps
        self.w_fm = w_fm
        self.w_lsd = w_lsd
        self.lsd_loss_space = lsd_loss_space
        self.lsd_warmup_steps = lsd_warmup_steps
        self.clock_monitor_interval = clock_monitor_interval
        self.use_ot = use_ot
        self.clip_velocity = clip_velocity
        self.iterative_cold_start = iterative_cold_start
        self.reset_cache_each_step = reset_cache_each_step


DiTConfig = {
    "DiT-B": {"input_embedding_dim": 768, "attention_head_dim": 64, "num_attention_heads": 12},
    "DiT-L": {"input_embedding_dim": 1536, "attention_head_dim": 48, "num_attention_heads": 32},
}


def _first_config_value(config, names, default=None):
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return value
    return default


class RollFlowActionHead(nn.Module):
    """GR00T DiT adapter for token-wise RollFlow ``(source, target)`` times."""

    def __init__(self, full_config):
        super().__init__()
        config = full_config.framework.action_model
        self.full_config = full_config
        self.config = config

        try:
            model_defaults = DiTConfig[config.action_model_type]
        except KeyError as error:
            raise ValueError(f"Unsupported action_model_type: {config.action_model_type}") from error

        diffusion_model_cfg = {**model_defaults, **dict(config.diffusion_model_cfg)}
        # Independent dropout masks would be amplified by the central difference.
        diffusion_model_cfg.update(dropout=0.0, final_dropout=False)

        self.model = DiT(**diffusion_model_cfg)
        self.input_embedding_dim = self.model.inner_dim
        self.action_horizon = int(config.action_horizon)
        self.action_dim = int(config.action_dim)
        self.execution_horizon = int(
            _first_config_value(
                config,
                ("execution_horizon", "chunk_size", "action_chunk_size"),
                1,
            )
        )
        self.num_timestep_buckets = float(_first_config_value(config, ("num_timestep_buckets",), 1000))

        hidden_size = int(config.hidden_size)
        state_dim = _first_config_value(config, ("state_dim",), 0)
        self.state_encoder = (
            MLP(int(state_dim), hidden_size, self.input_embedding_dim) if state_dim else None
        )
        self.action_encoder = ActionEncoder(self.action_dim, self.input_embedding_dim)
        self.action_decoder = MLP(self.model.config.output_dim, hidden_size, self.action_dim)

        self.future_tokens = nn.Embedding(config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        trainer_config = getattr(full_config, "trainer", None)
        total_train_steps = getattr(trainer_config, "max_train_steps", None)
        self.rollflow = RollFlow(
            RollFlowConfig(
                horizon=self.action_horizon,
                action_dim=self.action_dim,
                chunk_size=self.execution_horizon,
                finite_difference_delta=float(
                    _first_config_value(config, ("finite_difference_delta",), 0.01)
                ),
                inference_steps=_first_config_value(
                    config,
                    ("inference_steps", "num_inference_timesteps"),
                ),
                finite_difference_delta_jitter=float(
                    _first_config_value(config, ("finite_difference_delta_jitter",), 0.0)
                ),
                velocity_mode=str(_first_config_value(config, ("velocity_mode",), "average")),
                p_k1=float(_first_config_value(config, ("p_k1",), 0.7)),
                p_fm=float(_first_config_value(config, ("p_fm",), 0.3)),
                fm_curriculum_steps=int(
                    _first_config_value(
                        config,
                        # ``fm_only_steps`` is retained for old experiment YAMLs.
                        ("fm_only_steps", "fm_curriculum_steps"),
                        5000,
                    )
                ),
                total_train_steps=(
                    None if total_train_steps is None else int(total_train_steps)
                ),
                w_fm=float(_first_config_value(config, ("w_fm",), 1.0)),
                w_lsd=float(_first_config_value(config, ("w_lsd",), 0.5)),
                lsd_loss_space=str(_first_config_value(config, ("lsd_loss_space",), "velocity")),
                lsd_warmup_steps=int(_first_config_value(config, ("lsd_warmup_steps",), 0)),
                use_ot=bool(_first_config_value(config, ("use_ot",), True)),
                clip_velocity=float(_first_config_value(config, ("clip_velocity",), 0.0)),
                iterative_cold_start=bool(
                    _first_config_value(config, ("iterative_cold_start",), False)
                ),
                reset_cache_each_step=bool(
                    _first_config_value(config, ("reset_cache_each_step",), False)
                ),
            )
        )
        self.last_loss_stats = None
        self.clock_monitor_interval = int(_first_config_value(config, ("clock_monitor_interval",), 0))
        if self.clock_monitor_interval < 0:
            raise ValueError("clock_monitor_interval must be non-negative")

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def forward(
        self,
        vl_embs: torch.Tensor,
        actions: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        encoder_attention_mask=None,
        action_padding_mask: Optional[torch.Tensor] = None,
        training_step: int = 0,
    ) -> torch.Tensor:
        """Return the scalar RollFlow objective for actions shaped ``[B,H,A]``."""
        if action_padding_mask is not None:
            action_padding_mask = action_padding_mask.to(device=actions.device, dtype=torch.bool)
        state_features = self._encode_state(state)
        loss, self.last_loss_stats = self.rollflow.loss(
            self._predict_velocity,
            actions,
            step=training_step,
            context=vl_embs,
            pad=action_padding_mask,
            action_padding_mask=action_padding_mask,
            state_features=state_features,
            encoder_attention_mask=encoder_attention_mask,
        )
        # The trainer logs the optimizer step about to complete as step+1.
        # All ranks follow the same path so sharded modules can communicate.
        if self.clock_monitor_interval and (training_step + 1) % self.clock_monitor_interval == 0:
            self.last_loss_stats.update(self._clock_diagnostics())
        return loss

    @torch.no_grad()
    def _clock_diagnostics(self) -> dict[str, float]:
        """Measure time-conditioned modulation on a fixed diagonal clock grid."""
        device_type = self.device.type
        autocast_context = (
            torch.autocast(device_type=device_type, dtype=torch.get_autocast_dtype(device_type),
                           cache_enabled=False)
            if torch.is_autocast_enabled(device_type) else nullcontext()
        )
        # Do not leave detached casts in an enclosing autocast context: callers
        # may perform another gradient-bearing forward before leaving it.
        with autocast_context:
            times = torch.linspace(0, self.num_timestep_buckets, 33, device=self.device)
            temb = self.model.timestep_encoder(times)
            if not self.model.config.skip_zero_interval_embedding:
                temb = temb + self.model.interval_timestep_encoder(torch.zeros_like(times))
            scales, shifts = [], []
            for block in self.model.transformer_blocks:
                if hasattr(block.norm1, "linear"):
                    scale, shift = block.norm1.linear(block.norm1.silu(temb)).chunk(2, dim=-1)
                    scales.append(scale.float().abs().max())
                    shifts.append(shift.float().abs().max())
            output_shift, output_scale = self.model.proj_out_1(F.silu(temb)).chunk(2, dim=-1)
            zero = temb.new_zeros((), dtype=torch.float32)
            return {
                "clock_embedding_rms": float(temb.float().square().mean().sqrt().cpu()),
                "clock_adaln_scale_max": float((torch.stack(scales).max() if scales else zero).cpu()),
                "clock_adaln_shift_max": float((torch.stack(shifts).max() if shifts else zero).cpu()),
                "clock_output_scale_max": float(output_scale.float().abs().max().cpu()),
                "clock_output_shift_max": float(output_shift.float().abs().max().cpu()),
            }

    @torch.no_grad()
    def predict_action(
        self,
        vl_embs: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        encoder_attention_mask=None,
        refinement_steps: Optional[int] = None,
        velocity_mode: Optional[str] = None,
    ) -> torch.Tensor:
        """Return the next executable chunk shaped ``[B,C,A]`` and update the cache."""
        return self.rollflow.step(
            self._predict_velocity,
            batch=vl_embs.shape[0],
            context=vl_embs,
            device=vl_embs.device,
            dtype=self.dtype,
            refinement_steps=refinement_steps,
            velocity_mode=velocity_mode,
            state_features=self._encode_state(state),
            encoder_attention_mask=encoder_attention_mask,
        )

    def reset(self) -> None:
        """Clear rolling inference state at every episode/session boundary."""
        self.rollflow.reset()

    reset_cache = reset

    @property
    def cache_info(self):
        return self.rollflow.cache_info

    def _encode_state(self, state: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if state is None:
            return None
        if self.state_encoder is None:
            raise ValueError("state was provided but state_dim is disabled")
        state = state.to(device=self.device, dtype=self.dtype)
        if state.ndim == 2:
            state = state.unsqueeze(1)
        if state.ndim != 3:
            raise ValueError(f"state must have shape [B,S,D], got {tuple(state.shape)}")
        return self.state_encoder(state)

    def _predict_velocity(
        self,
        actions: torch.Tensor,
        source_time: torch.Tensor,
        target_time: torch.Tensor,
        context: torch.Tensor,
        *,
        state_features: Optional[torch.Tensor] = None,
        encoder_attention_mask=None,
        action_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, horizon, _ = actions.shape
        if action_padding_mask is not None and action_padding_mask.shape != (batch, horizon):
            raise ValueError(
                f"action_padding_mask must have shape {(batch, horizon)}, "
                f"got {tuple(action_padding_mask.shape)}"
            )
        source_time = self._action_times(source_time, batch, horizon)
        target_time = self._action_times(target_time, batch, horizon)

        actions = actions.to(device=self.device, dtype=self.dtype)
        context = self._cast_context(context)
        source_scaled = source_time * self.num_timestep_buckets
        target_scaled = target_time * self.num_timestep_buckets

        action_features = self.action_encoder(actions, source_scaled)
        if self.config.add_pos_embed:
            if horizon > self.position_embedding.num_embeddings:
                raise ValueError(f"action horizon {horizon} exceeds max_seq_len")
            positions = torch.arange(horizon, device=actions.device)
            action_features = action_features + self.position_embedding(positions)[None]

        prefix = [self.future_tokens.weight[None].expand(batch, -1, -1)]
        if state_features is not None:
            prefix.insert(0, state_features.to(device=self.device, dtype=self.dtype))
        prefix_length = sum(features.shape[1] for features in prefix)
        hidden_states = torch.cat((*prefix, action_features), dim=1)

        self_attention_mask = None
        if action_padding_mask is not None:
            action_valid = ~action_padding_mask.to(device=hidden_states.device, dtype=torch.bool)
            prefix_valid = torch.ones(
                batch, prefix_length, device=hidden_states.device, dtype=torch.bool
            )
            key_valid = torch.cat((prefix_valid, action_valid), dim=1)
            self_attention_mask = torch.zeros(
                batch,
                1,
                prefix_length + horizon,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            self_attention_mask.masked_fill_(
                ~key_valid[:, None, :], torch.finfo(hidden_states.dtype).min
            )

        source_scaled = self._prepend_prefix_time(source_scaled, prefix_length)
        target_scaled = self._prepend_prefix_time(target_scaled, prefix_length)
        model_output = self.model(
            hidden_states=hidden_states,
            encoder_hidden_states=context,
            encoder_attention_mask=encoder_attention_mask,
            timestep=target_scaled,
            start_timestep=source_scaled,
            return_all_hidden_states=False,
            self_attention_mask=self_attention_mask,
        )
        return self.action_decoder(model_output[:, -horizon:]).float()

    def _action_times(self, time: torch.Tensor, batch: int, horizon: int) -> torch.Tensor:
        if time.ndim == 3 and time.shape[-1] == 1:
            time = time.squeeze(-1)
        if time.ndim == 1 and time.shape[0] == batch:
            time = time[:, None].expand(-1, horizon)
        if time.shape != (batch, horizon):
            raise ValueError(f"time must have shape [B,H,1] or [B,H], got {tuple(time.shape)}")
        return time.to(device=self.device, dtype=torch.float32)

    @staticmethod
    def _prepend_prefix_time(time: torch.Tensor, prefix_length: int) -> torch.Tensor:
        prefix_time = time[:, :1].expand(-1, prefix_length)
        return torch.cat((prefix_time, time), dim=1)

    def _cast_context(self, context):
        if isinstance(context, (list, tuple)):
            return type(context)(state.to(device=self.device, dtype=self.dtype) for state in context)
        return context.to(device=self.device, dtype=self.dtype)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype


# Keep the names used by existing GR00T framework imports.
FlowmatchingActionHeadConfig = RollFlowActionHeadConfig
FlowmatchingActionHead = RollFlowActionHead


def get_action_model(config=None):
    return RollFlowActionHead(full_config=config)
