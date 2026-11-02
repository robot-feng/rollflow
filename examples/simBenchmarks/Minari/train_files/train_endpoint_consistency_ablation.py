"""Experimental RollFlow objective ablations for Minari.

The teacher is the current model evaluated under no_grad; no separate model is
trained. This process-only intervention retains production FM and LSD losses.
It additionally matches an average-velocity endpoint to an Euler-integrated
instantaneous endpoint on deployment-shaped clocks. It is not the production
RollFlow loss. The optional off-diagonal FM term directly supervises
u(x_s,s,t) against the linear-path velocity on staircase-sampled intervals; it
tests whether sparse/indirect LSD supervision is the cause of weak average-mode
rollouts.
"""
from contextlib import nullcontext
from dataclasses import replace
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from starVLA.model.framework.StateAction.StateFlow import StateFlow
from starVLA.model.modules.action_model.rolling_meanflow_matching_head import rolling_meanflow as rollflow_core
from starVLA.model.modules.action_model.rolling_meanflow_matching_head.rolling_meanflow import (
    StaircaseTimeSampler, lerp_path, zero_padded,
)

_original_training_sampler = StaircaseTimeSampler.sample_training
_original_loss=StateFlow.loss
_endpoint_weight=1.0
_front_weight=1.0
_endpoint_teacher_steps=8
_teacher_weights=None
_frozen_teacher=None
_lsd_normalization='active'
_lsd_robust_threshold=0.0
_endpoint_source='path'
_clock_sampling='mixed'
_execution_weight=0.0
_execution_k_probabilities=(0.25,0.25,0.25,0.25)
_offdiagonal_fm_weight=0.0
_previous_state=None
_previous_valid=None
_original_central_difference=rollflow_core.central_difference_lsd
_original_rollflow_loss=rollflow_core.RollFlow.loss
_original_masked_mse=rollflow_core.masked_mse


def resolve_endpoint_weight(requested, previous_metadata):
    """Keep the same endpoint objective across ordinary checkpoint resumes."""
    previous = None if previous_metadata is None else float(previous_metadata['weight'])
    weight = previous if requested is None and previous is not None else (1.0 if requested is None else requested)
    if not math.isfinite(weight) or weight < 0:
        raise ValueError('endpoint weight must be finite and nonnegative')
    if previous is not None and weight != previous:
        raise ValueError('Cannot change endpoint weight while resuming this intervention')
    return weight


def resolve_teacher_mode(requested, previous_metadata):
    previous = None if previous_metadata is None else previous_metadata.get('teacher_mode', 'self')
    mode = previous if requested is None and previous is not None else (requested or 'self')
    if mode not in ('self', 'frozen'):
        raise ValueError('teacher mode must be self or frozen')
    if previous is not None and mode != previous:
        raise ValueError('Cannot change endpoint teacher while resuming this intervention')
    return mode


def resolve_front_weight(requested, previous_metadata):
    previous = None if previous_metadata is None else float(previous_metadata.get('front_weight', 1.0))
    weight = previous if requested is None and previous is not None else (1.0 if requested is None else requested)
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError('front weight must be finite and positive')
    if previous is not None and weight != previous:
        raise ValueError('Cannot change front weight while resuming this intervention')
    return weight


def resolve_lsd_normalization(requested, previous_metadata):
    previous = None if previous_metadata is None else previous_metadata.get('lsd_normalization', 'active')
    mode = previous if requested is None and previous is not None else (requested or 'active')
    if mode not in ('active', 'valid', 'ramp10'):
        raise ValueError('LSD normalization must be active, valid or ramp10')
    if previous is not None and mode != previous:
        raise ValueError('Cannot change LSD normalization while resuming this intervention')
    return mode


def valid_token_fraction(active, pad):
    valid = torch.ones_like(active, dtype=torch.bool) if pad is None else ~pad.bool().unsqueeze(-1)
    return float((active.bool() & valid).sum()) / max(int(valid.sum()), 1)


def lsd_scale(active_fraction, mode):
    if mode == 'valid':
        return active_fraction
    if mode == 'ramp10':
        return min(1., active_fraction / .1)
    raise ValueError('LSD scale is only used for valid or ramp10 normalization')


def resolve_robust_threshold(requested, previous_metadata):
    previous = None if previous_metadata is None else float(previous_metadata.get('lsd_robust_threshold', 0.0))
    threshold = previous if requested is None and previous is not None else (0.0 if requested is None else requested)
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError('LSD robust threshold must be finite and nonnegative')
    if previous is not None and threshold != previous:
        raise ValueError('Cannot change LSD robust threshold while resuming this intervention')
    return threshold


def resolve_endpoint_source(requested, previous_metadata):
    previous = None if previous_metadata is None else previous_metadata.get('endpoint_source', 'path')
    source = previous if requested is None and previous is not None else (requested or 'path')
    if source not in ('path', 'cache'):
        raise ValueError('Endpoint source must be path or cache')
    if previous is not None and source != previous:
        raise ValueError('Cannot change endpoint source while resuming this intervention')
    return source


def resolve_clock_sampling(requested, previous_metadata):
    previous = None if previous_metadata is None else previous_metadata.get('clock_sampling', 'mixed')
    mode = previous if requested is None and previous is not None else (requested or 'mixed')
    if mode not in ('original', 'mixed', 'edge'):
        raise ValueError('Clock sampling must be original, mixed or edge')
    if previous is not None and mode != previous:
        raise ValueError('Cannot change clock sampling while resuming this intervention')
    return mode


def training_clock_sampler(clock_module, mode):
    """Select the sampler independently from the endpoint-loss clock source.

    ``original`` preserves the checkpoint's original K distribution and time
    sampler. ``mixed`` and ``edge`` use the process-local deployment-clock
    intervention for the training loss.
    """
    if mode == 'original':
        return _original_training_sampler
    if mode == 'edge':
        return edge_clock_sample(clock_module.sample_training)
    if mode == 'mixed':
        return clock_module.sample_training
    raise ValueError('Clock sampling must be original, mixed or edge')


def resolve_execution_weight(requested, previous_metadata):
    previous = None if previous_metadata is None else float(previous_metadata.get('execution_weight', 0.0))
    weight = previous if requested is None and previous is not None else (0.0 if requested is None else requested)
    if not math.isfinite(weight) or weight < 0:
        raise ValueError('Execution weight must be finite and nonnegative')
    if previous is not None and weight != previous:
        raise ValueError('Cannot change execution weight while resuming this intervention')
    return weight


def resolve_execution_k_probabilities(requested, previous_metadata):
    """Resolve endpoint-supervision K probabilities in the order 1,2,4,8.

    The legacy experiment sampled uniformly. Keep that behavior unless a
    continuation explicitly opts into a low-step-focused distribution.
    """
    key='execution_k_probabilities'
    if requested is None:
        values=previous_metadata.get(key,(0.25,0.25,0.25,0.25)) if previous_metadata else (0.25,0.25,0.25,0.25)
    elif isinstance(requested,str):
        values=tuple(float(part) for part in requested.split(','))
    else:
        values=tuple(float(value) for value in requested)
    values=tuple(float(value) for value in values)
    if len(values)!=4 or any(not math.isfinite(value) or value<0 for value in values):
        raise ValueError('execution K probabilities must be four finite nonnegative values for K=1,2,4,8')
    if not math.isclose(sum(values),1.0,rel_tol=0.0,abs_tol=1e-6):
        raise ValueError('execution K probabilities must sum to 1')
    if previous_metadata and key in previous_metadata:
        previous=tuple(float(value) for value in previous_metadata[key])
        if values!=previous:
            raise ValueError('Cannot change execution K probabilities while resuming this intervention')
    return values


def sample_execution_refinement_steps(cfg, device, probabilities):
    """Sample a valid deployment K with probabilities ordered as 1,2,4,8."""
    levels=(1,2,4,8)
    valid_indices=[i for i,k in enumerate(levels) if cfg.valid_steps(k)]
    probs=torch.tensor([probabilities[i] for i in valid_indices],device=device,dtype=torch.float32)
    if not bool((probs>0).any()):
        raise ValueError('execution K probabilities assign zero mass to every valid refinement level')
    probs=probs/probs.sum()
    return levels[valid_indices[int(torch.multinomial(probs,1).item())]]


def resolve_offdiagonal_fm_weight(requested, previous_metadata):
    """Resolve the direct mean-velocity FM probe without changing old runs."""
    has_previous = previous_metadata is not None and 'offdiagonal_fm_weight' in previous_metadata
    previous = float(previous_metadata['offdiagonal_fm_weight']) if has_previous else None
    weight = previous if requested is None and has_previous else (0.0 if requested is None else requested)
    if not math.isfinite(weight) or weight < 0:
        raise ValueError('off-diagonal FM weight must be finite and nonnegative')
    if has_previous and weight != previous:
        raise ValueError('Cannot change off-diagonal FM weight while resuming this intervention')
    return weight


def offdiagonal_flow_matching_loss(head, context, target, pad=None):
    """Directly supervise u(x_s,s,t) on the linear action path.

    This is a training-only diagnostic. It adds a second, non-diagonal FM
    prediction to the existing diagonal FM + LSD objective; production RollFlow
    remains untouched. A staircase K is sampled using the normal K distribution
    and all samples use s<t. Near-zero intervals are excluded from this term.
    """
    target = target.float()
    noise = torch.randn_like(target)
    noise = zero_padded(noise, pad)
    target = zero_padded(target, pad)
    times = head.rollflow.time.sample_training(
        len(target), device=target.device, dtype=torch.float32, p_fm=0.0
    )
    interval = times.t - times.s
    valid = interval.squeeze(-1) > 1e-3
    if pad is not None:
        valid &= ~pad.bool()
    source = zero_padded(lerp_path(noise, target, times.s), pad)
    state_features = head._encode_state(context)
    prediction = head._predict_velocity(
        source,
        times.s,
        times.t,
        context,
        state_features=state_features,
        action_padding_mask=pad,
    )
    target_velocity = target - noise
    loss = rollflow_core.masked_mse(prediction - target_velocity, pad=pad, valid=valid)
    valid_fraction = valid.float().mean()
    mean_interval = (interval.squeeze(-1) * valid).sum() / valid.sum().clamp_min(1)
    return loss, valid_fraction, mean_interval


def edge_clock_sample(original):
    """Move half the off-diagonal rows to an edge-reaching staircase.

    This wrapper preserves all diagonal FM rows and consumes no extra random
    numbers. The FM term for changed rows uses their new endpoint time as well.
    """
    def sample(self, batch, *, device, dtype=None, generator=None, p_fm=None, num_time_groups=None):
        times = original(self, batch, device=device, dtype=dtype, generator=generator,
                         p_fm=p_fm, num_time_groups=num_time_groups)
        selected=(torch.arange(batch,device=device)%2==0)[:,None,None] & ~times.fm_mask
        if not bool(selected.any()):
            return times
        max_delta=self.cfg.finite_difference_delta*(1+self.cfg.finite_difference_delta_jitter)
        upper=times.t.new_full((batch,),1.-max_delta-1e-6)
        lower=times.t.new_zeros((batch,))
        s_edge,t_edge,_=self._build(lower,upper,times.num_time_groups)
        s=torch.where(selected,s_edge,times.s)
        t=torch.where(selected,t_edge,times.t)
        ratio=torch.where(t>0,s/t.clamp_min(1e-8),torch.ones_like(t)).mean(dim=1,keepdim=True)
        return replace(times,s=s,t=t,active=rollflow_core._lsd_mask(s,t,self.cfg.finite_difference_delta),
                       ratio=torch.where(selected,ratio,times.ratio))
    return sample


def previous_conditions(data, stats, horizon, batch_size, rng, device, chunk, complete=True):
    """Reproduce sample's anchors without consuming its RNG; fetch prior observations."""
    sampler_rng = copy.deepcopy(rng)
    lengths = np.array([max(0, len(ep[2])-horizon+1) if complete else len(ep[2]) for ep in data])
    ends = lengths.cumsum()
    if not len(ends) or int(ends[-1]) == 0:
        raise ValueError('No action windows available')
    anchors = sampler_rng.integers(int(ends[-1]), size=batch_size)
    groups = horizon // chunk
    if groups < 1 or horizon % chunk:
        raise ValueError('chunk must divide horizon')
    states, valid = [], []
    for anchor in anchors:
        idx = int(np.searchsorted(ends, anchor, side='right'))
        start = int(anchor - (ends[idx-1] if idx else 0))
        obs = data[idx][1]
        states.append(np.stack([(obs[max(start-i*chunk,0)]-stats['state_mean'])/stats['state_std']
                                for i in range(groups,0,-1)]))
        valid.append([start >= i*chunk for i in range(1,groups+1)])
    return (torch.as_tensor(np.stack(states), device=device),
            torch.as_tensor(valid, device=device, dtype=torch.bool))


def cache_aware_sample(original, chunk):
    def sample(data, stats, horizon, batch_size, rng, device, complete=True):
        global _previous_state, _previous_valid
        _previous_state, _previous_valid = previous_conditions(
            data, stats, horizon, batch_size, rng, device, chunk, complete)
        return original(data, stats, horizon, batch_size, rng, device, complete=complete)
    return sample


@torch.no_grad()
def warm_cache_source(head, previous, k):
    """Build cache with K preceding observations and shift to the current chunk."""
    if previous.ndim != 3 or previous.shape[1] != k:
        raise ValueError('previous must contain K ordered observations per sample')
    head.rollflow.reset()
    try:
        for i in range(k):
            context=previous[:,i:i+1]
            head.rollflow.step(head._predict_velocity, batch=previous.shape[0], context=context,
                device=previous.device, dtype=head.dtype, refinement_steps=k, velocity_mode='average',
                state_features=head._encode_state(context))
        source, cold = head.rollflow._rolling_input(previous.shape[0], previous.device, head.dtype, k)
        if cold:
            raise RuntimeError('Failed to construct warm rolling input')
        return source.float()
    finally:
        head.rollflow.reset()


def pseudo_huber_lsd_displacement(error, *, pad, valid, delta, threshold):
    """Pseudo-Huber on velocity residual, returned in displacement MSE units."""
    mask = valid.bool()
    if pad is not None:
        mask = mask & ~pad.bool().unsqueeze(-1)
    safe_error = torch.where(mask, error.float(), torch.zeros_like(error).float())
    square = (safe_error / (2 * delta)).square()
    robust = 2 * square / (torch.sqrt(1 + square / threshold ** 2) + 1)
    return rollflow_core.masked_mean(robust, mask) * (2 * delta) ** 2


def robust_central(model, x0, x1, s, t, **kwargs):
    """Apply an experimental robust residual solely inside the LSD calculation."""
    if _lsd_normalization in ('valid', 'ramp10'):
        active = kwargs.get('active')
        if active is None:
            active = rollflow_core._lsd_mask(s, t, kwargs['delta'])
        kwargs['w_lsd'] = kwargs.get('w_lsd', .5) * lsd_scale(
            valid_token_fraction(active, kwargs.get('pad')), _lsd_normalization)
    delta = kwargs['delta']
    def masked_robust(error, *, pad=None, valid=None):
        if valid is None:
            return _original_masked_mse(error, pad=pad)
        return pseudo_huber_lsd_displacement(error, pad=pad, valid=valid,
                                              delta=delta, threshold=_lsd_robust_threshold)
    rollflow_core.masked_mse = masked_robust
    try:
        return _original_central_difference(model, x0, x1, s, t, **kwargs)
    finally:
        rollflow_core.masked_mse = _original_masked_mse


def valid_normalized_central(model, x0, x1, s, t, **kwargs):
    """Scale conditional LSD by the configured active-token fraction rule."""
    active=kwargs.get('active')
    if active is None:
        active=rollflow_core._lsd_mask(s,t,kwargs['delta'])
    fraction=valid_token_fraction(active,kwargs.get('pad'))
    kwargs['w_lsd'] = kwargs.get('w_lsd', .5) * lsd_scale(fraction,_lsd_normalization)
    return _original_central_difference(model,x0,x1,s,t,**kwargs)


def valid_normalized_rollflow_loss(self,*args,**kwargs):
    loss,stats=_original_rollflow_loss(self,*args,**kwargs)
    # Core reports the conditional LSD mean; record the actual effective weight.
    stats['lsd_weight'] *= lsd_scale(stats['active_lsd_frac'],_lsd_normalization)
    stats['weighted_lsd_loss'] = stats['lsd_weight'] * stats['lsd_loss']
    return loss,stats


def endpoint_mse(error, valid, chunk_size, front_weight):
    """Mean displacement MSE, optionally emphasizing the executable chunk."""
    if front_weight == 1.0:
        return error.masked_select(valid).mean() if bool(valid.any()) else error.masked_fill(~valid,0).sum()
    weights=torch.ones_like(error)
    weights[:,:chunk_size] *= front_weight
    weights=weights.masked_fill(~valid,0)
    safe_error=error.masked_fill(~valid,0)
    return (safe_error*weights).sum()/weights.sum().clamp_min(1.)


def masked_squared_error(difference, valid):
    """Mask before squaring so invalid NaNs cannot poison the backward pass."""
    safe = torch.where(valid, difference.float(), torch.zeros_like(difference).float())
    return safe.square()


def teacher_head_for(head):
    """Load a frozen snapshot lazily, or use the evolving student as teacher."""
    global _frozen_teacher, _teacher_weights
    if _frozen_teacher is not None:
        return _frozen_teacher
    if _teacher_weights is None:
        return head
    _frozen_teacher=copy.deepcopy(head)
    _frozen_teacher.load_state_dict(_teacher_weights,strict=True)
    _frozen_teacher.requires_grad_(False).eval()
    _teacher_weights=None
    return _frozen_teacher


@torch.no_grad()
def instantaneous_endpoint(predict, x, s, t, steps=8):
    """Integrate the detached instantaneous field over each token's interval."""
    if steps < 1:
        raise ValueError('steps must be positive')
    result=x.detach().float().clone()
    delta=(t-s).float()/steps
    for i in range(steps):
        now=s+(t-s)*(i/steps)
        result=result+delta*predict(result,now,now).float()
    return result


def augmented_loss(self,state,actions,*,pad=None,training_step=0):
    loss=_original_loss(self,state,actions,pad=pad,training_step=training_step)
    cfg=self.action_model.rollflow.cfg
    ramp=min(max((training_step-cfg.fm_curriculum_steps)/1000,0.),1.)
    if ramp and _offdiagonal_fm_weight:
        if self.variant!='rolling_lsd':
            raise ValueError('Off-diagonal FM ablation requires rolling_lsd')
        context=state[:,None] if state.ndim==2 else state
        context=context.repeat(self.repeated_diffusion_steps,1,1)
        target=actions.repeat(self.repeated_diffusion_steps,1,1).float()
        repeated_pad=pad.repeat(self.repeated_diffusion_steps,1) if pad is not None else None
        offdiag_loss, offdiag_frac, mean_interval=offdiagonal_flow_matching_loss(
            self.action_model,context,target,repeated_pad
        )
        effective_weight=ramp*_offdiagonal_fm_weight
        loss=loss+effective_weight*offdiag_loss
        self.action_model.last_loss_stats['offdiagonal_fm_loss']=float(offdiag_loss.detach())
        self.action_model.last_loss_stats['offdiagonal_fm_weight']=effective_weight
        self.action_model.last_loss_stats['offdiagonal_fm_valid_frac']=float(offdiag_frac)
        self.action_model.last_loss_stats['offdiagonal_fm_interval']=float(mean_interval)
        self.action_model.last_loss_stats['loss']=float(loss.detach())
    if ramp==0 or (_endpoint_weight==0 and _execution_weight==0):
        return loss
    if self.variant!='rolling_lsd':
        raise ValueError('Endpoint ablation requires rolling_lsd')
    if state.ndim==2:
        state=state[:,None]
    context=state.repeat(self.repeated_diffusion_steps,1,1)
    target=actions.repeat(self.repeated_diffusion_steps,1,1).float()
    pad=pad.repeat(self.repeated_diffusion_steps,1) if pad is not None else None
    head=self.action_model
    k=sample_execution_refinement_steps(cfg,target.device,_execution_k_probabilities)
    clocks=head.rollflow.time.deployment(len(target),refinement_steps=k,cold=False,device=target.device)
    s,t=clocks.s,clocks.t
    if _endpoint_source == 'cache':
        if _previous_state is None or _previous_valid is None or len(_previous_state) != len(state):
            raise RuntimeError('Cache endpoint source requires matching previous observations')
        previous = _previous_state[:,-k:].repeat(self.repeated_diffusion_steps,1,1)
        device_type=target.device.type
        cache_autocast=(torch.autocast(device_type=device_type,dtype=torch.get_autocast_dtype(device_type),cache_enabled=False)
                        if torch.is_autocast_enabled(device_type) else nullcontext())
        with cache_autocast:
            source = warm_cache_source(head, previous, k)
    else:
        noise=zero_padded(torch.randn_like(target),pad)
        target=zero_padded(target,pad)
        source=lerp_path(noise,target,s)
    if pad is not None:
        source=zero_padded(source,pad)
        s=s.masked_fill(pad.bool().unsqueeze(-1),0)
        t=t.masked_fill(pad.bool().unsqueeze(-1),0)
    active=(t>s)
    if pad is not None:
        active=active & ~pad.bool().unsqueeze(-1)
    if _endpoint_source == 'cache':
        active=active & _previous_valid[:,k-1].repeat(self.repeated_diffusion_steps)[:,None,None]
    device_type=source.device.type
    autocast=(torch.autocast(device_type=device_type,dtype=torch.get_autocast_dtype(device_type),cache_enabled=False)
              if torch.is_autocast_enabled(device_type) else nullcontext())
    with autocast:
        teacher_target=None
        if _endpoint_weight:
            with torch.no_grad():
                teacher_head=teacher_head_for(head)
                teacher_state=teacher_head._encode_state(context)
                def teacher(x,a,b):
                    return teacher_head._predict_velocity(x,a,b,context,state_features=teacher_state,
                                                          action_padding_mask=pad)
            teacher_target=instantaneous_endpoint(
                teacher, source, s, t, steps=_endpoint_teacher_steps
            )
        student_state=head._encode_state(context)
        velocity=head._predict_velocity(source,s,t,context,state_features=student_state,
                                         action_padding_mask=pad)
        student_target=source+(t-s)*velocity.float()
        valid=active.expand_as(target)
        endpoint_loss=(endpoint_mse(masked_squared_error(student_target-teacher_target,valid),valid,cfg.chunk_size,_front_weight)
                       if teacher_target is not None else student_target.new_zeros(()))
        execution_loss=student_target.new_zeros(())
        if _execution_weight:
            front_valid=valid[:,:cfg.chunk_size]
            execution_loss=endpoint_mse(masked_squared_error(student_target[:,:cfg.chunk_size]-target[:,:cfg.chunk_size],front_valid),
                                       front_valid,cfg.chunk_size,1.)
        total=loss+ramp*(_endpoint_weight*endpoint_loss+_execution_weight*execution_loss)
    head.last_loss_stats['endpoint_consistency_loss']=float(endpoint_loss.detach())
    head.last_loss_stats['endpoint_consistency_weight']=_endpoint_weight*ramp
    head.last_loss_stats['endpoint_refinement_steps']=float(k)
    if _execution_weight:
        head.last_loss_stats['execution_loss']=float(execution_loss.detach())
        head.last_loss_stats['execution_weight']=_execution_weight*ramp
    if _endpoint_source == 'cache':
        head.last_loss_stats['endpoint_history_valid_frac']=float(_previous_valid[:,k-1].float().mean())
    head.last_loss_stats['loss']=float(total.detach())
    return total


def main():
    global _endpoint_weight, _front_weight, _endpoint_teacher_steps, _teacher_weights, _frozen_teacher, _lsd_normalization, _lsd_robust_threshold, _endpoint_source, _clock_sampling, _execution_weight, _execution_k_probabilities, _offdiagonal_fm_weight
    _teacher_weights=None
    _frozen_teacher=None
    if len(sys.argv)<2 or sys.argv[1]!='train' or '--resume-from' not in sys.argv or '--output' not in sys.argv:
        raise SystemExit('Requires train --resume-from CHECKPOINT --output OUTPUT (run.py arguments)')
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--endpoint-weight', type=float, default=None)
    parser.add_argument('--endpoint-front-weight', type=float, default=None)
    parser.add_argument('--endpoint-teacher-steps', type=int, default=None)
    parser.add_argument('--endpoint-teacher', choices=('self','frozen'), default=None)
    parser.add_argument('--lsd-normalization', choices=('active','valid','ramp10'), default=None)
    parser.add_argument('--lsd-robust-threshold', type=float, default=None)
    parser.add_argument('--endpoint-source', choices=('path','cache'), default=None)
    parser.add_argument('--clock-sampling', choices=('original','mixed','edge'), default=None)
    parser.add_argument('--execution-loss-weight', type=float, default=None)
    parser.add_argument('--execution-k-probs', default=None,
                        help='Endpoint K probabilities in order K=1,2,4,8; four comma-separated values (default: uniform)')
    parser.add_argument('--offdiagonal-fm-weight', type=float, default=None,
                        help='Training-only direct FM supervision for u(x_s,s,t)')
    options, remaining=parser.parse_known_args(sys.argv[2:])
    sys.argv=sys.argv[:2]+remaining
    resume_path=Path(sys.argv[sys.argv.index('--resume-from')+1])
    checkpoint=torch.load(resume_path,map_location='cpu',weights_only=False)
    previous_path=Path(checkpoint['arguments']['output'])/'endpoint_consistency_ablation.json'
    chunk_size=int(checkpoint['arguments']['chunk'])
    del checkpoint
    previous=json.loads(previous_path.read_text()) if previous_path.is_file() else None
    _endpoint_weight=resolve_endpoint_weight(options.endpoint_weight,previous)
    _front_weight=resolve_front_weight(options.endpoint_front_weight,previous)
    _endpoint_teacher_steps=(
        int(previous.get('teacher_steps', 8)) if options.endpoint_teacher_steps is None and previous
        else (8 if options.endpoint_teacher_steps is None else options.endpoint_teacher_steps)
    )
    if _endpoint_teacher_steps < 1:
        raise ValueError('endpoint teacher steps must be positive')
    if previous and _endpoint_teacher_steps != int(previous.get('teacher_steps', 8)):
        raise ValueError('Cannot change endpoint teacher steps while resuming this intervention')
    teacher_mode=resolve_teacher_mode(options.endpoint_teacher,previous)
    lsd_normalization=resolve_lsd_normalization(options.lsd_normalization,previous)
    _lsd_normalization=lsd_normalization
    _lsd_robust_threshold=resolve_robust_threshold(options.lsd_robust_threshold,previous)
    _endpoint_source=resolve_endpoint_source(options.endpoint_source,previous)
    _clock_sampling=resolve_clock_sampling(options.clock_sampling,previous)
    _execution_weight=resolve_execution_weight(options.execution_loss_weight,previous)
    _execution_k_probabilities=resolve_execution_k_probabilities(options.execution_k_probs,previous)
    _offdiagonal_fm_weight=resolve_offdiagonal_fm_weight(options.offdiagonal_fm_weight,previous)
    if _execution_weight and _endpoint_source!='cache':
        raise ValueError('Execution loss requires cache endpoint source')
    teacher_source=None
    teacher_sha256=None
    if teacher_mode=='frozen':
        teacher_source=Path(previous['teacher_checkpoint']) if previous else resume_path.resolve()
        teacher_sha256=hashlib.sha256(teacher_source.read_bytes()).hexdigest()
        if previous and teacher_sha256!=previous.get('teacher_checkpoint_sha256'):
            raise ValueError('Frozen endpoint teacher checkpoint hash differs')
        teacher_checkpoint=torch.load(teacher_source,map_location='cpu',weights_only=False)
        _teacher_weights={k[len('action_model.'):]:v for k,v in teacher_checkpoint['model'].items()
                          if k.startswith('action_model.')}
        if not _teacher_weights:
            raise ValueError('Frozen teacher checkpoint contains no action head')
        del teacher_checkpoint
    output=Path(sys.argv[sys.argv.index('--output')+1])
    output.mkdir(parents=True,exist_ok=True)
    destination=output/'endpoint_consistency_ablation.py'
    if destination.exists():raise FileExistsError(destination)
    shutil.copyfile(Path(__file__),destination)
    clock_path=Path(__file__).with_name('train_deployment_clock_ablation.py')
    spec=importlib.util.spec_from_file_location('deployment_clock_ablation',clock_path)
    clock=importlib.util.module_from_spec(spec);spec.loader.exec_module(clock)
    shutil.copyfile(clock_path,output/'deployment_clock_ablation.py')
    (output/'endpoint_consistency_ablation.json').write_text(json.dumps({
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'clock_source_sha256':hashlib.sha256(clock_path.read_bytes()).hexdigest(),
        'teacher':'instantaneous action head under no_grad, Euler-integrated',
        'teacher_steps':_endpoint_teacher_steps,
        'teacher_mode':teacher_mode,
        'teacher_checkpoint':str(teacher_source) if teacher_source else None,
        'teacher_checkpoint_sha256':teacher_sha256,
        'student':'average velocity endpoint, deployment-shaped clocks K=1/2/4/8',
        'weight':_endpoint_weight,'front_weight':_front_weight,'endpoint_source':_endpoint_source,
        'clock_sampling':_clock_sampling,'execution_weight':_execution_weight,
        'execution_k_probabilities':list(_execution_k_probabilities),
        'offdiagonal_fm_weight':_offdiagonal_fm_weight,
        'offdiagonal_fm_intervention':_offdiagonal_fm_weight > 0,
        'lsd_normalization':lsd_normalization,'lsd_robust_threshold':_lsd_robust_threshold,
        'ramp_steps':1000,'loss_units':'displacement',
        'production_fm_lsd_unchanged':lsd_normalization=='active' and _lsd_robust_threshold==0 and _offdiagonal_fm_weight==0,
        'training_only_ablation':True,
    },indent=2)+'\n')
    StaircaseTimeSampler.sample_training=training_clock_sampler(clock,_clock_sampling)
    StateFlow.loss=augmented_loss
    rollflow_core.central_difference_lsd=_original_central_difference
    rollflow_core.RollFlow.loss=_original_rollflow_loss
    if lsd_normalization in ('valid','ramp10'):
        rollflow_core.central_difference_lsd=valid_normalized_central
        rollflow_core.RollFlow.loss=valid_normalized_rollflow_loss
    if _lsd_robust_threshold > 0:
        rollflow_core.central_difference_lsd=robust_central
    os.environ['STARVLA_MINARI_INTERVENTION']='endpoint'
    from examples.simBenchmarks.Minari import run
    if _endpoint_source == 'cache':
        run.sample = cache_aware_sample(run.sample, chunk_size)
    run.main()


if __name__=='__main__':main()
