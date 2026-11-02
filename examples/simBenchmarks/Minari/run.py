"""Three-arm state-only Minari experiment using production action heads."""
from __future__ import annotations
import argparse
import json
import os
import sys
import copy
import hashlib
import subprocess
import time
from importlib.metadata import PackageNotFoundError, version
from contextlib import nullcontext
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from starVLA.model.framework.StateAction.StateFlow import StateFlow
from starVLA.model.modules.action_model import GR00T_ActionHeader as standard
from starVLA.model.modules.action_model import GR00T_RollFlow_ActionHeader as rolling


def seed_all(seed):
    np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def collect(args):
    import gymnasium as gym
    import minari
    # Locally generated diagnostic data, not an official expert benchmark.
    env=minari.DataCollector(gym.make('Pendulum-v1'))
    rng=np.random.default_rng(args.seed)
    returns=[]
    for ep in range(args.episodes):
        obs,_=env.reset(seed=args.seed+ep);total=0.
        for _ in range(200):
            # Random-shooting MPC, actual Pendulum dynamics. Independent of learner.
            candidates=rng.uniform(-2,2,(128,24))
            candidates[:3]=np.array([-2.,0.,2.])[:,None]
            theta=np.full(128,np.arctan2(obs[1],obs[0]));vel=np.full(128,obs[2]);cost=np.zeros(128)
            for k in range(24):
                angle=(theta+np.pi)%(2*np.pi)-np.pi
                u=candidates[:,k];cost+=angle**2+.1*vel**2+.001*u**2
                vel=np.clip(vel+(15*np.sin(theta)+3*u)*.05,-8,8);theta+=vel*.05
            action=np.array([candidates[cost.argmin(),0]],dtype=np.float32)
            obs,r,terminated,truncated,_=env.step(action);total+=float(r)
            if terminated or truncated:break
        returns.append(total)
    ds=env.create_dataset(dataset_id=args.dataset,algorithm_name='random-shooting MPC diagnostic',
        author='RollFlow diagnostic',author_email='diagnostic@example.com',
        description='Local Pendulum diagnostic trajectories; not a certified expert policy.')
    env.close()
    print(json.dumps(dict(dataset=ds.id,episodes=len(returns),mean_return=float(np.mean(returns)))))


def read_episodes(ds,limit):
    import gymnasium as gym
    if not isinstance(ds.observation_space,gym.spaces.Box) or len(ds.observation_space.shape)!=1:
        raise ValueError('This diagnostic supports flat Box observations only')
    if not isinstance(ds.action_space,gym.spaces.Box) or len(ds.action_space.shape)!=1:
        raise ValueError('This diagnostic supports continuous vector Box actions only')
    episodes=[]
    for ep in ds.iterate_episodes():
        obs=np.asarray(ep.observations,dtype=np.float32);actions=np.asarray(ep.actions,dtype=np.float32)
        if len(obs)!=len(actions)+1:raise ValueError('Expected T+1 observations for T actions')
        if not np.isfinite(obs).all() or not np.isfinite(actions).all():raise ValueError('Nonfinite data')
        if len(actions):episodes.append((int(ep.id),obs,actions))
        if limit and len(episodes)>=limit:break
    if len(episodes)<5:raise ValueError('Need at least five nonempty episodes for train/validation/test')
    return episodes


def prepare(ds,args):
    episodes=read_episodes(ds,args.max_episodes)
    rng=np.random.default_rng(args.seed);order=rng.permutation(len(episodes))
    n=max(1,int(len(episodes)*.15));indices={'train':order[:-2*n],'validation':order[-2*n:-n],'test':order[-n:]}
    train_obs=np.concatenate([episodes[i][1][:-1] for i in indices['train']])
    low=np.asarray(ds.action_space.low,dtype=np.float32);high=np.asarray(ds.action_space.high,dtype=np.float32)
    if not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(high<=low):
        raise ValueError('Finite nonempty action bounds required')
    stats=dict(state_mean=train_obs.mean(0),state_std=np.maximum(train_obs.std(0),1e-4),
               action_center=(high+low)/2,action_scale=(high-low)/2)
    data={key:[episodes[i] for i in ix] for key,ix in indices.items()}
    return data,stats


def dataset_fingerprint(data):
    """Hash the exact split order, observations, and actions used by training."""
    digest=hashlib.sha256()
    for split in ('train','validation','test'):
        digest.update(split.encode())
        for episode_id,obs,actions in data[split]:
            digest.update(np.asarray([episode_id],dtype='<i8').tobytes())
            for array in (obs,actions):
                array=np.ascontiguousarray(array)
                digest.update(str((array.shape,array.dtype.str)).encode())
                digest.update(array.tobytes())
    return digest.hexdigest()


def sample(data,stats,horizon,batch_size,rng,device,complete=True):
    # Common support: original FM cannot mask padded target actions. All arms
    # therefore sample complete windows; no episode boundary enters training.
    lengths=np.array([max(0,len(ep[2])-horizon+1) if complete else len(ep[2]) for ep in data])
    total=int(lengths.sum());ends=lengths.cumsum()
    if total == 0:raise ValueError('No complete action windows in this split')
    anchors=rng.integers(total,size=batch_size);states=[];targets=[];pads=[]
    for anchor in anchors:
        idx=int(np.searchsorted(ends,anchor,side='right'));start=int(anchor-(ends[idx-1] if idx else 0))
        _,obs,act=data[idx];valid=min(horizon,len(act)-start)
        action=np.zeros((horizon,act.shape[1]),np.float32)
        action[:valid]=(act[start:start+valid]-stats['action_center'])/stats['action_scale']
        states.append((obs[start]-stats['state_mean'])/stats['state_std']);targets.append(action)
        pads.append(np.arange(horizon)>=valid)
    return tuple(torch.as_tensor(np.stack(v),device=device) for v in (states,targets,pads))


def build_config(args,state_dim,action_dim):
    lsd=args.variant=='rolling_lsd'
    return dict(action_model_type='diagnostic',action_horizon=args.horizon,execution_horizon=args.chunk,
        action_dim=action_dim,state_dim=state_dim,hidden_size=args.width,add_pos_embed=True,
        repeated_diffusion_steps=args.repeat,
        max_seq_len=args.horizon,num_target_vision_tokens=2,num_timestep_buckets=1000,
        inference_steps=args.nfe,velocity_mode='average' if lsd else 'instant',
        num_inference_timesteps=args.nfe,noise_beta_alpha=1.5,noise_beta_beta=1.,noise_s=.999,
        p_k1=args.p_k1,p_fm=args.p_fm,
        fm_curriculum_steps=args.fm_warmup,
        w_fm=1.,w_lsd=args.w_lsd if lsd else 0.,
        lsd_loss_space=args.loss_space,lsd_warmup_steps=args.lsd_ramp,
        finite_difference_delta=args.delta,use_ot=args.ot,iterative_cold_start=True,
        diffusion_model_cfg=dict(input_embedding_dim=args.width,num_attention_heads=4,
            attention_head_dim=args.width//4,num_layers=args.layers,output_dim=args.width,
            cross_attention_dim=state_dim,dropout=0.,final_dropout=False,
            positional_embeddings=None,interleave_self_attention=True,norm_type='ada_norm',
            skip_zero_interval_embedding=args.skip_zero_interval))


def make_model(config,steps,device,variant):
    full=SimpleNamespace(framework=SimpleNamespace(name='StateFlow',variant=variant,action_model=SimpleNamespace(**config)),
                         trainer=SimpleNamespace(max_train_steps=steps))
    # The original head derives encoder width from its named preset. Register
    # the small diagnostic shape ONLY for construction, without changing it.
    shape={k:config['diffusion_model_cfg'][k] for k in ('input_embedding_dim','num_attention_heads','attention_head_dim')}
    with patch.dict(standard.DiTConfig,diagnostic=shape),patch.dict(rolling.DiTConfig,diagnostic=shape):
        model=StateFlow(full)
    return model.to(device)


def forward(model,b,step):
    state,action,pad=b
    # State is also the cross-attention context: no VLM and no added trainable adapter.
    return model.loss(state,action,pad=pad,training_step=step)


def precision_context(args):
    return (torch.autocast(torch.device(args.device).type,dtype=torch.bfloat16,
                           cache_enabled=not getattr(args,'disable_autocast_cache',False))
            if args.precision=='bf16' else nullcontext())


def velocity(model,x,t,state):
    head=model.action_model
    if model.variant!='fm':
        return head._predict_velocity(x,t,t,state[:,None],state_features=head._encode_state(state[:,None]))
    # Deterministic original network path for a common validation metric only.
    # Training and Euler inference use the original head's own implementations.
    clock=(t[:,0,0]*head.num_timestep_buckets).long()
    features=head.action_encoder(x,clock)
    features=features+head.position_embedding(torch.arange(x.shape[1],device=x.device))[None]
    hidden=torch.cat((head.state_encoder(state[:,None]),head.future_tokens.weight[None].expand(len(x),-1,-1),features),1)
    return head.action_decoder(head.model(hidden,state[:,None],timestep=clock,return_all_hidden_states=False))[:,-x.shape[1]:].float()


@torch.no_grad()
def validate(model,data,stats,args,step):
    # Fixed FM metric across objectives, separate from noncomparable total FM+LSD loss.
    model.eval();metrics=[]
    with torch.random.fork_rng(devices=[] if args.device=='cpu' else [torch.device(args.device).index or 0]):
        torch.manual_seed(9341);rng=np.random.default_rng(9341)
        for _ in range(4):
            state,actions,pad=sample(data,stats,args.horizon,args.batch,rng,args.device)
            noise=torch.randn_like(actions).masked_fill(pad[...,None],0)
            # Bucket-aligned common times remove continuous/quantized clock
            # differences from this metric, while training remains native.
            t=(torch.randint(1,8,(len(state),1,1),device=args.device)/8).expand(-1,args.horizon,-1)
            x=(1-t)*noise+t*actions
            with precision_context(args):
                pred=velocity(model,x,t,state)
                shuffled=velocity(model,x,t,state.roll(1,0))
            error=(pred-(actions-noise)).square().masked_fill(pad[...,None],0)
            metrics.append(dict(validation_fm_mse=float(error.mean()),
                shuffled_state_fm_mse=float((shuffled-(actions-noise)).square().mean()),
                zero_velocity_fm_mse=float((actions-noise).square().mean()),
                velocity_rms=float(pred.square().mean().sqrt())))
    model.train();return {k:float(np.mean([v[k] for v in metrics])) for k in metrics[0]}


@torch.no_grad()
def conditioning_metrics(model):
    """Probe learned time modulation on a fixed diagonal grid, without RNG draws."""
    dit=model.action_model.model
    parameter=next(dit.parameters())
    times=torch.linspace(0,1000,33,device=parameter.device)
    endpoint=dit.timestep_encoder(times)
    interval=torch.zeros_like(endpoint)
    if hasattr(dit,'interval_timestep_encoder'):
        interval=dit.interval_timestep_encoder(torch.zeros_like(times))
    temb=endpoint if getattr(dit.config,'skip_zero_interval_embedding',False) else endpoint+interval
    modulation=[block.norm1.linear(block.norm1.silu(temb))
                for block in dit.transformer_blocks if hasattr(block.norm1,'linear')]
    output=dit.proj_out_1(torch.nn.functional.silu(temb))
    return dict(clock_endpoint_rms=float(endpoint.float().square().mean().sqrt()),
        clock_interval_zero_rms=float(interval.float().square().mean().sqrt()),
        clock_combined_rms=float(temb.float().square().mean().sqrt()),
        clock_adaln_max_abs=max((float(value.abs().max()) for value in modulation),default=0.),
        clock_output_max_abs=float(output.abs().max()))


def evaluation_refinements(args):
    values = getattr(args, 'eval_nfe', None)
    values = sorted(set(values if values is not None else [1, args.nfe]))
    if not values or any(k < 1 or k > args.horizon // args.chunk for k in values):
        raise ValueError('Evaluation refinement steps must lie in [1, horizon/chunk]')
    return values


@torch.no_grad()
def evaluation_environment_seeds(args):
    start = getattr(args, 'eval_seed_start', 60000)
    count = args.eval_episodes
    if start < 0 or count < 1 or start + count > 2**32:
        raise ValueError('Evaluation seed range must be nonnegative and fit in uint32')
    return list(range(start, start + count))


def evaluate(model,ds,data,stats,args):
    model.eval();results={}
    environment_seeds=evaluation_environment_seeds(args)
    for mode in (['instant','average'] if args.variant=='rolling_lsd' else ['instant']):
        for nfe in evaluation_refinements(args):
            key=f'{mode}_k{nfe}';seed_all(50001)
            calls=[];latencies=[]
            def predict(state):
                counter=[0]
                def count(*unused):counter[0]+=1
                handle=model.action_model.model.register_forward_hook(count)
                if torch.device(args.device).type=='cuda':torch.cuda.synchronize()
                start=time.perf_counter()
                try:
                    with precision_context(args):
                        out=model.sample_chunk(state,refinement_steps=nfe,velocity_mode=mode)
                    if torch.device(args.device).type=='cuda':torch.cuda.synchronize()
                finally:handle.remove()
                calls.append(counter[0]);latencies.append(time.perf_counter()-start)
                return out[0].float().cpu().numpy()
            squared=0.;count=0;per_episode=[]
            # Offline replay: advance recorded observations by C, preserve rolling cache.
            for ep_id,obs,actions in data['test']:
                model.reset();predictions=[]
                for t in range(0,len(actions),args.chunk):
                    state=torch.as_tensor((obs[t]-stats['state_mean'])/stats['state_std'],device=args.device)[None,None]
                    pred=predict(state)
                    predictions.append(pred[:min(args.chunk,len(actions)-t)])
                pred=np.concatenate(predictions);truth=(actions-stats['action_center'])/stats['action_scale']
                err=(pred-truth)**2;squared+=float(err.sum());count+=err.size
                per_episode.append(dict(episode_id=ep_id,mse=float(err.mean())))
                np.savez_compressed(args.output/f'replay_{key}_ep{ep_id}.npz',
                    prediction=pred*stats['action_scale']+stats['action_center'],target=actions)
            env=ds.recover_environment();returns=[];lengths=[];clip_count=0;action_count=0
            try:
                for environment_seed in environment_seeds:
                    seed_all(environment_seed);obs,_=env.reset(seed=environment_seed);model.reset();total=0.;steps=0;done=False
                    while not done and steps<args.eval_max_steps:
                        state=torch.as_tensor((np.asarray(obs,dtype=np.float32)-stats['state_mean'])/stats['state_std'],device=args.device)[None,None]
                        chunk=predict(state)
                        for normalized in chunk:
                            action=normalized*stats['action_scale']+stats['action_center']
                            if not np.isfinite(action).all():raise FloatingPointError('Nonfinite policy action')
                            clipped=np.clip(action,env.action_space.low,env.action_space.high)
                            clip_count+=int(np.any(action!=clipped));action_count+=1
                            obs,reward,terminated,truncated,_=env.step(clipped)
                            total+=float(reward);steps+=1;done=terminated or truncated
                            if done or steps>=args.eval_max_steps:break
                    returns.append(total);lengths.append(steps)
            finally:env.close();model.reset()
            results[key]=dict(replay_normalized_mse=squared/max(count,1),replay_episodes=per_episode,
                environment_returns=returns,environment_lengths=lengths,
                environment_seeds=environment_seeds,
                mean_return=float(np.mean(returns)),action_clip_fraction=clip_count/max(action_count,1),
                dit_calls_per_chunk=calls,seconds_per_chunk=latencies)
    dump(args.output/'evaluation.json',results)
    print(json.dumps({k:{n:v[n] for n in ('replay_normalized_mse','mean_return','action_clip_fraction')} for k,v in results.items()},indent=2),flush=True)
    return results


def validate_intervention_resume(resume):
    """Reject a continuation that silently drops a process-local ablation."""
    if resume is None:
        return
    previous = Path(resume['arguments']['output'])
    if (previous / 'lsd_clock_only_ablation.json').is_file():
        expected = 'lsd_clock_only'
    elif (previous / 'endpoint_consistency_ablation.json').is_file():
        expected = 'endpoint'
    elif (previous / 'deployment_clock_ablation.json').is_file():
        expected = 'clock'
    else:
        return
    actual = os.environ.get('STARVLA_MINARI_INTERVENTION')
    if actual != expected:
        raise ValueError(
            f'Checkpoint requires the {expected} intervention, but this process '
            f'has {actual or "none"}; resume using the same ablation training '
            'entry point. --allow-resume-code-change does not enable an intervention.'
        )


def train(args, resume=None):
    validate_intervention_resume(resume)
    import minari
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'config.json').exists():raise FileExistsError('Use a new output directory')
    ds=minari.load_dataset(args.dataset,download=args.download);data,stats=prepare(ds,args)
    data_hash=dataset_fingerprint(data)
    seed_all(args.seed);cfg=build_config(args,len(stats['state_mean']),len(stats['action_center']))
    if resume is not None:
        if cfg != resume['head_config']:raise ValueError('Resume head configuration differs')
        if {k:[ep[0] for ep in v] for k,v in data.items()} != resume['split']:
            raise ValueError('Resume dataset split differs')
        if any(not np.array_equal(stats[k],np.asarray(resume['normalization'][k],dtype=np.float32)) for k in stats):
            raise ValueError('Resume normalization differs')
        if resume.get('dataset_sha256') is not None and resume['dataset_sha256']!=data_hash:
            raise ValueError('Resume dataset content differs')
    # All arms start with identical common weights. RollFlow's existing strict
    # loading compatibility initializes interval embeddings from time embeddings.
    base_cfg=copy.deepcopy(cfg);base_cfg['diffusion_model_cfg'].pop('skip_zero_interval_embedding',None)
    base=make_model(base_cfg,args.steps,'cpu','fm')
    initial=copy.deepcopy(base.action_model.state_dict())
    del base
    model=make_model(cfg,args.steps,args.device,args.variant)
    model.action_model.load_state_dict(initial,strict=True)
    init_hash=hashlib.sha256(b''.join(v.cpu().contiguous().numpy().tobytes() for v in initial.values())).hexdigest()
    del initial
    sources=[Path(__file__).resolve(),ROOT/'starVLA/model/framework/StateAction/StateFlow.py',
             ROOT/'starVLA/model/modules/action_model/GR00T_ActionHeader.py',
             ROOT/'starVLA/model/modules/action_model/GR00T_RollFlow_ActionHeader.py']
    for folder in ('flow_matching_head','rolling_meanflow_matching_head'):
        sources.extend((ROOT/'starVLA/model/modules/action_model'/folder).glob('*.py'))
    hashes={}
    for source in sources:
        relative=source.relative_to(ROOT);dest=args.output/'source'/relative
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(source.read_bytes())
        hashes[str(relative)]=hashlib.sha256(source.read_bytes()).hexdigest()
    changed=[]
    if resume is not None:
        previous=Path(resume['arguments']['output'])/'manifest.json'
        if not previous.is_file():raise FileNotFoundError(f'Missing prior source manifest: {previous}')
        old_hashes=json.loads(previous.read_text())['source_sha256']
        # Runner changes are recorded; model and framework changes require an
        # explicit flag so an experiment cannot silently change algorithms.
        changed=sorted(name for name in set(hashes)|set(old_hashes) if old_hashes.get(name)!=hashes.get(name))
        critical=[name for name in changed if name.startswith('starVLA/model/')]
        if critical and not args.allow_resume_code_change:
            raise ValueError(f'Model source differs from checkpoint: {critical}; use --allow-resume-code-change only for a deliberate new comparison')
        model.load_state_dict(resume['model'],strict=True)
    dump(args.output/'manifest.json',dict(source_sha256=hashes,shared_initial_sha256=init_hash,
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        torch=torch.__version__,python=sys.version,training_windows='complete_only',
        validation_metric='fixed global bucket-aligned FM',precision=args.precision,
        resumed_from=None if resume is None else str(args.resume_from),changed_sources=changed,
        dataset_sha256=data_hash,dataset_resume_verification=(None if resume is None else
            ('fingerprint' if resume.get('dataset_sha256') is not None else 'split_and_normalization_only')),
        scheduler_after_original_budget='hold final LR'))
    dump(args.output/'config.json',dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},head=cfg,
        split={k:[ep[0] for ep in v] for k,v in data.items()},normalization={k:v.tolist() for k,v in stats.items()},
        parameters=sum(p.numel() for p in model.parameters()),
        raw_batch=args.batch,diffusion_batch=args.batch*args.repeat))
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.steps,eta_min=args.lr*.05)
    first_step=1;last_step=args.steps
    if resume is not None:
        opt.load_state_dict(resume['optimizer']);sched.load_state_dict(resume['scheduler'])
        first_step=int(resume['step'])+1;last_step=first_step+args.additional_steps-1
    best=float('inf');started=time.perf_counter()
    with (args.output/'metrics.jsonl').open('w') as log:
        for step in range(first_step,last_step+1):
            # Step-local RNG means evaluation and differing LSD forward counts
            # cannot alter subsequent minibatch/noise draws across paired arms.
            seed_all(args.seed*100000+step)
            rng=np.random.default_rng(args.seed*100000+step)
            model.train();opt.zero_grad();b=sample(data['train'],stats,args.horizon,args.batch,rng,args.device)
            with precision_context(args):loss=forward(model,b,step)
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(loss) or not torch.isfinite(norm):
                torch.save(dict(model=model.state_dict(),batch=b,step=step,
                    optimizer=opt.state_dict(),head_config=cfg),args.output/'nonfinite.pt')
                raise FloatingPointError(f'Nonfinite step {step}; diagnostic saved before optimizer update')
            opt.step()
            if step<=args.steps:sched.step()
            if step==first_step or step%args.log_every==0 or step%args.save_every==0 or step==last_step:
                record={k:float(v) for k,v in model.last_loss_stats.items()}
                record.update(step=step,loss=float(loss),grad_norm=float(norm),lr=sched.get_last_lr()[0],elapsed_seconds=time.perf_counter()-started)
                record.update(validate(model,data['validation'],stats,args,step))
                record.update(conditioning_metrics(model))
                log.write(json.dumps(record,allow_nan=False)+'\n');log.flush();print(json.dumps(record),flush=True)
                payload=dict(model=model.state_dict(),head_config=cfg,arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                             normalization={k:v.tolist() for k,v in stats.items()},split={k:[ep[0] for ep in v] for k,v in data.items()},step=step,
                             dataset_sha256=data_hash,
                             optimizer=opt.state_dict(),scheduler=sched.state_dict(),torch_rng=torch.get_rng_state())
                torch.save(payload,args.output/'last.pt')
                if step%args.save_every==0:
                    torch.save(payload,args.output/f'step_{step}.pt')
                if record['validation_fm_mse']<best:
                    best=record['validation_fm_mse'];torch.save(payload,args.output/'best_fm.pt')
    # Final checkpoint: FM selection alone is not suitable to select distilled average velocity.
    evaluate(model,ds,data,stats,args)


def restore_test_order(all_eps, saved_ids):
    # Episode order consumes RNG in replay. Preserve it across checkpoint
    # reloads, rather than reconstructing the test set in dataset storage order.
    by_id={ep[0]:ep for ep in all_eps}
    if len(by_id)!=len(all_eps):raise ValueError('Duplicate episode IDs')
    if any(ep_id not in by_id for ep_id in saved_ids):raise ValueError('Test episode IDs missing')
    return [by_id[ep_id] for ep_id in saved_ids]


def eval_checkpoint(args):
    import minari
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    saved=argparse.Namespace(**payload['arguments'])
    saved.device=args.device;saved.output=args.output;saved.eval_episodes=args.eval_episodes;saved.eval_max_steps=args.eval_max_steps
    saved.eval_seed_start=args.eval_seed_start
    if args.eval_nfe is not None:saved.eval_nfe=args.eval_nfe
    if args.eval_precision is not None:saved.precision=args.eval_precision
    args.output.mkdir(parents=True,exist_ok=True)
    ds=minari.load_dataset(saved.dataset,download=args.download)
    env=ds.recover_environment()
    try:
        env.action_space.seed(saved.eval_seed_start)
        observation,_=env.reset(seed=saved.eval_seed_start)
        transition=env.step(env.action_space.sample())
        if len(transition)!=5:
            raise RuntimeError(f'Expected Gymnasium five-value step API, got {len(transition)} values')
        env_preflight=dict(status='passed',env_id=getattr(getattr(env,'spec',None),'id',None),
            entry_point=getattr(getattr(env,'spec',None),'entry_point',None),
            observation_shape=list(np.asarray(observation).shape),
            action_shape=list(getattr(env.action_space,'shape',())))
    finally:
        env.close()
    runtime_versions={}
    for package in ('minari','gymnasium','mujoco','numpy','torch'):
        try:runtime_versions[package]=version(package)
        except PackageNotFoundError:runtime_versions[package]=None
    import gymnasium
    dump(args.output/'evaluation_manifest.json',dict(checkpoint=str(args.checkpoint),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        training_precision=payload['arguments']['precision'],evaluation_precision=saved.precision,
        refinements=evaluation_refinements(saved),environment_episodes=saved.eval_episodes,
        environment_max_steps=saved.eval_max_steps,replay_seed=50001,environment_seed_start=saved.eval_seed_start,
        dataset=saved.dataset,dataset_env_spec=str(getattr(ds,'env_spec',None)),
        environment_preflight=env_preflight,package_versions=runtime_versions,
        gymnasium_import_path=gymnasium.__file__,python_executable=sys.executable))
    all_eps=read_episodes(ds,saved.max_episodes)
    data={'test':restore_test_order(all_eps,payload['split']['test'])}
    stats={k:np.asarray(v,dtype=np.float32) for k,v in payload['normalization'].items()}
    model=make_model(payload['head_config'],saved.steps,args.device,saved.variant);model.load_state_dict(payload['model'],strict=True)
    evaluate(model,ds,data,stats,saved)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['collect','train','eval'])
    p.add_argument('--dataset',default='rollflow/pendulum-mpc-v0');p.add_argument('--dataset-root',type=Path,default=Path(os.environ.get('MINARI_DATASET_ROOT', str(Path.home()/'.cache'/'minari'))))
    p.add_argument('--download',action='store_true');p.add_argument('--output',type=Path,default=Path(os.environ.get('ROLLFLOW_MINARI_OUTPUT', 'rollflow_minari_run')))
    p.add_argument('--checkpoint',type=Path);p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=2)
    p.add_argument('--resume-from',type=Path,default=None,help='Resume model, AdamW, and scheduler into a new output directory')
    p.add_argument('--additional-steps',type=int,default=0,help='Extra steps from the checkpoint; hold minimum LR beyond original budget')
    p.add_argument('--allow-resume-code-change',action='store_true',help='Allow and record deliberate model source changes')
    p.add_argument('--seed',type=int,default=42);p.add_argument('--episodes',type=int,default=24);p.add_argument('--max-episodes',type=int,default=100)
    p.add_argument('--steps',type=int,default=2000);p.add_argument('--batch',type=int,default=8);p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--repeat',type=int,default=4,help='Independent noise/time samples per raw observation, as in QwenGR00T')
    p.add_argument('--horizon',type=int,default=32);p.add_argument('--chunk',type=int,default=8);p.add_argument('--nfe',type=int,default=4)
    p.add_argument('--eval-nfe',nargs='+',type=int,default=None,help='Explicit evaluation refinements, e.g. 1 2 4; default 1 and --nfe')
    p.add_argument('--width',type=int,default=64);p.add_argument('--layers',type=int,default=2)
    p.add_argument('--variant',choices=['fm','rolling_fm','rolling_lsd'],default='rolling_fm')
    p.add_argument('--precision',choices=['fp32','bf16'],default='fp32',help='BF16 autocast, with FP32 weights and optimizer')
    p.add_argument('--disable-autocast-cache',action='store_true',help='Diagnostic: disable BF16 autocast weight cache for training/evaluation')
    p.add_argument('--eval-precision',choices=['fp32','bf16'],default=None,help='Override only checkpoint evaluation precision; default uses saved training precision')
    p.add_argument('--p-k1',type=float,default=0.,help='0 samples only multi-group staircase times for rolling arms')
    p.add_argument('--p-fm',type=float,default=.3);p.add_argument('--fm-warmup',type=int,default=200)
    p.add_argument('--w-lsd',type=float,default=.05);p.add_argument('--lsd-ramp',type=int,default=200)
    p.add_argument('--loss-space',choices=['velocity','displacement'],default='velocity');p.add_argument('--delta',type=float,default=.01)
    p.add_argument('--ot',action='store_true');p.add_argument('--skip-zero-interval',action='store_true')
    p.add_argument('--log-every',type=int,default=100);p.add_argument('--eval-episodes',type=int,default=3);p.add_argument('--eval-max-steps',type=int,default=1000)
    p.add_argument('--eval-seed-start',type=int,default=60000,help='First environment seed for the evaluation suite')
    p.add_argument('--save-every',type=int,default=1000)
    a=p.parse_args()
    resume=None
    if a.resume_from is not None:
        if a.command!='train' or a.additional_steps<=0:p.error('--resume-from requires train and positive --additional-steps')
        resume=torch.load(a.resume_from,map_location='cpu',weights_only=False)
        saved=argparse.Namespace(**resume['arguments'])
        saved.output=a.output;saved.device=a.device;saved.threads=a.threads
        saved.resume_from=a.resume_from;saved.additional_steps=a.additional_steps
        saved.allow_resume_code_change=a.allow_resume_code_change
        if a.eval_nfe is not None:saved.eval_nfe=a.eval_nfe
        a=saved
    if a.width%4 or a.width<4 or a.steps<1 or a.batch<1 or a.repeat<1 or a.log_every<1 or a.save_every<1 or a.eval_episodes<1: p.error('Invalid dimensions/budget')
    if getattr(a,'eval_seed_start',60000)<0 or getattr(a,'eval_seed_start',60000)+a.eval_episodes>2**32: p.error('Invalid evaluation seed range')
    if a.command=='eval' and a.checkpoint is None:p.error('--checkpoint is required for eval')
    if a.command=='train' and a.variant=='rolling_lsd' and a.steps<=a.fm_warmup:p.error('LSD run must extend past FM warmup')
    if a.horizon<2 or a.chunk<1 or a.horizon%a.chunk or not 1<=a.nfe<=a.horizon//a.chunk:p.error('Invalid horizon/chunk/refinement steps')
    if a.variant=='fm' and a.ot:p.error('Original FM does not implement OT; use --ot only in rolling arms')
    os.environ['MINARI_DATASETS_PATH']=str(a.dataset_root);torch.set_num_threads(a.threads)
    if a.command=='train':train(a,resume=resume)
    else:{'collect':collect,'eval':eval_checkpoint}[a.command](a)
if __name__=='__main__':main()
