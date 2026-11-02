"""Paired replay and environment comparison with actual DiT call counts."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def bootstrap_difference(candidate, reference, seed=42, samples=10000):
    difference=np.asarray(candidate,dtype=np.float64)-np.asarray(reference,dtype=np.float64)
    if difference.ndim!=1 or len(difference)<2 or not np.isfinite(difference).all():
        raise ValueError('Need at least two paired finite measurements')
    rng=np.random.default_rng(seed)
    means=rng.choice(difference,size=(samples,len(difference)),replace=True).mean(axis=1)
    return dict(mean=float(difference.mean()),ci95=[float(x) for x in np.quantile(means,[.025,.975])],n=len(difference))


def paired_episodes(candidate, reference):
    def lookup(rows):
        out={row['episode_id']:row['mse'] for row in rows}
        if len(out)!=len(rows):raise ValueError('Duplicate replay episode ID')
        return out
    left,right=lookup(candidate),lookup(reference)
    if set(left)!=set(right):raise ValueError('Replay episode IDs differ')
    ids=sorted(left)
    return [left[i] for i in ids],[right[i] for i in ids]


def environment_seeds(value, allow_legacy):
    seeds=value.get('environment_seeds')
    if seeds is None:
        if not allow_legacy:raise ValueError('Missing environment seeds; use --legacy-runner-seeds only after checking source')
        seeds=list(range(60000,60000+len(value['environment_returns'])))
    if len(seeds)!=len(value['environment_returns']) or len(set(seeds))!=len(seeds):
        raise ValueError('Invalid environment seeds')
    return seeds


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--reference-mode',default='instant_k4')
    p.add_argument('--candidates',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--legacy-runner-seeds',action='store_true')
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    reference=json.loads(args.reference.read_text())[args.reference_mode]
    ref_seeds=environment_seeds(reference,args.legacy_runner_seeds)
    results=[]
    for candidate_path in args.candidates:
        for mode,entry in json.loads(candidate_path.read_text()).items():
            if environment_seeds(entry,args.legacy_runner_seeds)!=ref_seeds:
                raise ValueError('Environment seeds differ or are ordered differently')
            left,right=paired_episodes(entry['replay_episodes'],reference['replay_episodes'])
            results.append(dict(candidate=str(candidate_path),mode=mode,
                mean_return=entry['mean_return'],replay_global_mse=entry['replay_normalized_mse'],
                average_dit_calls=float(np.mean(entry['dit_calls_per_chunk'])),
                return_difference=bootstrap_difference(entry['environment_returns'],reference['environment_returns']),
                episode_mse_difference=bootstrap_difference(left,right)))
    output=dict(reference=str(args.reference),reference_mode=args.reference_mode,
        reference_mean_return=reference['mean_return'],reference_global_mse=reference['replay_normalized_mse'],
        legacy_seed_assumption=args.legacy_runner_seeds,results=results)
    # Validate inputs before creating the output directory, so a failed
    # comparison can be corrected and retried with the same path.
    args.output.mkdir(parents=True)
    (args.output/'comparison.json').write_text(json.dumps(output,indent=2)+'\n')
    candidate_ids={str(path):index+1 for index,path in enumerate(args.candidates)}
    report=['# 少步滚动生成对照','',
        f"参考：`{args.reference}` 的 `{args.reference_mode}`，平均回报 {reference['mean_return']:.3f}。",
        '回报差=候选−参考，越接近 0 越接近参考；区间按相同环境种子做 10000 次配对自助采样。',
        '这不包含训练种子的不确定性；比较需多个独立训练种子。平均调用数包含回放和闭环的冷启动与热态。','']
    report += [f"候选 {index}: `{path}`" for path,index in candidate_ids.items()]
    report += ['',
        '|候选|模式|回放归一化 MSE|平均 return|return 差（95% 配对区间）|平均 DiT 次数/chunk|',
        '|---|---|---:|---:|---:|---:|']
    for row in results:
        d=row['return_difference'];lo,hi=d['ci95']
        report.append(f"|{candidate_ids[row['candidate']]}: {Path(row['candidate']).parent.name}|{row['mode']}|{row['replay_global_mse']:.5f}|{row['mean_return']:.2f}|{d['mean']:.2f} [{lo:.2f}, {hi:.2f}]|{row['average_dit_calls']:.3f}|")
    labels=[f"{candidate_ids[r['candidate']]}: {Path(r['candidate']).parent.name}\n{r['mode']}" for r in results]
    means=np.array([r['return_difference']['mean'] for r in results])
    intervals=np.array([r['return_difference']['ci95'] for r in results])
    fig,ax=plt.subplots(figsize=(max(8,len(results)*1.25),4.8))
    ax.bar(range(len(results)),means,color=['#4682b4' if 'instant' in r['mode'] else '#e18d48' for r in results])
    # A percentile bootstrap interval need not contain the point estimate.
    # Plot its endpoints directly instead of requiring non-negative yerr.
    positions=np.arange(len(results))
    ax.vlines(positions,intervals[:,0],intervals[:,1],color='black')
    ax.hlines(intervals[:,0],positions-.06,positions+.06,color='black')
    ax.hlines(intervals[:,1],positions-.06,positions+.06,color='black')
    ax.axhline(0,color='black',linewidth=.8);ax.set_ylabel('Paired return difference vs reference')
    ax.set_xticks(range(len(labels)),labels,rotation=35,ha='right');ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(args.output/'paired_returns.png',dpi=160);plt.close(fig)
    report+=['','![Paired returns](paired_returns.png)','']
    (args.output/'report.md').write_text('\n'.join(report))
    print(args.output/'report.md')


if __name__=='__main__':main()
