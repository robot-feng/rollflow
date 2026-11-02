"""Generate curves and a Markdown status report without touching training jobs."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    args=parser.parse_args();root=args.root
    suite=json.loads((root/'suite.json').read_text())
    report=['# Minari 实验概况','',
        '训练使用实际动作头。回放 MSE 衡量记录动作拟合，环境 return 衡量闭环控制；二者不能互相替代。',
        '此报告是生成时的快照，running 表示该子进程尚未向队列报告完成；是否存活需查询 PID。','',
        '|实验|队列状态|最新 step|验证 FM MSE|shuffle − 原状态 MSE|',
        '|---|---|---:|---:|---:|']
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    loss_fig,loss_axes=plt.subplots(2,2,figsize=(12,8))
    has_lsd_metrics=False
    for run in suite['runs']:
        directory=root/run['name'];path=directory/'metrics.jsonl'
        rows=[]
        if path.exists():
            for line in path.read_text().splitlines():
                try:rows.append(json.loads(line))
                except json.JSONDecodeError:pass  # final line may currently be in flight
        if not rows:continue
        last=rows[-1];gap=last['shuffled_state_fm_mse']-last['validation_fm_mse']
        report.append(f"|{run['name']}|{run['status']}|{last['step']}|{last['validation_fm_mse']:.6f}|{gap:.6f}|")
        steps=[r['step'] for r in rows]
        for ax,values,title in zip(axs.flat,
            ([r['validation_fm_mse'] for r in rows],[r.get('fm_loss',r['loss']) for r in rows],
             [r['grad_norm'] for r in rows],[r['velocity_rms'] for r in rows]),
            ('Fixed validation FM MSE','Training FM loss (not FM + LSD)', 'Gradient norm before clipping','Validation velocity RMS')):
            ax.plot(steps,values,label=run['name']);ax.set_title(title);ax.set_xlabel('optimizer step');ax.grid(alpha=.2)
        if any('lsd_loss' in row for row in rows):
            has_lsd_metrics=True
            for ax,key,title in zip(loss_axes.flat,
                ('lsd_loss','weighted_lsd_loss','p_fm','active_lsd_frac'),
                ('Conditional LSD loss','Weighted LSD loss','FM-only probability','Active LSD fraction')):
                ax.plot(steps,[r.get(key,float('nan')) for r in rows],label=run['name'])
                ax.set_title(title);ax.set_xlabel('optimizer step');ax.grid(alpha=.2)
    axs[0,0].legend(fontsize=7);fig.tight_layout();fig.savefig(root/'training_curves.png',dpi=150);plt.close(fig)
    if has_lsd_metrics:
        loss_axes[0,0].legend(fontsize=7);loss_fig.tight_layout()
        loss_fig.savefig(root/'distillation_curves.png',dpi=150)
    plt.close(loss_fig)
    report+=['','![Training curves](training_curves.png)','','## 滚动回放与环境闭环','',
        '|实验|模式|回放归一化 MSE|return 均值 ± 样本标准差|平均 DiT 次数/chunk|',
        '|---|---|---:|---:|---:|']
    for run in suite['runs']:
        path=root/run['name']/'evaluation.json'
        if not path.exists():continue
        for mode,values in json.loads(path.read_text()).items():
            returns=values['environment_returns'];std=np.std(returns,ddof=1) if len(returns)>1 else 0.
            report.append(f"|{run['name']}|{mode}|{values['replay_normalized_mse']:.6f}|{np.mean(returns):.2f} ± {std:.2f}|{np.mean(values['dit_calls_per_chunk']):.3f}|")
    report+=['','注意：K 是 refinement 设置；rolling 缓存热态通常每 chunk 仅一次 DiT 调用，冷启动可能多次。',
        '相同 K 并非相同调用预算。BF16 指 autocast，参数和 AdamW 状态仍为 FP32，不等同完整 DeepSpeed 运行。',
        '各组采样完整窗口，验证按 episode 划分；原版 FM 保留 Beta/整数时间，rolling 保留原生随机阶梯时间。',
        '单种子、短程或小模型结果不能证明历史大模型崩溃根因，也不能宣称 LSD 达到收敛或优于 FM。','']
    if has_lsd_metrics:
        report+=['## 蒸馏指标','',
                 'LSD loss 是活跃 token 的条件均值；活跃比例和权重随训练日程变化。', '',
                 '![Distillation curves](distillation_curves.png)','']
    (root/'report.md').write_text('\n'.join(report))
    print(root/'report.md')


if __name__=='__main__':main()
