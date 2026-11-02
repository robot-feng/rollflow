# Reacher H32/C4 three-arm screening

Run date: 2026-10-31. This is a fast, state-only diagnostic using the production FM and RollFlow action heads. It is not a production-sized policy benchmark.

## Setup

- Dataset: official `mujoco/reacher/expert-v0`; first 256 episodes selected, split by episode into 180 train / 38 validation / 38 held-out test episodes.
- State/action dimensions: 10 / 2. Action horizon 32, execution chunk 4.
- Three arms: original FM; rolling FM with LSD disabled; rolling FM + LSD.
- Small DiT: width 64, 2 layers, 4 heads, head dimension 16. Each arm trained for 20,000 updates with seed 42, CPU FP32, batch 8, repeat 4, learning rate `3e-4`.
- Rolling time sampling: `p_k1=0`, FM-only warmup 2,000 updates, then `p_fm` decays to 0.3. LSD arm uses `w_lsd=0.05`, 1,000-step LSD weight ramp, and `delta=0.01`. OT was disabled.
- All arms share initial weights (`7d064a5d…bd608`). Source revision at run time: `c7c720fff36d46bce559211667aec7887a9eb2c2`.
- Evaluation: K=1/2/4/8; 10 paired environment seeds 62000–62009; maximum 1,000 steps per episode. Replay MSE is measured on the 38 held-out dataset episodes.

## Results

Each cell is `held-out normalized replay MSE / mean environment return`. Higher (less negative) return is better. MSE is an action-fitting metric; environment return measures closed-loop behavior.

| Head | K=1 | K=2 | K=4 | K=8 |
|---|---:|---:|---:|---:|
| Original FM, instantaneous | 0.010563 / -4.3032 | 0.010368 / -4.0965 | 0.010833 / -4.2341 | 0.012612 / -4.6084 |
| Rolling FM, instantaneous | 0.011550 / -4.9717 | 0.010036 / -3.9257 | 0.010471 / -3.9757 | 0.012100 / -4.0544 |
| Rolling + LSD, instantaneous | 0.011465 / -4.8775 | 0.010011 / -3.9040 | 0.010417 / -3.9448 | 0.012073 / -4.0172 |
| Rolling + LSD, average | 0.015010 / -4.4916 | 0.017285 / -4.7406 | 0.018985 / -4.9643 | 0.018665 / -4.7635 |

## Reading the result

- **LSD did not show a reliable return gain over rolling FM at the same inference mode.** Paired mean return differences (LSD instantaneous minus rolling-FM instantaneous) were +0.094, +0.022, +0.031, and +0.037 for K=1/2/4/8. Every 95% paired bootstrap interval included zero. Positive differences indicate lower return here.
- **Average velocity hurt replay fit and closed-loop return at K=2/4/8.** Compared with the same LSD model using instantaneous velocity, paired mean return changes were -0.837, -1.020, and -0.746; their intervals excluded zero. K=1 was inconclusive. This small run does not support switching to average velocity just because LSD is enabled.
- Relative to original FM, rolling FM was worse at K=1, indistinguishable at K=2, and better at K=4/8 on these ten environment seeds. The result is suggestive only; the network and training budget are intentionally small.
- No arm clipped actions during these evaluations.

| Paired return difference (first minus second) | K=1, 95% CI | K=2, 95% CI | K=4, 95% CI | K=8, 95% CI |
|---|---:|---:|---:|---:|
| Rolling FM − original FM | -0.668 [-1.224, -0.168] | +0.171 [-0.090, +0.417] | +0.258 [+0.035, +0.495] | +0.554 [+0.283, +0.832] |
| LSD instant − rolling FM instant | +0.094 [-0.026, +0.202] | +0.022 [-0.022, +0.075] | +0.031 [-0.015, +0.094] | +0.037 [-0.010, +0.085] |
| LSD average − LSD instant | +0.386 [-0.252, +1.051] | -0.837 [-1.061, -0.623] | -1.020 [-1.268, -0.747] | -0.746 [-0.887, -0.576] |

The bootstrap resampled the ten paired environment seeds 200,000 times (seed 20261102). Its intervals represent evaluation-seed variation, not independent training variation: there is only one training seed per arm.

## Comparability limits

The original FM keeps its native Beta-distributed global time sampling. RollFlow uses grouped staircase times and an interval-time embedding, including when LSD is disabled. This is not an exact objective-only ablation. To identify the effect of LSD separately from the time/model-conditioning changes, compare rolling FM against rolling + LSD first.

Reacher episodes are only 50 steps, so H32 leaves few complete execution chunks per episode. Repeat the three arms with multiple training seeds and a longer task such as Hopper before making a cross-environment or production-scale claim. The aggregate metrics and paired-seed results are in the [machine-readable result](rollflow_minari_reacher_h32_c4_20261031.json); local checkpoints and the dataset are intentionally not included in Git.
