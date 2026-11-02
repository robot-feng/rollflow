# Verification report — RollFlow Skill v0.5

## Executed in this deliverable

- `python -m pytest -q tests`: **40 passed** (CPU, PyTorch 2.10.0+cpu, Python 3.13, pytest 9.0.2).
- `python tools/check_originals.py`: 3 exact user source files and the original docstring matched the SHA256/byte manifest.
- Original `rolling_meanflow.py` is directly imported and its sampler, `central_difference_lsd` training loss, full public `.loss` and cold/warm `.step` are actually called.
- An independent adapter example is actually used; original policy callback is not mutated.
- Reference instant buffer is tested in both flow directions; optional radial LSD field gradient/teacher scope tested in both directions; these are separate from the original absolute-FD algorithm.
- Prefix/RTC-shape helpers, explicit time ownership for auxiliary tokens, output mask, no-grad teacher and zero-interval branch invariants are tested.

## Key source-specific findings, NOT silently corrected

1. Original LSD groups sample divisors of **H/C**; instant fine-grain groups sample divisors of **H**. H64/C16 yields {1,2,4} vs {1,2,4,8,16,32,64}.
2. Original LSD uses absolute `t±δ` and a packed student call; reference straight-path LSD uses `t±h(t-s)` and requires an explicit path contract for coupled action tokens.
3. Original DiT defaults `skip_zero_interval_embedding=False`; exact `r=t` equivalence with native scalar model is not implied. Source-specific future adaptation should use a guaranteed zero interval branch and test its output & old-weight gradients.
4. The supplied config defaults disagree (`p_fm` 0.7 in core versus 0.3 in the StarVLA wrapper), so do not silently unify them.
5. Original `zero_padded` imposes a dynamics mask on padded tokens; for other models loss-only masks must not be turned into dynamics masks.
6. Original `iterative_cold_start=False` is the original average-mode default. The current instant default cold K/warm 1 is a distinct configured mode.

## Not executed / cannot be certified from these attachments alone

- Full StarVLA action-head model forward/backward: its dependencies `diffusers` and `transformers`, configs, full repository and weights are not installed in this CPU sandbox.
- A universal automatic conversion for every XPolicy VLA strategy, and full XPolicy training, GPU peak VRAM, true task success or robot closed-loop.
- Whether LSD improves 1/2 NFE quality for a *specific* production checkpoint/dataset. Must be evaluated against native fresh and matched-compute baselines.
- Combined real-time chunking, action prefill and asynchronous rolling execution. The included `token_time_torch.py` exercises the prefix math, not the full RTC runtime.

This file is a **portable draft Skill**, not an installation in the protected system skill directory. The host agent must follow `SKILL.md` and prove target-specific invariants before making a deployment claim.
