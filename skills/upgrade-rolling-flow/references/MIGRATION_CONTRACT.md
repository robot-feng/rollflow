# RollFlow migration contract and independent comparison checklist

The user's original code and pseudocode are immutable references. This document supplements, never edits, them.

## 1. Semantic invariants

| Check | Must be demonstrated |
|---|---|
| Flow convention | Native `noise_at_zero` uses `x_t=(1-t)noise+t*actions`, `v=actions-noise`, integrate increasing t. Native `noise_at_one` uses `x_t=t*noise+(1-t)actions`, `v=noise-actions`, integrate decreasing t. A sign flip cannot be skipped. |
| Time shape | `[B]`, `[B,H]` or explicit `[B,H,1]` accepted only at declared interface; internal attention remains over H; all modulation layers and terminal layer receive correct time. |
| Position and time | Action horizon position indices and noise times never share an implicit axis. Nonaction prefix tokens use their original semantics; they are not silently assigned the first action noise time when that changes computation. |
| Original LSD group count | `M=H/C`; `K∈Div(M)` with `P(K=1)=p_k1`. Do NOT replace with `Div(H)` or a new distribution in original mode. |
| Instant grouped count | `G∈Div(H)` with native time sampler; this is a later optional mode, not what the original LSD pseudocode states. |
| Diagonal identity | `model(x,t,t)` must have no new nonzero interval contribution; test both output and old-weight gradient equality before training, at every supported time shape. `s=t` means whole time vector unless explicitly stated otherwise. |
| Original LSD | Exactly teacher `V_st -> X_t_hat -> v_teacher` under no_grad; student `V_minus`, `V_plus`, `v_local` in that order; `delta` central FD; original native FM anchor and reduction retained. |
| Source-specific optional LSD | Original **absolute** endpoint perturbation `t±δ` and directional **straight-path** `t±h(t-s)` are distinct; keep names, derivations and comparison runs separate, never report as identical. For coupled action tokens the directions generally give different tangents. |
| Mask | `loss_valid` and `dynamics_present` are independent. Loss-masked-but-present tokens must remain available to attention as original model expects. |
| RNG/dropout | LSD tangent/student and stopped teacher must have deterministic field evaluations; no-grad alone does not disable dropout. Preserve original time sampler dtype/RNG stream for baseline tests. |
| Cache | Save latent only, not suffix hidden states; exactly C confirmed actions consumed per warm step; refresh context prefix and no suffix KV leakage; no concurrent/shared stream cache; failures must leave old cache unchanged. |
| OT and prior | Preserve native OT coupling and joint noise distribution; independent fresh tail is invalid if prior correlations across time must be maintained. |
| RTC / prefill | RTC is an independent feature. Training has clean prefix, correct time masks and loss exclusions; combined RTC+rolling must be separately proven before exposure. |
| Original model | Same files/checkpoint/registry still usable; separate algorithm name and config; rollback by selecting baseline. |

## 2. Mandatory comparison experiment identities

- `baseline`: original time training + original sampler.
- `grouped_fresh`: grouped-time training + same fresh sampler/inference compute budget.
- `instant_rolling`: same grouped checkpoint + retained rolling latent.
- `lsd_original`: original *absolute-step* LSD objective, with original sampling recipe and separate interval schema.
- `lsd_straight_path_experimental`: optional explicitly different target; never mislabeled as `lsd_original`.

Control: action normalization, dataset split, backbone, optimizer schedule, batch, RNG seeds, training tokens and reported total forward FLOPs. Evaluate 1/2/4+ invocations, cold and multiple warm windows, sudden goal/observation changes, smoothness and control success where available.

## 3. Test evidence levels

- `A`: AST/source inspection, no execution.
- `B`: selected source function/encoder with synthetic inputs.
- `C`: actual small native action head forward/backward, save/reload, cache behavior.
- `D`: actual production weights/data under native stack.
- `E`: closed-loop simulator.
- `F`: physical robot.

Reports must state the exact level per algorithm branch, CPU/GPU version, commit, pass/fail, missing dependency and hash of tested patch. No blanket "all strategies passed" for a repository merely because registration/shape/component checks passed.

## 4. Known source-specific pitfalls

- `rolling_meanflow.py` from the user takes `M=H/C` to enumerate `K`; the prior instant kernel enumerates `Div(H)` for G. Both are deliberately retained.
- `RollFlowActionHeadConfig` example's `p_fm` default differs from the `RollFlowConfig` used inside core. Resolve configuration from actual running instance; no silent unification.
- Original `cross_attention_dit.py` has `skip_zero_interval_embedding=False`: since `TimestepEncoder(0)` can be nonzero, `start_timestep=timestep` is not automatically equal to the native single-time computation. To get strict backward compatibility, enforce a truly zero interval contribution and test it.
- The original source `zero_padded()` removes padded tokens from attention input. Do not adopt that behavior blindly when a target policy has loss-only masked tokens that still influence other positions.
- The original source has a central difference in a vector time landscape by perturbing active dimensions with **absolute δ**, not a radial path fraction. Off-diagonal attention coupling means changing this stencil can change the mathematical objective.
- A source that only supports scalar time cannot be made *jointly token-time correct* by re-running the entire action expert once per group and slicing predictions: this alters the conditioning of other positions and increases effective model evaluation count. Such a fallback is a different algorithm and should be labeled as such.
