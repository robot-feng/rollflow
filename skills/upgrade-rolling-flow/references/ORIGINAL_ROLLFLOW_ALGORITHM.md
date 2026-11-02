# RollFlow 原始伪代码（不可静默修改）

下文完整摘自用户提供的 `rolling_meanflow.py` 模块文档字符串。**原样保存**；当发现路径、采样或实现存在可疑之处，应在旁边建立补充规范/测试，不覆盖原始算法。

```text
RollFlow: Flow Matching + Central-Difference Lagrangian Self-Distillation.

Core training algorithm
-----------------------
    s, t = sample_training_times()
    x_s = (1-s) * x0 + s * x1
    x_t = (1-t) * x0 + t * x1
    v_gt = x1 - x0

    # Build the stop-gradient teacher path for LSD rows first.
    with no_grad:
        V_st = model(x_s, s, t)
        X_t_hat = x_s + (t-s) * V_st
        v_teacher = model(X_t_hat, t, t)

    # Pack tangent endpoints for LSD rows and local FM for the full batch.
    V_minus, V_plus, v_local = model(
        cat(x_s, x_s, x_t),
        cat(s, s, t),
        cat(t-delta, t+delta, t),
    ).split(B_lsd, B_lsd, B)
    X_minus = x_s + (t-delta-s) * V_minus
    X_plus  = x_s + (t+delta-s) * V_plus
    v_tangent = (X_plus - X_minus) / (2*delta)

    loss_fm  = MSE(v_local, v_gt)
    loss_lsd = MSE(v_tangent, stopgrad(v_teacher))
    loss = w_fm * loss_fm + w_lsd * loss_lsd

Time convention
---------------
    x0 = Gaussian prior at flow time 0
    x1 = clean action trajectory at flow time 1
    s = source time
    t = terminal time
    X_{s,t}(z) = z + (t-s) * u_theta(z,s,t),  s <= t

Time sampling
-------------
    Input: horizon H, chunk size C, probabilities p_k1, p_fm

    1. Temporal refinement:
        M = H / C, K ∈ Div(M) = {k ∈ N : k divides M}
        P(K=1)=p_k1, P(K>1)=(1-p_k1)/(#K-1)
        Sample K

    When M is a power of two, the default candidate set is
    Div(M) = {1, 2, 4, ..., M}, i.e. powers of two in a geometric sequence
    with common ratio 2 (for example, M=8 gives {1, 2, 4, 8}). For other
    values of M, the sampler uses all divisors of M.

    2. Ordered time sampling:
        Sample: t_i ~ U(0,1)
        Sort: t_1 > t_2 > ... > t_K

    3. Interval sampling:
        Sample: ratio ~ p_fm·δ(1)+(1-p_fm)·U(0,1)

    4. Construct:
        s_i = ratio * t_i

Training samples K from the divisors of M=H/C. K=1 is favoured, while the
remaining choices share the residual probability. K independently sorted target
times are expanded over equal action blocks, and one source ratio per sample gives
s=ratio*t. An FM-only curriculum starts with s=t and anneals to the configured
mixture probability. Specifically, ``fm_curriculum_steps`` keeps p_fm=1, then
p_fm decays linearly to its floor over the remaining ``total_train_steps``.
Deployment still uses the exact rolling staircase needed for cache alignment.

There is no JVP, CSF, Split loss, persistent training cache, or adjacent-window
training sampler. Rolling state exists only at inference.

```
