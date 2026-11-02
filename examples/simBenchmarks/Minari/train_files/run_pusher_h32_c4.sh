#!/usr/bin/env bash
# Pusher 1/2/4/8 comparison; pass additional runner arguments at the end.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../common.sh"
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    echo 'Set OUTPUT_DIR and MINARI_DATASET_ROOT, then run this script; extra arguments are forwarded to run_suite.py.'
    echo 'Optional: STARVLA_PYTHON, TRAIN_STEPS, FM_WARMUP_STEPS, LSD_RAMP_STEPS, TRAIN_SEED, BATCH_SIZE, REPEAT, THREADS.'
    exit 0
fi
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a new experiment directory}"
: "${MINARI_DATASET_ROOT:?Set MINARI_DATASET_ROOT to the Minari dataset root}"
train_steps=${TRAIN_STEPS:-45000}
if [[ ! $train_steps =~ ^[0-9]+$ || $train_steps -lt 3 ]]; then
    echo 'TRAIN_STEPS must be an integer >= 3' >&2
    exit 2
fi
fm_warmup=${FM_WARMUP_STEPS:-$((train_steps * 2 / 3))}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
exec "$STARVLA_PYTHON" "$MINARI_EXAMPLE_DIR/run_suite.py" \
    --output "$OUTPUT_DIR" --seeds "${TRAIN_SEED:-42}" --precisions fp32 --variants fm rolling_lsd \
    --dataset-root "$MINARI_DATASET_ROOT" --dataset mujoco/pusher/expert-v0 \
    --max-episodes 200 --horizon 32 --chunk 4 --nfe 8 \
    --steps "$train_steps" --fm-warmup "$fm_warmup" --lsd-ramp "${LSD_RAMP_STEPS:-1000}" \
    --batch "${BATCH_SIZE:-8}" --repeat "${REPEAT:-4}" --threads "${THREADS:-2}" \
    --width 64 --layers 2 --log-every 250 --save-every 2000 \
    --eval-episodes 10 --eval-max-steps 100 --eval-nfe 1 2 4 8 "$@"
