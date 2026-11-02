#!/usr/bin/env bash
# Evaluate explicit training milestones into separate directories.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../common.sh"
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    echo 'Set RUN_DIR, OUTPUT_DIR, and CHECKPOINT_STEPS (space-separated integers).'
    echo 'Additional arguments are forwarded to run.py eval, e.g. --dataset-root ... --eval-nfe 1 2 4 8.'
    exit 0
fi
: "${RUN_DIR:?Set RUN_DIR to the training run directory}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a new evaluation directory}"
: "${CHECKPOINT_STEPS:?Set CHECKPOINT_STEPS to space-separated step numbers}"
if [[ -e $OUTPUT_DIR ]]; then
    echo "Output already exists: $OUTPUT_DIR" >&2
    exit 2
fi
read -r -a steps <<< "$CHECKPOINT_STEPS"
if (( ${#steps[@]} == 0 )); then
    echo 'CHECKPOINT_STEPS must contain at least one step' >&2
    exit 2
fi
declare -A seen=()
for step in "${steps[@]}"; do
    if [[ ! $step =~ ^[0-9]+$ || ! -f $RUN_DIR/step_${step}.pt ]]; then
        echo "Missing or invalid checkpoint step: $step" >&2
        exit 2
    fi
    if [[ -n ${seen[$step]:-} ]]; then
        echo "Duplicate checkpoint step: $step" >&2
        exit 2
    fi
    seen[$step]=1
done
mkdir -p -- "$OUTPUT_DIR"
for step in "${steps[@]}"; do
    "$STARVLA_PYTHON" "$MINARI_EXAMPLE_DIR/run.py" eval \
        --checkpoint "$RUN_DIR/step_${step}.pt" --output "$OUTPUT_DIR/step_${step}" "$@"
done
