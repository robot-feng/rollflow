#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../common.sh"
exec "$STARVLA_PYTHON" "$MINARI_EXAMPLE_DIR/run.py" eval "$@"
