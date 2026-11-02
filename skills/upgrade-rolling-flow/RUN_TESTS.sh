#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python scripts/run_checks.py "$@"
