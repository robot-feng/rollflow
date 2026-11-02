#!/usr/bin/env bash
# Shared paths for the public Minari launchers. Set STARVLA_PYTHON to an
# environment containing starVLA, Minari and the selected simulator.
MINARI_EXAMPLE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
STARVLA_PYTHON=${STARVLA_PYTHON:-python}
