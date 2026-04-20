#!/bin/bash
# MuJoCo シミュレーターを起動するラッパー

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXTERN="$REPO_ROOT/extern/GR00T-WholeBodyControl"

source "$EXTERN/.venv_sim/bin/activate"
cd "$EXTERN"

python gear_sonic/scripts/run_sim_loop.py \
    --enable-offscreen \
    --enable-image-publish \
    "$@"
