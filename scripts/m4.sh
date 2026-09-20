#!/usr/bin/env bash
# M4: PPO and BPTT on the same simulator, the same config and the same wall-clock
# budget, so the comparison is like for like. Both write history.json, which
# fluxwar.eval.compare turns into win rate against environment steps and against
# wall clock.
set -euo pipefail
BUDGET="${1:-2400}"
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
python3 -u -m fluxwar.policy.ppo  --config soft.yaml --env density \
        --out runs/m4_ppo  --max-seconds "$BUDGET" > runs/m4_ppo.log  2>&1 &
PPO=$!
python3 -u -m fluxwar.policy.bptt --config soft.yaml \
        --out runs/m4_bptt --max-seconds "$BUDGET" > runs/m4_bptt.log 2>&1 &
BPTT=$!
wait $PPO $BPTT
python3 -u -m fluxwar.eval.compare --config soft.yaml --ppo runs/m4_ppo \
        --bptt runs/m4_bptt --games 64 | tee runs/m4_report.txt
