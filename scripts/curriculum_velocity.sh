#!/usr/bin/env bash
# A second attempt at the human-constrained agent, with a curriculum.
#
# Trained directly against an unconstrained mod-brute, the velocity agent never got a
# foothold: entropy barely moved, reward stayed negative, and its share sat below
# where mod-follow already is. Against an opponent that much stronger the reward is
# dominated by the opponent's choices and there is nothing to climb.
#
# So: learn against mod-follow first, then warm-start into mod-brute. It still ends
# against the unconstrained bot, which is the comparison that matters.
set -uo pipefail
STAGE1="${1:-2700}"
STAGE2="${2:-5400}"
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
python3 -u -m fluxwar.policy.ppo --config default.yaml --env lw6 --opponent follow \
        --action-mode velocity --num-envs 96 --out runs/curr_v1 \
        --max-seconds "$STAGE1" > runs/curr_v1.log 2>&1
python3 -u -m fluxwar.policy.ppo --config default.yaml --env lw6 --opponent brute \
        --action-mode velocity --num-envs 96 --out runs/curr_v2 \
        --init runs/curr_v1/policy.pt --max-seconds "$STAGE2" > runs/curr_v2.log 2>&1
