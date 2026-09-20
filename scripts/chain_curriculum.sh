#!/usr/bin/env bash
# After the controlled vs-brute experiment finishes, run the curriculum attempt at a
# human-constrained agent and benchmark it the same way.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
while [ ! -f runs/vs_brute_report.txt ]; do sleep 60; done
./scripts/curriculum_velocity.sh 2700 5400
{
  echo "=== curriculum velocity agent, $(date) ==="
  for stage in v1 v2; do
    [ -f "runs/curr_$stage.log" ] || continue
    echo "--- stage $stage"
    python3 -c "
import json
rows=[json.loads(l) for l in open('runs/curr_$stage.log') if l.startswith('{')]
ev=[r for r in rows if 'eval_win_rate' in r]
for r in ev: print(f\"    steps {r['env_steps']:>10,}  win {r['eval_win_rate']:.3f}  share {r['eval_share']:.3f}\")
"
  done
  if [ -f runs/curr_v2/policy.pt ]; then
    python3 -m fluxwar.deploy.export --config default.yaml --action-mode velocity \
            --policy runs/curr_v2/policy.pt --out build/curr_velocity.flxw
    echo "--- deployed, real LW6, opponents unconstrained:"
    python3 -m fluxwar.eval.deployed --weights build/curr_velocity.flxw --a nn \
            --against brute follow stationary --games 12 --rounds 600 --size 64 \
            2>/dev/null | grep "mod-nn" | sed 's/^/    /'
    echo "--- same, brute capped to the policy's own mobility (4 cells/round):"
    python3 -m fluxwar.eval.deployed --weights build/curr_velocity.flxw --a nn \
            --against brute --games 12 --rounds 600 --size 64 --cap 4 \
            2>/dev/null | grep "mod-nn" | sed 's/^/    /'
  fi
} > runs/curriculum_report.txt 2>&1
