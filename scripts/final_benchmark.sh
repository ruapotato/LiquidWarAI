#!/usr/bin/env bash
# Wait for runs/ppo_final to finish training, then export the policy and benchmark it
# inside real Liquid War 6 the way the game drives a bot. Writes runs/final_benchmark.txt.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
while pgrep -f "policy[.]ppo --config default.yaml" >/dev/null; do sleep 60; done
{
  echo "=== $(date) ==="
  echo "--- training curve"
  python3 -c "
import json
rows=[json.loads(l) for l in open('runs/ppo_final.log') if l.startswith('{')]
ev=[r for r in rows if 'eval_win_rate' in r]
for r in ev: print(f\"  steps {r['env_steps']:>10,}  wall {r['wall']:7.0f}s  win {r['eval_win_rate']:.3f}  share {r['eval_share']:.3f}\")
"
  echo "--- export"
  python3 -m fluxwar.deploy.export --config default.yaml --policy runs/ppo_final/policy.pt \
          --out build/policy_final.flxw
  echo "--- in real Liquid War 6, bots driving cursors as the game does"
  for cap in 4 8 999; do
    echo "  opponents capped at $cap cells/round:"
    python3 -m fluxwar.eval.deployed --weights build/policy_final.flxw --a nn \
            --against brute follow stationary --games 12 --rounds 600 --size 64 \
            --cap "$cap" 2>/dev/null | grep "mod-nn"
  done
} > runs/final_benchmark.txt 2>&1
