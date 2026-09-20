#!/usr/bin/env bash
# Wait for the two vs-brute trainings to finish, then export both and benchmark them
# inside real Liquid War 6 against an UNCONSTRAINED mod-brute -- the same harness and
# the same opponent settings that produced the 0.04 baseline.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
# Match on a pattern that cannot appear in the launching shell's own command line:
# using the literal argument string means pgrep finds the `bash -c` wrapper that
# created this script (its command line contains the heredoc) and waits forever.
while pgrep -f "policy[.]ppo" >/dev/null; do sleep 60; done
{
  echo "=== $(date) ==="
  for name in human bot; do
    mode=velocity; [ "$name" = bot ] && mode=target
    echo
    echo "### $name ($mode action space), trained vs unconstrained mod-brute"
    if [ ! -f "runs/vs_brute_$name/policy.pt" ]; then echo "  no checkpoint"; continue; fi
    python3 -c "
import json
rows=[json.loads(l) for l in open('runs/vs_brute_$name.log') if l.startswith('{')]
ev=[r for r in rows if 'eval_win_rate' in r]
print('  training curve (win rate / share vs brute):')
for r in ev: print(f\"    steps {r['env_steps']:>10,}  wall {r['wall']:7.0f}s  win {r['eval_win_rate']:.3f}  share {r['eval_share']:.3f}\")
"
    python3 -m fluxwar.deploy.export --config default.yaml --action-mode "$mode" \
            --policy "runs/vs_brute_$name/policy.pt" --out "build/vs_brute_$name.flxw"
    echo "  deployed, in real LW6, opponents unconstrained:"
    python3 -m fluxwar.eval.deployed --weights "build/vs_brute_$name.flxw" --a nn \
            --against brute follow stationary --games 12 --rounds 600 --size 64 \
            2>/dev/null | grep "mod-nn" | sed 's/^/    /'
  done
} > runs/vs_brute_report.txt 2>&1
