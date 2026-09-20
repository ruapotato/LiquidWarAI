#!/usr/bin/env bash
# Two agents, same opponent, different cursor freedom.
#
#   human : bounded velocity, MAX_CURSOR_SPEED cells per round -- what a hand can do
#   bot   : absolute target, cursor placed anywhere -- what LW6's mod-brute does
#
# Both train against mod-brute with no constraint on *its* cursor, driven natively
# inside the worker processes (routing it through the parent as a velocity would cap
# the strongest opponent available to one cell per round).
set -uo pipefail
BUDGET="${1:-10800}"
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"

python3 -u -m fluxwar.policy.ppo --config default.yaml --env lw6 --opponent brute \
        --action-mode velocity --num-envs 48 --out runs/vs_brute_human \
        --max-seconds "$BUDGET" > runs/vs_brute_human.log 2>&1 &
A=$!
python3 -u -m fluxwar.policy.ppo --config default.yaml --env lw6 --opponent brute \
        --action-mode target --num-envs 48 --out runs/vs_brute_bot \
        --max-seconds "$BUDGET" > runs/vs_brute_bot.log 2>&1 &
B=$!
wait $A $B
echo "training done $(date)"
for name in human bot; do
  if [ ! -f "runs/vs_brute_$name/policy.pt" ]; then
    echo "no checkpoint for $name -- training did not get far enough; not writing a report"
    exit 1
  fi
done

{
  echo "=== $(date) ==="
  for name in human bot; do
    mode=velocity; [ "$name" = bot ] && mode=target
    echo "--- $name ($mode), trained against unconstrained mod-brute"
    python3 -c "
import json
rows=[json.loads(l) for l in open('runs/vs_brute_$name.log') if l.startswith('{')]
ev=[r for r in rows if 'eval_win_rate' in r]
for r in ev[-12:]: print(f\"  steps {r['env_steps']:>10,}  wall {r['wall']:7.0f}s  win {r['eval_win_rate']:.3f}  share {r['eval_share']:.3f}\")
"
    python3 -m fluxwar.deploy.export --config default.yaml \
            --policy runs/vs_brute_$name/policy.pt --out build/vs_brute_$name.flxw
  done
} > runs/vs_brute_report.txt 2>&1
echo "report written"
