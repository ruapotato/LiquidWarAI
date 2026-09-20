# Two cursor action spaces

Liquid War gives the player exactly one control, the cursor — so how the cursor may
move *is* the action space, and LW6's own bots do not play under the same one a human
does.

| mode | what the policy outputs | what the cursor does |
| --- | --- | --- |
| `velocity` | a direction in [-1, 1]² | moves at most `MAX_CURSOR_SPEED` cells this round |
| `target` | a position in [0, 1]² | goes there |

`velocity` is the human constraint: a hand moves a mouse at a finite speed. The
default of 4 cells per round is roughly what that is. LW6 runs at 50 rounds/s, so a
player sweeping the cursor across a 64-cell-wide board in a third of a second covers
64 / (0.33 x 50) ~ 4 cells per round. Two to five is the human band; one cell per
round, the original default here, is about ten times slower than a hand and left half
of all games with no contact at all.

So the comparison has a clean reading: if the velocity agent cannot beat `mod-brute`,
the honest conclusion is not that the policy is weak but that **LW6's strongest
shipped bot plays at superhuman cursor speed** -- it repositions instantly, every
round, fifty times a second.
`target` is the bot constraint, and it is not a hypothetical — it is what
`mod-brute` does, literally:

```c
cursor.pos.x = lw6sys_random (sys_context, shape.w);
cursor.pos.y = lw6sys_random (sys_context, shape.h);
```

anywhere on the map, every round, chosen by forward-simulating a sandbox copy of the
game. The LW6 kernel does not limit cursor speed at all, so nothing stops it.

That difference is most of the gap. A velocity policy against an uncapped brute takes
0.04 of the population; cap brute to the same one cell per round and the same policy
is level with it at 0.44. Whether "our bot beats the default bots" therefore depends
entirely on which action space you think is fair, which is why both are built.

## The ceiling for a bounded cursor

Measured directly rather than inferred from the trained agents, and reproducible:

```bash
python -m fluxwar.eval.ceiling            # capped velocity policies vs uncapped brute
```

Hand-written velocity policies, capped to 4 cells per round, against an
unconstrained `mod-brute` (it drives its own cursor straight into the kernel; only
the policy is capped). Two runs at different seed counts and episode lengths:

| policy | 4 seeds / 500 rounds | 3 seeds / 400 rounds |
| --- | --- | --- |
| `chase_enemy` | 0.003 | 0.011 |
| `hold_own` | 0.005 | 0.063 |
| `lean2` | 0.007 | 0.024 |
| `lean5` | 0.076 | 0.131 |

Call it **roughly 0.01 to 0.13**, and note how unstable that is: `lean5` nearly
doubled between runs, and both of its means are carried by a single map (0.286 and
0.315 respectively, with the rest near 0.01). Any single number here is a bad
summary, and a trained agent landing at 0.05 cannot be called better or worse than
the scripted band on this evidence.

The point stands regardless of the exact figure: against a searcher that repositions
instantly, a speed-limited cursor gets a small and erratic share whatever drives it.
That band is what a trained velocity agent should be compared against, and it is
easy to get wrong -- quoting `mod-follow`'s 0.079 as the baseline is not like for
like, because in the deployed harness mod-follow also returns an absolute position
each round and is effectively unconstrained too.

A flat reward landscape and a policy failing to learn look identical from the
training curves -- flat entropy, negative reward -- so measuring the band is what
tells the two apart, and they call for opposite responses.

## How it is implemented

`policy/net.py` carries both heads and exposes `sample` / `evaluate_actions` /
`act_deterministic`, so PPO never branches on the mode.

* **velocity**: a Gaussian over a 2-vector, tanh-squashed. Differentiable, so M3 can
  use it.
* **target**: a 1×1 convolution on the trunk's 16×16 feature map gives one logit per
  4×4 block of the board, and the action is a categorical draw. Discrete, so there is
  no gradient path — the target agent is PPO-only.

The environments (`sim/env.py`, `lw6/env.py`) read `sim.action_mode` and interpret the
action accordingly. The exported weight file (format v3) records the mode and the
cursor speed, and `mod_nn.c` implements both, so a deployed bot behaves the way it
was trained.

## Training against an unconstrained opponent

```bash
./scripts/train_vs_brute.sh 10800
```

trains both agents against `mod-brute` with no cap on *its* cursor. The opponent runs
natively inside each worker process, because routing it through the parent as a
velocity is exactly the cap we are trying not to apply.

One trap, worth knowing about because it fails silently and flatteringly:
`LW6Env.step` writes cursors for both teams from the action array, which overwrites
whatever a native bot just chose. A policy "trained against mod-brute" reached a 1.0
win rate in 156k steps that way, with brute's cursor being parked every round.
`LW6Env.external_teams` marks the teams the env must not touch, and
`test_action_modes.py::test_native_opponent_is_not_overwritten_by_the_action` checks
that an idle learner still gets destroyed.
