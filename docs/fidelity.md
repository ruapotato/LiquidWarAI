# Measuring the density sim against the real kernel

## The scenario

`eval/fidelity.py` takes a map from this project's generator, injects it into a real
LW6 game so both engines play the identical board, reads the kernel's own fighter
placement back out, and replays it in the density sim. Both are then driven by the
same cursor script: team 0 holds its start, team 1 walks onto it.

Three things about this were learned the hard way:

* **Both cursors on the same cell tells you nothing.** Both armies relax onto the same
  attractor, the two density fields become identical, and combat is exactly symmetric
  whatever the rule is. Every candidate scores a flat 0.5 and all look equally good.
* **A symmetric start with a symmetric script has the same problem.**
* **`lw6map_builtin_custom` returns the same board every time.** `noise_percent`
  changes the texture, not the walls. A sweep over four "scenarios" was a sweep over
  one, and reported four identical numbers before anyone noticed.

## The metric

Maximum absolute difference between the two population curves, divided by the
reference's own maximum departure from 0.5. Comparing raw values would flatter any
pair of sims, because both curves start at exactly 0.5 by construction. Averaged over
several maps.

LW6 is deterministic — it has to be, for network play — so there is no noise floor
under this number and one run per map is enough. An earlier, hand-rolled stochastic
reference had a seed-to-seed spread of 0.077 against a signal of 0.09, which made the
spec's 10% bar strictly meaningless: 10% of the signal was smaller than the
reference's own standard error. Worth checking before trusting any agreement number.

## Where it stands

About **0.32 relative error**, against the spec's 10% bar. The two engines agree on
the winner and roughly on the margin; the residual is the shape of the transient. The
tests assert the achieved bound as a regression guard and say so.

The most useful thing to know is that the best configuration found by sweeping is the
*faithful* one: `acts_per_tick = 2` and `RATE_DT = 0.05` are LW6's own
`moves_per_round` and `fighter_attack / max_fighter_health`. The free parameters did
not want to move away from the values the source dictates, which is the outcome you
want from a tuning sweep over a physical model.

## Cursor speed is a design parameter, and 1 cell/round was wrong

`MAX_CURSOR_SPEED` is not an LW6 rule — the kernel does not limit the cursor at all,
which is how `mod-brute` gets away with teleporting. It is a property of the
controller, and the original default of one cell per round is far slower than a hand:
LW6 runs at 50 rounds/s, so one cell per round is 50 cells per second across a map
hundreds of cells wide.

It showed up as dead games. Measured in the real kernel over scripted matchups, the
fraction ending within 0.02 of a draw:

| cells/round | near-draws | matchup fidelity |
| --- | --- | --- |
| 1 | 0.50 | 0.242 |
| 4 | 0.38 | 0.234 |
| 8 | 0.12 | 0.294 |

Half of all games producing no contact is half the training signal thrown away, and
it also blunts every evaluation: a metric that returns 0.500 for both a good and a
bad policy cannot rank them. 4 is the current default — it halves the dead games at
no cost in fidelity, where 8 buys more signal but starts to disagree with the kernel.

The deployed bot has to agree about this. The exported weight file carries the cursor
speed the policy trained with (format v2) and `mod_nn.c` scales by it; a policy
trained at 4 cells/round and deployed assuming 1 moves a quarter as far as it means
to.

## What is free to tune

`RATE_DT`, `DEFEND_DT`, `REGEN_DT` and `capacity` are **not** free — they are LW6's
rule constants divided by `max_fighter_health`, and one fighter per cell. Moving them
means the two engines are playing different games.

Free is the numerics of the continuous approximation:

* `temp` — softmax temperature over the 9 moves. Small approaches "take the first free
  cell in the fan"; large is smoother and better conditioned for M3. Tuning against
  the real kernel prefers ~1.0, which is convenient: the fidelity optimum and the
  differentiability optimum point the same way, where tuning against the hand port had
  wanted 0.1.
* `SPEED` — fraction of a cell's mass that moves per tick. Must be in [0, 1]: above 1
  the cell keeps `density * (1 - SPEED) < 0` and the sim NaNs a few ticks later with
  no other warning. `Params.__post_init__` refuses it.
* `capacity_iters`, `crowd_sharpness` — the exclusion fixed point. More iterations is
  more faithful and linearly slower; this is the main throughput cost.
* `K_RELAX` — Jacobi passes on the potential field per tick. Measured to make almost
  no difference above 8.

## Gotchas found the hard way

* **Finite-difference gradient checks.** In float32 the score is ~0.5 and an FD step
  moves it by ~1e-8, below the representable resolution — the numeric derivative comes
  back quantised to powers of two. Use float64. Even then, at low `temp` the softmax is
  nearly a step function, its curvature goes as `1/temp**2`, and the central difference
  carries a truncation error of order `eps**2 / temp**3`. The analytical value is the
  accurate one.
* **Tuning against a buggy sim.** Two sweeps here were fitting parameters that were
  partly compensating for mass loss in the advection, and the fitted optimum moved
  once it was fixed. Two more were fitting against the hand port rather than the real
  kernel. If a fidelity fit and a conservation test are both outstanding, fix
  conservation first — and make sure the reference is the reference.
