# What I would do next, and why

Ordered by expected value, with the evidence that points at each.

## 1. A health field — tried, and it did not work

This was my top-ranked hypothesis and it is wrong, at least in the obvious form.
`sim.track_health` is implemented and tested (mass conserved, health bounded) but
defaults to off, because on the matchup metric it scores **0.31 against 0.24 without
it** — measurably worse, across isotropy and rate sweeps.

The reasoning still looks right, which is what makes the negative result interesting.
The density sim over-rewards aggression: `chase_enemy vs hold_own` comes out 1.00 for
the attacker where the real kernel scores 0.43 and 0.14. With no health dimension,
**converted mass is immediately full strength**.

In LW6 a converted fighter respawns at `fighter_new_health = 5000` out of 10000 —
half — deep inside the enemy blob, surrounded by enemies, and is usually converted
straight back. That churn is what makes attacking into a packed defender expensive.
In the mean-field model a conversion is permanent the instant it happens, so a local
advantage compounds instead of churning.

What is implemented: `stock[B, 2, H, W]`, the mass-weighted health carried by the
advection with the same fractions and the same acceptance as the mass; combat removes
health stock rather than mass, and mass flips at the rate health crosses zero,
assuming health is uniform on `[0, 2h]` within a cell.

Why that probably is not enough: a *mean* health per cell cannot represent the effect
it was meant to capture. The churn in LW6 is individual — one fighter flips, is
surrounded, flips back — and the health distribution at a contested front is bimodal
(fresh arrivals and nearly-dead defenders), nothing like uniform around a mean. The
first moment throws away exactly the structure that matters.

Worth trying instead: two mass bands per team (fresh and damaged) with transitions
between them, which is the cheapest representation that can express bimodality; or
accepting that this is where a density model stops and the discrete kernel starts.

## 2. An absolute-target action, not a velocity

`mod-brute` teleports its cursor anywhere on the map each round. The policy emits a
velocity bounded by `MAX_CURSOR_SPEED`, and at equal mobility the two are about even
(0.44) while uncapped brute wins 0.96. Some of that gap is action space, not policy.

A spatial head — logits over the trunk's 8x8 feature grid, argmax to a cell — would
give the policy the same affordance. It makes the action discrete, so M3 loses its
gradient path for that head and PPO carries it; the velocity head can stay for the
differentiable arm.

## 3. Train against `mod-brute`

Neither trained policy has ever played it. `lw6.vecenv` now accepts
`opponent="brute"` and drives it natively inside the workers, but it runs at ~730
env-steps/s because brute forward-simulates a sandbox every round. Options: a curriculum
that spends most steps against fast opponents and a slice against brute, or distilling
brute into a cheap network first and training against the distillate.

## 4. Fix the reward's dead zone for M3

8 of 89 BPTT windows returned a gradient of exactly zero, because the reward — the
change in population share — is identically zero while the armies are out of contact.
PPO bootstraps across that; BPTT gets nothing. A shaping term that is nonzero before
contact (the geodesic distance between the two mass centroids, say) would give every
window signal. It changes the objective, so it belongs behind a flag and wants
measuring against the unshaped one.

## 5. Package `mod-nn` into a real LW6 build

The C bot is written and exercised against the real kernel; what is missing is the
`Makefile.am` and the `bot-register.c` entry, and a machine with SDL, guile, libcurl
and libjpeg to build the full game. See `docs/deployment.md`.

## Smaller things

* The curve-based and matchup-based fidelity metrics disagree about `attack_isotropy`.
  They should be combined into one objective rather than optimised separately.
* `sim/reference.py` is a 66%-divergent hand port kept for documentation. Either fix
  it against the real kernel or delete it; leaving a wrong reference around invites
  someone to trust it.
* The density sim lost 4.5x throughput to the exclusion fixed point. `capacity_iters`
  is the dial, and 2 is the current setting; a cheaper convergent scheme would buy
  most of it back.
