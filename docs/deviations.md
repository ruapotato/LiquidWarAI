# Relationship to Liquid War 6

## Provenance and licence

`fluxwar/lw6/` compiles and links the GNU Liquid War 6 kernel by Christian Mauduit.
`fluxwar/sim/reference.py` is a hand port of the same code. The project was originally
specified clean-room; that was abandoned deliberately, because the mechanic could not
be reproduced faithfully from prose — the three errors below were all in the
clean-room version and all of them changed who wins. This repository is a derivative
work, licensed AGPLv3.

The real kernel is the ground truth. The density sim (`fluxwar/sim/`) is a continuous,
differentiable approximation of it, and every difference below is either a mean-field
approximation of something the kernel does exactly, or a numerical necessity.

## Three things that are easy to get wrong

1. **A fighter moves if it can, and fights only when it is blocked.** The per-act
   order is move → attack → defend → regenerate. Attack-first gives a rigid battle
   line; move-first is what makes armies behave like a liquid. It also inverts the
   outcome: with attack-first, charging the enemy was suicide (5% win rate); with the
   real order the charging army flows around the defender and comes out ahead.

2. **Attacks are directional, and movement uses a fan.** A fighter swings in the
   direction its cursor is pulling it, at whatever is in the way, after trying the
   first `nb_move_tries = 5` directions of an ordered fan around its preference. So an
   army that has enveloped another attacks inward while the enveloped army's own fan
   also points inward — at its own allies, whom it *heals*. That asymmetry is the
   whole reason aggression works in this game, and an isotropic neighbour-count
   pressure cannot express it: with one, the density sim had the defender winning a
   matchup the kernel decides the other way, ~90% curve error however it was tuned.
   LW6 resolves 12 directions on an 8-neighbour grid, so the fan has finer angular
   resolution than the set of reachable cells.

3. **The gradient is max-plus, monotone, and the source ramps.** Higher potential
   means closer to the cursor, and the field is relaxed by fast sweeps that carry it
   across the map in one pass. A monotone relaxation on its own would mean the well
   dug by a previous cursor position never fills back in, and the army keeps flowing
   to where the cursor used to be — observed directly here, cursor at (62, 1) and army
   settling at (1.5, 61.5). LW6 fixes it not by changing the relaxation but by raising
   the cursor's own potential every round, so old wells stay put while the live one
   climbs past them. The field has a deliberate, decaying memory of where the cursor
   has been.

Also: **cursors may sit on walls.** LW6 applies them at the nearest free slot. Blocking
a cursor on walls is unfaithful and catastrophic — a straight-line chaser wedges in
the first concave corner and never moves again, which turned 81% of games into
no-contact draws.

Rule constants come from `map.h`: health 10000, attack 500, side attack 20%, converted
fighters respawn at 5000, regenerate 5/act, defense 50, 5/3/1 move/attack/defense
tries, 2 moves and 5 spreads per round.

## How the density sim approximates it

**Combat** (`core.combat`) is the mean-field limit of the LW6 attack rule. Attacking
strength on a cell is the *blocked* enemy mass that tried to move into it — `advect`
returns that directionally, and only that mass fights. `attack_isotropy` blends in a
component that spreads a cell's blocked mass over all eight neighbours: purely
directional, a defender facing its own cursor cannot fight back at all, and a sweep
against the real kernel prefers a mix (0.6) over either extreme. That matches LW6,
where a blocked fighter tries three of its twelve fan directions for an enemy. The rate
`RATE_DT = fighter_attack / max_fighter_health = 0.05` is LW6's, not a free parameter,
and the defence and regeneration terms come from the same table. Saturating through
`1 - exp(-x)` bounds the loss by the mass present so densities stay non-negative
without a clamp, which would put a kink in the gradient exactly where M3 needs it
smooth. Concentration pays without an exponent bolted on: a cell can be pressed into
from eight directions and has only its own mass to lose.

**Exclusion.** LW6 holds one fighter per cell. `core.advect` enforces this as a real
constraint through a fixed-point iteration — see its docstring for the three failure
modes that forced each part. A soft preference cannot refuse mass: with one, an army
of 300 collapsed onto two cells at density 150 each, with total mass perfectly
conserved the whole way down.

**The move fan** becomes a softmax over the 9 moves at temperature `temp`, with
acceptance folded back in so refused mass is re-offered elsewhere. Without that an
army flows as a one-cell filament instead of advancing as a body.

**The gradient** is relaxed with `K_RELAX` Jacobi passes per tick rather than LW6's
sequential sweeps, which do not vectorise across a batch on a GPU. The cursor seed is
a euclidean cone rather than a single cell: a hard index has exactly zero gradient
with respect to the cursor, which would silently kill M3 at its first line
(`tests/test_gradients.py`).

**Health** has no separate dimension; conversion is continuous at the mean rate. This
is the model's main remaining error and it is one-sided: in LW6 a converted fighter
respawns at half health inside the enemy blob and is usually converted straight back,
which is what makes attacking into a packed defender expensive. With no health to
carry, a conversion here is permanent the instant it happens, so the density sim
over-rewards aggression — it scores `chase_enemy vs hold_own` at 1.00 for the attacker
where the kernel scores 0.43. `docs/next-steps.md` sketches the fix. The renderer
shows the equivalent of health — hue is ownership, brightness is how contested a cell
is — so liquid in contact darkens and comes back up in the other colour.

## Departures from the original project spec

The spec's combat rule, `net = RATE * (dB * pA - dA * pB)`, cannot move the score at
all. It is antisymmetric pointwise, but summed over the grid both terms enumerate the
same adjacent (A, B) pairs, so the total transfer is identically zero — measured drift
on a random state is ~1e-5, float noise. Kept verbatim in `sim/dynamics.py` and pinned
by `tests/test_conservation.py::test_spec_combat_is_degenerate`.

## The Python port

`sim/reference.py` ports the kernel by hand and is useful as readable documentation of
the algorithm, but it is **not** the ground truth: `eval/validate_port.py` measures it
against the real kernel and finds a 66% relative divergence on the standard scenario
(0.46 against 0.38). Once the real kernel was building there was no reason to keep
debugging a re-implementation that is also 280x slower.
