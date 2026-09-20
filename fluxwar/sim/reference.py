"""M0: a faithful port of the Liquid War 6 game kernel. numpy, unbatched, slow.

Ported from the GPLv3 sources of GNU Liquid War 6 by Christian Mauduit
(src/lib/ker/ker-move.c, ker-spread.c, ker-fighter.c, ker-mapstate.c, ker-team.c,
ker-tables.c, and the rule defaults in src/lib/map/map.h). This file therefore makes
the project a derivative work; see LICENSE.

This exists as ground truth for the density sim. It is deliberately not optimised.

Three things here are easy to get wrong from a prose description of the game, and all
three change how it plays:

1.  A fighter *moves if it can* and only fights when it is blocked. The order is
    move -> attack -> defend -> regenerate, not attack-first. This is what makes armies
    behave like a liquid rather than like a battle line.

2.  Movement is not "the single best neighbour". Each fighter has an ordered fan of up
    to `nb_move_tries` directions around its preferred one and takes the first free
    cell in that fan, which is how a packed army flows around obstacles and around
    itself. LW6 resolves 12 directions on an 8-neighbour grid, so the fan has finer
    angular resolution than the set of reachable cells; `MOVE_FAN` is LW6's table.

3.  The gradient is max-plus (higher potential = closer to the cursor) and monotonically
    increasing, which on its own would mean the well dug by a previous cursor position
    never fills back in. LW6 fixes that not by changing the relaxation but by raising
    the cursor's own potential by `round_delta` every round: old wells stay put while
    the live one climbs past them, so the field has a deliberate, decaying memory of
    where the cursor has been.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# The 12 directions LW6 resolves on a square grid, in circular order. Several share an
# (dx, dy) offset -- ENE and ESE are both (+1, 0) -- which is what gives the move fan
# below a finer angular resolution than the 8 cells a fighter can actually reach.
DIR_DX = np.array([1, 1, 1, 0, 0, -1, -1, -1, -1, 0, 0, 1], np.int32)
DIR_DY = np.array([0, 0, 1, 1, 1, 1, 0, 0, -1, -1, -1, -1], np.int32)
NB_DIRS = 12

# _LW6KER_TABLES_MOVE_DIR: [parity][preferred direction] -> directions to try, in order.
MOVE_FAN = np.array([
    [[0, 11, 2, 10, 3, 8, 5], [1, 2, 11, 3, 10, 5, 8], [2, 3, 1, 5, 11, 6, 10],
     [3, 2, 5, 1, 6, 11, 8], [4, 5, 2, 6, 1, 8, 11], [5, 6, 4, 8, 2, 9, 1],
     [6, 5, 8, 4, 9, 2, 11], [7, 8, 5, 9, 3, 11, 2], [8, 9, 7, 11, 5, 0, 4],
     [9, 8, 11, 7, 0, 5, 2], [10, 11, 8, 0, 7, 2, 5], [11, 0, 10, 2, 8, 3, 7]],
    [[0, 11, 2, 10, 3, 8, 5], [1, 2, 11, 3, 10, 5, 8], [2, 1, 3, 11, 5, 10, 6],
     [3, 2, 5, 1, 6, 11, 8], [4, 5, 2, 6, 1, 8, 11], [5, 4, 6, 2, 8, 1, 9],
     [6, 5, 8, 4, 9, 2, 11], [7, 8, 5, 9, 3, 11, 2], [8, 7, 9, 5, 11, 4, 0],
     [9, 8, 11, 7, 0, 5, 2], [10, 11, 8, 0, 7, 2, 5], [11, 10, 0, 8, 2, 7, 3]],
], np.int32)

# _LW6KER_TABLES_STRAIGHT_DIRS: [parity][UP|RIGHT|DOWN|LEFT bitmask] -> direction.
# Used when a fighter is already in the cursor's cell and has no uphill neighbour.
STRAIGHT_DIR_UP, STRAIGHT_DIR_RIGHT, STRAIGHT_DIR_DOWN, STRAIGHT_DIR_LEFT = 1, 2, 4, 8
STRAIGHT_DIRS = np.array([
    [-1, 10, 1, 11, 4, -1, 2, -1, 7, 8, -1, -1, 5, -1, -1, -1],
    [-1, 9, 0, 11, 3, -1, 2, -1, 6, 8, -1, -1, 5, -1, -1, -1],
], np.int32)

# The forward sweep pushes potential south and east, the backward sweep north and west.
# Sweeping rather than relaxing in place is a fast-sweeping scheme: potential travels
# arbitrarily far across the map in a single pass, so a handful of passes per round
# leaves the field essentially converged.
FORWARD_DIRS = (0, 1, 2, 3, 4, 5)
BACKWARD_DIRS = (6, 7, 8, 9, 10, 11)

# LW6's default SPREAD_MODE_HALF: each pass advances the current sweep direction round
# the circle and then pushes along the *three* directions of that quadrant, not just
# the one. Doing a single direction per pass (the obvious reading) makes the gradient
# propagate about three times too slowly, and the population curve comes out visibly
# behind the real kernel's.
HALF_DIRS = {
    (0, 1, 2): ((0, 1, 2), True),        # ENE ESE SE  -> forward
    (3, 4, 5): ((3, 4, 5), True),        # SSE SSW SW  -> forward
    (6, 7, 8): ((6, 7, 8), False),       # WSW WNW NW  -> backward
    (9, 10, 11): ((9, 10, 11), False),   # NNW NNE NE  -> backward
}


def _half_set(d):
    for key, val in HALF_DIRS.items():
        if d in key:
            return val
    raise ValueError(d)


@dataclass
class Rules:
    """LW6 rule defaults, from src/lib/map/map.h."""
    max_fighter_health: int = 10000
    fighter_attack: int = 500
    fighter_defense: int = 50
    fighter_new_health: int = 5000
    fighter_regenerate: int = 5
    side_attack_factor: int = 20        # percent, for the non-frontal directions
    side_defense_factor: int = 20
    nb_move_tries: int = 5
    nb_attack_tries: int = 3
    nb_defense_tries: int = 1
    moves_per_round: int = 2
    spreads_per_round: int = 5
    round_delta: int = 1
    max_round_delta: int = 1000
    cursor_pot_init: int = 100000
    max_cursor_pot: int = 1000000
    max_cursor_pot_offset: int = 100
    spread_mode_all: bool = False       # LW6 default is HALF: one sweep direction/pass


class ReferenceGame:
    def __init__(self, walls, positions, teams, *, rules: Rules | None = None, seed: int = 0,
                 max_health: int | None = None):
        self.rules = rules or Rules()
        if max_health is not None:      # convenience knob used by the fidelity harness
            self.rules.fighter_attack = max(1, self.rules.max_fighter_health // max_health)
            self.rules.fighter_new_health = self.rules.max_fighter_health // 2
        self.walls = np.asarray(walls, np.float32)
        self.H, self.W = self.walls.shape
        self.pos = np.asarray(positions, np.int32).copy()
        self.team = np.asarray(teams, np.int32).copy()
        self.N = len(self.team)
        self.health = np.full(self.N, self.rules.max_fighter_health, np.int32)
        self.last_dir = np.zeros(self.N, np.int32)
        self.rng = np.random.default_rng(seed)
        self.occ = np.full((self.H, self.W), -1, np.int32)
        for i, (y, x) in enumerate(self.pos):
            self.occ[y, x] = i
        self.cursor = np.zeros((2, 2), np.float32)
        self.pot = [np.zeros((self.H, self.W), np.int64) for _ in range(2)]
        self.cursor_ref_pot = [self.rules.cursor_pot_init, self.rules.cursor_pot_init]
        self.last_spread_dir = [0, 0]
        self.round = 0

    # --------------------------------------------------------------- helpers
    def set_cursor(self, cursor):
        self.cursor = np.asarray(cursor, np.float32).reshape(2, 2)

    def cursor_cell(self, t):
        return (int(np.clip(round(float(self.cursor[t, 0])), 0, self.H - 1)),
                int(np.clip(round(float(self.cursor[t, 1])), 0, self.W - 1)))

    def population(self):
        return np.array([int((self.team == 0).sum()), int((self.team == 1).sum())])

    def score(self) -> float:
        return float((self.team == 0).sum()) / self.N

    def density_image(self) -> np.ndarray:
        img = np.zeros((2, self.H, self.W), np.float32)
        img[self.team, self.pos[:, 0], self.pos[:, 1]] = 1.0
        return img

    def health_image(self) -> np.ndarray:
        """Per-cell health in [0, 1], for rendering fighters darkening before they flip."""
        img = np.zeros((self.H, self.W), np.float32)
        img[self.pos[:, 0], self.pos[:, 1]] = self.health / self.rules.max_fighter_health
        return img

    # -------------------------------------------------------------- gradient
    def apply_cursors(self):
        """Raise each team's cursor cell to a potential that climbs every round."""
        r = self.rules
        for t in (0, 1):
            cy, cx = self.cursor_cell(t)
            if self.walls[cy, cx] < 0.5:
                continue
            here = int(self.pot[t][cy, cx])
            max_pot = max(self.cursor_ref_pot[t], here)
            delta = max(r.round_delta, self.cursor_ref_pot[t] - here)
            delta = min(delta, r.max_round_delta)
            self.cursor_ref_pot[t] = max_pot + delta
            if self.cursor_ref_pot[t] + r.max_cursor_pot_offset > r.max_cursor_pot:
                self.normalize_pot(t)
            self.pot[t][cy, cx] = max(int(self.pot[t][cy, cx]), self.cursor_ref_pot[t])

    def normalize_pot(self, t):
        r = self.rules
        p = self.pot[t]
        lo, hi = int(p.min()), int(p.max())
        delta = max(lo, hi // 2)
        p -= delta
        bad = (p <= 0) | (p > r.max_cursor_pot)
        p[bad] = r.cursor_pot_init
        self.cursor_ref_pot[t] = max(hi - delta, r.cursor_pot_init)

    def _sweep(self, t, dirs, forward):
        """One fast-sweeping pass: pot[n] = max(pot[n], pot[c] - 1) over `dirs`."""
        p = self.pot[t]
        open_ = self.walls > 0.5
        rows = range(self.H) if forward else range(self.H - 1, -1, -1)
        cols = range(self.W) if forward else range(self.W - 1, -1, -1)
        for y in rows:
            for x in cols:
                if not open_[y, x]:
                    continue
                v = p[y, x] - 1
                for d in dirs:
                    ny, nx = y + int(DIR_DY[d]), x + int(DIR_DX[d])
                    if 0 <= ny < self.H and 0 <= nx < self.W and open_[ny, nx] and p[ny, nx] < v:
                        p[ny, nx] = v

    def spread(self):
        r = self.rules
        for t in (0, 1):
            for _ in range(r.spreads_per_round):
                self.last_spread_dir[t] = (self.last_spread_dir[t] + 1) % NB_DIRS
                d = self.last_spread_dir[t]
                if r.spread_mode_all:
                    forward = d in FORWARD_DIRS
                    self._sweep(t, FORWARD_DIRS if forward else BACKWARD_DIRS, forward)
                else:
                    dirs, forward = _half_set(d)
                    self._sweep(t, dirs, forward)

    def _best_dir(self, i, parity):
        """The neighbouring direction with the highest potential (closest to cursor)."""
        y, x = self.pos[i]
        p = self.pot[self.team[i]]
        best, ret = int(p[y, x]), -1
        order = range(NB_DIRS) if parity else range(NB_DIRS - 1, -1, -1)
        for d in order:
            ny, nx = y + int(DIR_DY[d]), x + int(DIR_DX[d])
            if not (0 <= ny < self.H and 0 <= nx < self.W) or self.walls[ny, nx] < 0.5:
                continue
            if int(p[ny, nx]) > best:
                best, ret = int(p[ny, nx]), d
        if ret < 0:
            # No uphill neighbour: we are standing on the cursor. Head straight at it.
            cy, cx = self.cursor_cell(self.team[i])
            mask = 0
            if cy < y:
                mask |= STRAIGHT_DIR_UP
            if cx > x:
                mask |= STRAIGHT_DIR_RIGHT
            if cy > y:
                mask |= STRAIGHT_DIR_DOWN
            if cx < x:
                mask |= STRAIGHT_DIR_LEFT
            ret = int(STRAIGHT_DIRS[parity][mask])
        if ret < 0:
            ret = int(self.last_dir[i])
        return ret

    # ------------------------------------------------------------------ tick
    def step(self):
        """One LW6 round: apply cursors, spread the gradient, then move the fighters."""
        r = self.rules
        self.apply_cursors()
        self.spread()
        for _ in range(r.moves_per_round):
            self._move_all()
        self.round += 1

    def _move_all(self):
        r = self.rules
        parity = self.round % 2
        order = range(self.N) if parity else range(self.N - 1, -1, -1)
        for i in order:
            y, x = int(self.pos[i][0]), int(self.pos[i][1])
            t = int(self.team[i])
            best = self._best_dir(i, parity)
            self.last_dir[i] = best
            fan = MOVE_FAN[parity][best]

            # 1. move into the first free cell in the fan
            done = False
            for j in range(r.nb_move_tries):
                d = int(fan[j])
                ny, nx = y + int(DIR_DY[d]), x + int(DIR_DX[d])
                if not (0 <= ny < self.H and 0 <= nx < self.W):
                    continue
                if self.walls[ny, nx] < 0.5 or self.occ[ny, nx] >= 0:
                    continue
                self.occ[y, x] = -1
                self.occ[ny, nx] = i
                self.pos[i] = (ny, nx)
                self._regenerate(i)
                done = True
                break

            # 2. blocked -- attack an enemy in the fan. Straight ahead hits full
            #    strength, the flanking tries hit for side_attack_factor percent.
            if not done:
                for j in range(r.nb_attack_tries):
                    d = int(fan[j])
                    ny, nx = y + int(DIR_DY[d]), x + int(DIR_DX[d])
                    if not (0 <= ny < self.H and 0 <= nx < self.W):
                        continue
                    k = self.occ[ny, nx]
                    if k >= 0 and self.team[k] != t:
                        dmg = (r.fighter_attack if j == 0
                               else max(1, r.fighter_attack * r.side_attack_factor // 100))
                        self._attack(i, k, dmg)
                        done = True
                        break

            # 3. no enemy either -- heal a team-mate
            if not done:
                for j in range(r.nb_defense_tries):
                    d = int(fan[j])
                    ny, nx = y + int(DIR_DY[d]), x + int(DIR_DX[d])
                    if not (0 <= ny < self.H and 0 <= nx < self.W):
                        continue
                    k = self.occ[ny, nx]
                    if k >= 0 and self.team[k] == t:
                        heal = (r.fighter_defense if j == 0
                                else max(1, r.fighter_defense * r.side_defense_factor // 100))
                        self.health[k] = min(self.health[k] + heal, r.max_fighter_health)
                        done = True
                        break

            if not done:
                self._regenerate(i)

    def _regenerate(self, i):
        self.health[i] = min(self.health[i] + self.rules.fighter_regenerate,
                             self.rules.max_fighter_health)

    def _attack(self, attacker, victim, damage):
        """Population is conserved: a fighter is never removed, only converted."""
        self.health[victim] -= damage
        if self.health[victim] <= 0:
            self.team[victim] = self.team[attacker]
            self.health[victim] = self.rules.fighter_new_health

    def run(self, ticks: int, cursor_fn):
        """cursor_fn(t) -> [2, 2]. Returns the per-tick score curve."""
        curve = np.empty(ticks + 1, np.float32)
        curve[0] = self.score()
        for t in range(ticks):
            self.set_cursor(cursor_fn(t))
            self.step()
            curve[t + 1] = self.score()
        return curve


def bfs_potential(walls, cursor_yx):
    """Plain 8-connected BFS distance. Not used by the game; kept for tests."""
    H, W = walls.shape
    pot = np.full((H, W), 1 << 24, np.int32)
    cy = int(np.clip(round(float(cursor_yx[0])), 0, H - 1))
    cx = int(np.clip(round(float(cursor_yx[1])), 0, W - 1))
    if walls[cy, cx] < 0.5:
        return pot
    pot[cy, cx] = 0
    frontier, d = [(cy, cx)], 0
    while frontier:
        d += 1
        nxt = []
        for y, x in frontier:
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and walls[ny, nx] > 0.5 and pot[ny, nx] > d:
                    pot[ny, nx] = d
                    nxt.append((ny, nx))
        frontier = nxt
    return pot
