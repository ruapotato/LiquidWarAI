"""Check the Python reference port against the real Liquid War 6 kernel.

The real kernel generates the map and places the fighters, so the only way to give
both implementations an identical starting position is to read it out of the real one
and replay it. Then drive both with the same cursor script and compare the population
curves round by round.

This is the test that decides whether `sim/reference.py` is worth anything: if the two
disagree, the real kernel is right and the port is not the ground truth it claims to be.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from ..lw6.kernel import LW6Game
from ..sim.reference import ReferenceGame, Rules


def snapshot(game: LW6Game):
    """(walls, positions, teams) as the Python port wants them."""
    walls = game.walls().copy()
    team, _health, y, x = game.fighter_arrays()
    keep = team < 2
    return walls, np.stack([y[keep], x[keep]], axis=1), team[keep].copy()


def cursor_script(W, H, kind="converge"):
    """Team 0 holds its start; team 1 walks onto team 0. Same for both engines."""
    def make(start0, start1):
        def fn(t):
            f = min(1.0, t / 120.0)
            return [[start0[0], start0[1]],
                    [start1[0] + (start0[0] - start1[0]) * f,
                     start1[1] + (start0[1] - start1[1]) * f]]
        return fn
    return make


def compare(rounds=300, width=64, height=64, noise=10, verbose=True):
    real = LW6Game(width=width, height=height, noise_percent=noise)
    walls, pos, team = snapshot(real)
    # Each side's starting centre of mass, in (y, x).
    starts = []
    for t in range(2):
        m = team == t
        starts.append(pos[m].mean(axis=0))
    fn = cursor_script(real.W, real.H)(starts[0], starts[1])

    port = ReferenceGame(walls, pos, team, rules=Rules(), seed=0)

    real_curve = np.empty(rounds + 1, np.float32)
    port_curve = np.empty(rounds + 1, np.float32)
    real_curve[0] = real.score()
    port_curve[0] = port.score()
    for t in range(rounds):
        c = fn(t)
        for team_i in range(2):
            real.set_cursor(team_i, int(round(c[team_i][1])), int(round(c[team_i][0])))
        real.step()
        port.set_cursor(c)
        port.step()
        real_curve[t + 1] = real.score()
        port_curve[t + 1] = port.score()
    real.close()

    signal = max(float(np.abs(real_curve - 0.5).max()), 1e-6)
    err = float(np.abs(port_curve - real_curve).max())
    out = dict(max_abs_err=err, rel_err=err / signal, signal=signal,
               real_final=float(real_curve[-1]), port_final=float(port_curve[-1]),
               real=real_curve, port=port_curve)
    if verbose:
        step = max(rounds // 6, 1)
        print("real LW6  ", real_curve[::step].round(3))
        print("python port", port_curve[::step].round(3))
        print(f"max abs err {err:.4f}   relative to signal {err / signal:.3f}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--noise", type=int, default=10)
    a = ap.parse_args()
    r = compare(rounds=a.rounds, width=a.size, height=a.size, noise=a.noise)
    print(json.dumps({k: v for k, v in r.items() if not isinstance(v, np.ndarray)}, indent=2))
