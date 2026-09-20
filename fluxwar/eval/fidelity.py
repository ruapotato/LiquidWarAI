"""Measure the differentiable density sim against the real Liquid War 6 kernel.

The reference here is LW6 itself -- `liblw6ker` compiled from source and called
through ctypes -- not a re-implementation. The real kernel generates the map and
places the fighters, so the only way to give both engines an identical starting
position is to read it out of the real one and replay it, which is what `scenario`
does. Both are then driven by the same cursor script and their population curves
compared round by round.

Two scenarios tell you nothing, and both were used here by mistake before this was
written down:

* **Both cursors on the same cell.** Both armies relax onto the same attractor, the
  two density fields become identical, and combat is exactly symmetric whatever the
  rule is. Every candidate parameter set scores a flat 0.5 and all look equally good.
* **A symmetric start with a symmetric script.** Same problem, same reason.

So the asymmetry comes from the cursor script: team 0 holds its ground, team 1 walks
onto it. Both engines have to agree on who that favours and by how much.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from ..config import load
from ..lw6.kernel import LW6Game
from ..sim import core
from ..sim.core import Params, score


def scenario(width=48, height=48, noise=10, seed=0, cfg=None):
    """Map, starting placement and cursor script, taken from a real LW6 game.

    The map comes from this project's generator and is injected into the kernel, so
    both engines play the identical board. Left to itself `lw6map_builtin_custom`
    returns the same map every time whatever noise_percent says, which makes a sweep
    over several "scenarios" a sweep over one.
    """
    walls_in = None
    if cfg is not None:
        import torch
        from ..sim.maps import make_maps
        gen = torch.Generator(device="cpu").manual_seed(seed)
        c = load() if cfg is True else cfg
        c = dict(c)
        c["sim"] = dict(c["sim"], H=height, W=width)
        from ..config import _wrap
        walls_in = make_maps(1, _wrap(c), "cpu", gen)[0].numpy()
    game = LW6Game(width=width, height=height, noise_percent=noise, walls=walls_in)
    walls = game.walls().copy()
    team, _health, y, x = game.fighter_arrays()
    keep = team < 2
    pos = np.stack([y[keep], x[keep]], axis=1)
    team = team[keep].copy()
    starts = [pos[team == t].mean(axis=0) for t in range(2)]
    return game, walls, pos, team, starts


def cursor_script(starts, charge_rounds=120):
    """Team 0 holds its start, team 1 walks onto team 0. (y, x)."""
    (ay, ax), (by, bx) = starts

    def fn(t):
        f = min(1.0, t / float(charge_rounds))
        return np.array([[ay, ax], [by + (ay - by) * f, bx + (ax - bx) * f]], np.float32)
    return fn


def run_real(game, fn, rounds):
    curve = np.empty(rounds + 1, np.float32)
    curve[0] = game.score()
    for t in range(rounds):
        c = fn(t)
        for team_i in range(2):
            game.set_cursor(team_i, int(round(float(c[team_i][1]))),
                            int(round(float(c[team_i][0]))))
        game.step()
        curve[t + 1] = game.score()
    return curve


def run_density(walls, pos, team, fn, rounds, cfg, device="cuda"):
    p = Params.from_cfg(cfg)
    H, W = walls.shape
    w = torch.as_tensor(walls, device=device).unsqueeze(0)
    dens = torch.zeros(1, 2, H, W, device=device)
    dens[0, team, pos[:, 0], pos[:, 1]] = 1.0     # raw counts: one fighter per cell
    pot = torch.full((1, 2, H, W), p.LARGE, device=device)
    cur = torch.as_tensor(fn(0), device=device).view(1, 2, 2)
    pot = core.relax(pot, core.cursor_seed(cur, w, p.seed_radius, p.LARGE),
                     w.unsqueeze(1), k=H + W, p=p)
    curve = np.empty(rounds + 1, np.float32)
    curve[0] = float(score(dens))
    with torch.no_grad():
        for t in range(rounds):
            tgt = torch.as_tensor(fn(t), device=device).view(1, 2, 2)
            action = ((tgt - cur) / p.MAX_CURSOR_SPEED).clamp(-1, 1)
            dens, pot, cur = core.tick(dens, pot, cur, w, action, p)
            curve[t + 1] = float(score(dens))
    return curve


def compare(cfg, rounds=300, width=48, height=48, noise=10, device="cuda",
            verbose=False, seed=0, generated_map=True):
    game, walls, pos, team, starts = scenario(width, height, noise, seed,
                                              cfg if generated_map else None)
    fn = cursor_script(starts)
    try:
        real = run_real(game, fn, rounds)
    finally:
        game.close()
    dens = run_density(walls, pos, team, fn, rounds, cfg, device)
    signal = max(float(np.abs(real - 0.5).max()), 1e-6)
    err = float(np.abs(dens - real).max())
    if verbose:
        step = max(rounds // 6, 1)
        print("real LW6", real[::step].round(3))
        print("density ", dens[::step].round(3))
        print(f"max abs err {err:.4f}  relative {err / signal:.3f}")
    return dict(real=real, density=dens, max_abs_err=err, rel_err=err / signal,
                signal=signal)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="default.yaml")
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--size", type=int, default=48)
    a = ap.parse_args()
    r = compare(load(a.config), rounds=a.rounds, width=a.size, height=a.size, verbose=True)
    print(json.dumps({k: float(v) for k, v in r.items() if not isinstance(v, np.ndarray)},
                     indent=2))
