"""Fidelity measured the way training actually uses the simulator.

`eval/fidelity.py` replays one fixed cursor script and compares population curves.
That is a clean signal but a narrow one, and it misses the failure that matters: the
density sim scoring a *matchup* differently from the real game. Measured on the
scripted pair chase_enemy vs hold_own, the tuned density sim called it 1.00 for the
chaser where the real kernel called it 0.58 -- a policy trained against that learns
that charging is free.

So this plays the same agents on the same maps in both engines and compares the
outcome. It is the number to tune against, because it is the number training depends
on.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from ..config import load
from ..lw6.env import LW6Env
from ..policy.agents import SCRIPTED
from ..sim.env import FluxWar
from ..sim.maps import make_maps

PAIRS = [("chase_enemy", "hold_own"), ("chase_enemy", "stationary"),
         ("lean5", "hold_own"), ("lean2", "chase_enemy")]


def _maps(n, size, seed0=0):
    out = []
    for i in range(n):
        gen = torch.Generator(device="cpu").manual_seed(seed0 + i)
        c = load()
        c.sim.H = c.sim.W = size
        out.append(make_maps(1, c, "cpu", gen)[0].numpy())
    return out


def starting_positions(walls, size, cfg):
    """The kernel's own fighter placement, as a [2, H, W] density.

    Matching the map is not enough. Letting each engine pick its own starting
    positions is a confound big enough to swamp the measurement: on one map the
    density sim's spawns left the two armies unable to reach each other at all and
    every matchup came back at exactly 0.500, while the real kernel resolved all of
    them decisively.
    """
    env = LW6Env(width=size, height=size, device="cpu", walls=walls,
                 max_cursor_speed=float(cfg.sim.MAX_CURSOR_SPEED))
    d = env.game.density().copy()
    env.close()
    return d


@torch.no_grad()
def play_real(walls, a, b, rounds, size, cfg):
    env = LW6Env(width=size, height=size, device="cpu", walls=walls,
                 max_cursor_speed=float(cfg.sim.MAX_CURSOR_SPEED))
    A, B = SCRIPTED[a], SCRIPTED[b]
    A.reset(env)
    B.reset(env)
    for _ in range(rounds):
        env.step(torch.stack([A.act(env, 0), B.act(env, 1)], 1))
    s = float(env.score()[0])
    env.close()
    return s


@torch.no_grad()
def play_density(maps, a, b, rounds, size, cfg, device="cuda", starts=None):
    """All the maps for one matchup in a single batch.

    One game at a time leaves the GPU idle between tiny kernels and made a parameter
    sweep take hours; the agents are the same across the batch, so the maps can share
    an env.
    """
    walls = torch.as_tensor(np.stack(maps), device=device, dtype=torch.float32)
    env = FluxWar(cfg, device, seed=0, B=len(maps))
    env.reset(maps=walls)
    if starts is not None:
        # Start from the kernel's own placement, and put each cursor on its army.
        env.density = torch.as_tensor(np.stack(starts), device=device,
                                      dtype=env.density.dtype)
        ys = torch.arange(size, device=device, dtype=env.density.dtype)
        for t in range(2):
            layer = env.density[:, t]
            m = layer.sum((1, 2)).clamp_min(1e-9)
            env.cursor[:, t, 0] = (layer.sum(2) * ys).sum(1) / m
            env.cursor[:, t, 1] = (layer.sum(1) * ys).sum(1) / m
        env.relax_field(k=2 * size)
    A, B = SCRIPTED[a], SCRIPTED[b]
    for _ in range(rounds):
        env.step(torch.stack([A.act(env, 0), B.act(env, 1)], 1))
    return env.score().cpu().numpy()


def compare(cfg, n_maps=3, rounds=400, size=48, device="cuda", pairs=PAIRS,
            real_cache=None, maps=None):
    """Mean absolute difference in final share, over maps and matchups."""
    maps = maps if maps is not None else _maps(n_maps, size)
    starts = [starting_positions(w, size, cfg) for w in maps]
    rows, diffs = [], []
    for a, b in pairs:
        dens = play_density(maps, a, b, rounds, size, cfg, device, starts=starts)
        for i, walls in enumerate(maps):
            key = (a, b, i)
            if real_cache is not None and key in real_cache:
                r = real_cache[key]
            else:
                r = play_real(walls, a, b, rounds, size, cfg)
                if real_cache is not None:
                    real_cache[key] = r
            d = float(dens[i])
            rows.append((a, b, i, r, d))
            diffs.append(abs(r - d))
    return dict(mean_abs_diff=float(np.mean(diffs)), max_abs_diff=float(np.max(diffs)),
                rows=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="default.yaml")
    ap.add_argument("--maps", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--size", type=int, default=48)
    a = ap.parse_args()
    cfg = load(a.config, **{"sim.H": a.size, "sim.W": a.size})
    r = compare(cfg, n_maps=a.maps, rounds=a.rounds, size=a.size)
    for x, y, i, real, dens in r["rows"]:
        print(f"{x:12s} vs {y:12s} map {i}: real {real:.3f}  density {dens:.3f}  "
              f"diff {abs(real - dens):.3f}")
    print(json.dumps({k: v for k, v in r.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
