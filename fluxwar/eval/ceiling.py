"""What can a bounded cursor achieve against an unconstrained opponent?

`mod-brute` places its cursor anywhere on the map every round. A policy under the
human constraint moves at most `MAX_CURSOR_SPEED` cells. Before concluding that a
trained velocity agent is failing, it is worth knowing what *anything* under that
constraint can manage -- otherwise a flat training curve is indistinguishable from a
flat reward landscape, and the two call for opposite responses.

This measures the band directly: hand-written velocity policies, capped, against a
brute that is not. The learner is driven through `LW6Env` (so the cap applies);
brute writes its own cursor straight into the kernel, with `external_teams` stopping
the env from overwriting it.

Getting the comparison wrong is easy. Quoting `mod-follow`'s share against brute as
the baseline is not like for like: in the deployed harness mod-follow also returns an
absolute position each round, so it is effectively unconstrained too.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from ..config import load
from ..lw6.env import LW6Env
from ..policy.agents import SCRIPTED
from ..sim.maps import make_maps
from .deployed import NativeBot

DEFAULT_POLICIES = ("chase_enemy", "hold_own", "lean2", "lean5")


@torch.no_grad()
def play(policy_name, seed, rounds, size, cursor_speed, opponent="brute"):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    cfg = load()
    cfg.sim.H = cfg.sim.W = size
    walls = make_maps(1, cfg, "cpu", gen)[0].numpy()
    env = LW6Env(size, size, 10, device="cpu", walls=walls,
                 max_cursor_speed=cursor_speed)
    bot = NativeBot(opponent)
    bot.attach(env.game, 1)
    env.external_teams.add(1)          # the env must not overwrite the bot's choice
    agent = SCRIPTED[policy_name]
    agent.reset(env)
    idle = torch.zeros(1, 2)
    for _ in range(rounds):
        bot.move()
        env.step(torch.stack([agent.act(env, 0), idle], 1))
    share = float(env.score()[0])
    env.close()
    return share


def band(policies=DEFAULT_POLICIES, seeds=4, rounds=500, size=64, cursor_speed=4.0,
         opponent="brute"):
    out = {}
    for name in policies:
        scores = np.array([play(name, s, rounds, size, cursor_speed, opponent)
                           for s in range(seeds)])
        out[name] = dict(mean=float(scores.mean()), scores=scores.round(3).tolist())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opponent", default="brute")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=500)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--cursor-speed", type=float, default=4.0)
    a = ap.parse_args()
    r = band(seeds=a.seeds, rounds=a.rounds, size=a.size, cursor_speed=a.cursor_speed,
             opponent=a.opponent)
    print(f"velocity policies capped at {a.cursor_speed} cells/round "
          f"vs unconstrained mod-{a.opponent}:")
    for name, v in r.items():
        print(f"  {name:12s} share {v['mean']:.3f}   {v['scores']}")
    best = max(v["mean"] for v in r.values())
    print(f"\nceiling for a bounded cursor here: ~{best:.2f} of the population")
    print(json.dumps(r))


if __name__ == "__main__":
    main()
