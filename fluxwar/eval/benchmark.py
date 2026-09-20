"""Benchmark a policy against LW6's own bots -- with the bots CAPPED.

Note which harness you want. This one routes every agent through `LW6Env`, whose
action is a cursor *velocity*, so a bot that returns an absolute position gets its
move converted to a unit step. That caps `mod-brute` at one cell per round, which is
a large handicap: uncapped it teleports anywhere on the map each round and takes 0.92
off `mod-follow`.

That makes this the right harness for "how does the policy do against bots with the
same cursor mobility it has", and the wrong one for "how does the policy do against
the game's AI as shipped". For the latter use `eval/deployed.py`, which drives
cursors the way the game does and lets the bots use their real action space.

Games run one per worker process: LW6's fighter update is sequential and
order-dependent, so a game cannot be batched.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp

import numpy as np
import torch

from ..config import load
from ..lw6.bots import LW6Bot
from ..lw6.env import LW6Env
from ..policy.agents import SCRIPTED, NetAgent
from ..policy.net import CursorPolicy
from ..sim.maps import make_maps


def _make_agent(spec, cfg):
    if spec.startswith("lw6_"):
        return LW6Bot(spec[4:])
    if spec in SCRIPTED:
        return SCRIPTED[spec]
    net = CursorPolicy(cfg)
    net.load_state_dict(torch.load(spec, map_location="cpu"))
    net.eval()
    return NetAgent(net, name=spec)


def _play(args):
    spec_a, spec_b, cfg, seed, rounds, size, swap = args
    from ..lw6.kernel import reset_process_state, silence
    reset_process_state()
    silence()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    c = load()
    c.sim.H = c.sim.W = size
    walls = make_maps(1, c, "cpu", gen)[0].numpy()
    env = LW6Env(width=size, height=size, device="cpu", walls=walls)
    a, b = _make_agent(spec_a, cfg), _make_agent(spec_b, cfg)
    a.reset(env)
    b.reset(env)
    ta, tb = (1, 0) if swap else (0, 1)
    with torch.no_grad():
        for _ in range(rounds):
            act_a, act_b = a.act(env, ta), b.act(env, tb)
            env.step(torch.stack([act_b, act_a], 1) if swap
                     else torch.stack([act_a, act_b], 1))
    s = float(env.score()[0])
    env.close()
    return 1.0 - s if swap else s


def head_to_head(spec_a, spec_b, cfg, games=16, rounds=600, size=48, processes=None):
    """Half the games with A on each side, to control for map asymmetry."""
    jobs = [(spec_a, spec_b, cfg, s, rounds, size, s % 2 == 1) for s in range(games)]
    # fork, not spawn: spawn re-imports torch in every worker and was observed to
    # stall here. Each worker calls reset_process_state() so it does not inherit the
    # parent's LW6 context pointer across the fork.
    with mp.get_context("fork").Pool(processes or min(mp.cpu_count(), games)) as pool:
        scores = pool.map(_play, jobs)
    s = np.array(scores)
    eps = 1e-3
    win = (s > 0.5 + eps).astype(float)
    draw = (np.abs(s - 0.5) <= eps).astype(float)
    return dict(win_rate=float(win.mean()), draw_rate=float(draw.mean()),
                loss_rate=float(1 - win.mean() - draw.mean()),
                mean_share=float(s.mean()), games=games)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="default.yaml")
    ap.add_argument("--policy", required=True)
    ap.add_argument("--against", nargs="+",
                    default=["lw6_brute", "lw6_follow", "hold_own", "chase_enemy"])
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--rounds", type=int, default=600)
    ap.add_argument("--size", type=int, default=48)
    a = ap.parse_args()
    cfg = load(a.config, **{"sim.H": a.size, "sim.W": a.size})
    out = {}
    for opp in a.against:
        out[opp] = head_to_head(a.policy, opp, cfg, games=a.games, rounds=a.rounds,
                                size=a.size)
        r = out[opp]
        print(f"{a.policy} vs {opp:14s} W {r['win_rate']:.2f} D {r['draw_rate']:.2f} "
              f"L {r['loss_rate']:.2f}  share {r['mean_share']:.3f}", flush=True)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
