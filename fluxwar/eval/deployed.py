"""Play real Liquid War 6 games with bots driving cursors the way the game does.

Every other harness here converts a bot's output into this project's velocity action
and lets the env move the cursor. That is wrong for measuring a *deployed* bot:
LW6 bots return an absolute cursor position, the game writes it straight in, and
mod-nn keeps its own sub-cell position. Round-tripping through a velocity makes
mod-nn and the caller fight over the cursor.

So this drives `LW6Game` directly: each round, ask each bot where its cursor should
be and put it there. That is the code path a real Liquid War 6 build uses, which
makes this the end-to-end check on deployment, and the one to quote for "does it
beat the bots the game ships with".

`--cap` optionally limits every native bot to a given number of cells per round, for
the separate question of how the policy does against bots with its own mobility.
`eval/benchmark.py` applies that cap implicitly and cannot turn it off.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import multiprocessing as mp
import os

import numpy as np
import torch

from ..config import load
from ..lw6 import bots as botmod
from ..lw6.kernel import LW6Game, context, load as load_lib
from ..sim.maps import make_maps


class NativeBot:
    """One of LW6's C bot backends, returning an absolute (x, y) cursor target.

    `max_cells_per_round` optionally caps how far the cursor may travel per round.
    It matters for fair comparison: mod-brute repositions its cursor anywhere on the
    map every round (`cursor.pos.x = lw6sys_random(shape.w)`), which no human hand can
    do and which this project's policies cannot do either, since they emit a bounded
    velocity. Uncapped, mod-brute takes 0.92 of the population against mod-follow and
    0.996 against a stationary cursor. None means no cap -- the game's own behaviour.
    """

    def __init__(self, kind, iq=100, speed=1.0, max_cells_per_round=None):
        self.kind = kind
        self.lib = load_lib()
        botmod._declare(self.lib)
        self.ctx = context(self.lib)
        self.iq, self.speed = int(iq), float(speed)
        self.max_step = max_cells_per_round
        self.handle = None

    def attach(self, game: LW6Game, team: int):
        self.data = botmod.BotData()
        self.data.game_state = game.game_state
        self.data.param = botmod.BotParam(self.speed, self.iq, game.cursor_ids[team])
        self.handle = getattr(self.lib, f"_mod_{self.kind}_init")(
            self.ctx, 0, None, ctypes.byref(self.data))
        if not self.handle:
            raise RuntimeError(f"_mod_{self.kind}_init failed")
        self.game, self.team = game, team

    def move(self):
        x, y = ctypes.c_int(0), ctypes.c_int(0)
        self.data.game_state = self.game.game_state
        ok = getattr(self.lib, f"_mod_{self.kind}_next_move")(
            self.ctx, self.handle, ctypes.byref(x), ctypes.byref(y),
            ctypes.byref(self.data))
        if not ok:
            return
        tx, ty = float(x.value), float(y.value)
        if self.max_step is not None:
            cur = self.game.cursor(self.team)
            dx, dy = tx - cur.pos.x, ty - cur.pos.y
            n = (dx * dx + dy * dy) ** 0.5
            if n > self.max_step:
                tx = cur.pos.x + dx * self.max_step / n
                ty = cur.pos.y + dy * self.max_step / n
        self.game.set_cursor(self.team, int(round(tx)), int(round(ty)))


class StaticBot:
    """Holds position. The simplest possible control, useful as a floor."""

    def attach(self, game, team):
        self.game, self.team = game, team

    def move(self):
        pass


def _make(kind, weights=None, cap=None):
    if kind == "stationary":
        return StaticBot()
    if kind == "nn":
        os.environ["FLUXWAR_POLICY"] = str(weights)
        cap = None      # mod-nn already emits a bounded velocity
    return NativeBot(kind, max_cells_per_round=cap)


def _play(args):
    kind_a, kind_b, weights, seed, rounds, size, swap, cap = args
    from ..lw6.kernel import reset_process_state, silence
    reset_process_state()
    silence()
    if kind_a == "nn" or kind_b == "nn":
        os.environ["FLUXWAR_POLICY"] = str(weights)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    cfg = load()
    cfg.sim.H = cfg.sim.W = size
    walls = make_maps(1, cfg, "cpu", gen)[0].numpy()
    game = LW6Game(size, size, 10, walls=walls)
    a, b = _make(kind_a, weights, cap), _make(kind_b, weights, cap)
    ta, tb = (1, 0) if swap else (0, 1)
    a.attach(game, ta)
    b.attach(game, tb)
    for _ in range(rounds):
        a.move()
        b.move()
        game.step()
    s = game.score()
    game.close()
    return 1.0 - s if swap else s


def head_to_head(kind_a, kind_b, weights=None, games=16, rounds=600, size=64,
                 processes=None, cap=None):
    jobs = [(kind_a, kind_b, weights, s, rounds, size, s % 2 == 1, cap)
            for s in range(games)]
    with mp.get_context("fork").Pool(processes or min(mp.cpu_count(), games)) as pool:
        scores = np.array(pool.map(_play, jobs))
    eps = 1e-3
    win = (scores > 0.5 + eps).astype(float)
    draw = (np.abs(scores - 0.5) <= eps).astype(float)
    return dict(win_rate=float(win.mean()), draw_rate=float(draw.mean()),
                loss_rate=float(1 - win.mean() - draw.mean()),
                mean_share=float(scores.mean()), games=games)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="build/policy.flxw")
    ap.add_argument("--a", default="nn")
    ap.add_argument("--against", nargs="+", default=["brute", "follow", "stationary"])
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--rounds", type=int, default=600)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--cap", type=float, default=None,
                    help="cap every native bot's cursor to this many cells per round, "
                         "so a teleporting bot is compared like for like")
    args = ap.parse_args()
    out = {}
    for opp in args.against:
        r = head_to_head(args.a, opp, args.weights, games=args.games,
                         rounds=args.rounds, size=args.size, cap=args.cap)
        out[opp] = r
        print(f"mod-{args.a} vs mod-{opp:11s} W {r['win_rate']:.2f} D {r['draw_rate']:.2f} "
              f"L {r['loss_rate']:.2f}  share {r['mean_share']:.3f}", flush=True)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
