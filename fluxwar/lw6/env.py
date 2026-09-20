"""Env-shaped wrapper around the real Liquid War 6 kernel.

Same interface the density sim exposes, so any agent can play the actual game. This is
the arbiter: anything the differentiable sim claims about a policy is checked here.

Batch size is 1. LW6's fighter update is sequential and order-dependent, so games
cannot be batched without changing results -- run many across processes, with
OMP_NUM_THREADS=1 so the kernel's own OpenMP does not fight the process pool.
"""
from __future__ import annotations

import numpy as np
import torch

from ..sim.observation import observation
from .kernel import LW6Game


class LW6Env:
    """One real LW6 game, presented as a batch of 1."""

    def __init__(self, width=64, height=64, noise_percent=10, max_cursor_speed=1.0,
                 device="cpu", lib_path=None, walls=None, action_mode="velocity"):
        self.game = LW6Game(width=width, height=height, noise_percent=noise_percent,
                            lib_path=lib_path, walls=walls)
        self.device = torch.device(device)
        self.dtype = torch.float32
        self.B = 1
        self.H, self.W = self.game.H, self.game.W
        self.max_cursor_speed = float(max_cursor_speed)
        self.action_mode = action_mode
        # Teams whose cursor is driven by something else -- a native LW6 bot writing
        # straight into the kernel. Their entry in `action` is ignored and their
        # position is read back instead. Without this, `step` overwrites whatever the
        # bot just chose with the action array's (unused) entry for that team, which
        # silently turns the opponent off: a policy "trained against mod-brute"
        # reached a 1.0 win rate in 156k steps because brute's cursor was being
        # parked in a corner every round.
        self.external_teams = set()
        self._walls_np = self.game.walls()
        self.walls = torch.as_tensor(self._walls_np, device=self.device).unsqueeze(0)
        # Start each cursor on its own army, which is where the game itself puts the
        # fighters; otherwise the first hundred rounds are just walking.
        self._cursor = np.zeros((2, 2), np.float32)   # (y, x) per team, this repo's order
        d = self.game.density()
        for t in range(2):
            self._cursor[t] = self._centroid(d[t])
        self._push_cursors()

    @staticmethod
    def _centroid(layer):
        m = layer.sum()
        if m <= 0:
            return np.array([layer.shape[0] / 2, layer.shape[1] / 2], np.float32)
        ys, xs = np.nonzero(layer)
        wts = layer[ys, xs]
        return np.array([(ys * wts).sum() / m, (xs * wts).sum() / m], np.float32)

    def _push_cursors(self, teams=None):
        for t in (range(2) if teams is None else teams):
            # LW6's API is (x, y); this repo is (y, x) throughout.
            self.game.set_cursor(t, int(round(self._cursor[t, 1])),
                                 int(round(self._cursor[t, 0])))

    # ------------------------------------------------------------------ views
    @property
    def density(self):
        return torch.as_tensor(self.game.density(), device=self.device).unsqueeze(0)

    @property
    def cursor(self):
        return torch.as_tensor(self._cursor, device=self.device).unsqueeze(0)

    def obs(self, team: int = 0):
        d = self.density
        return observation(d[:, team], d[:, 1 - team], self.walls)

    def obs_into(self, out, team: int = 0):
        """Write the observation straight into a [3, H, W] float32 numpy buffer.

        Same definition as `observation`, in numpy. Going through torch for this
        costs more than stepping the game does: it is ~4 small tensor allocations per
        env per round, which at a few hundred envs dominates the whole vec env.
        """
        t, _h, y, x = self.game.fighter_arrays()
        out[:2] = 0.0
        keep = t < 2
        if keep.any():
            tt = t[keep] if team == 0 else (1 - t[keep])
            np.add.at(out, (tt, y[keep], x[keep]), 1.0)
        total = out[0].sum() + out[1].sum()
        scale = max(total, 1e-12) / (self.H * self.W)
        out[:2] /= scale
        out[2] = self._walls_np
        return out

    def score(self):
        return torch.tensor([self.game.score()], device=self.device)

    # ------------------------------------------------------------------- tick
    def step(self, action: torch.Tensor):
        """action: [1, 2, 2] per team.

        `velocity` mode: a direction in [-1, 1], scaled by MAX_CURSOR_SPEED.
        `target` mode: a normalised position in [0, 1], which is what LW6's own bots
        get to use.
        """
        a = action[0].detach().cpu().numpy()
        if self.action_mode == "target":
            nxt = np.clip(a, 0.0, 1.0) * np.array([self.H - 1, self.W - 1], np.float32)
        else:
            nxt = self._cursor + a * self.max_cursor_speed
        for t in self.external_teams:
            c = self.game.cursor(t)          # believe the kernel, not the action
            nxt[t] = (c.pos.y, c.pos.x)
        # Cursors move freely, walls included. LW6 allows it explicitly: a cursor
        # "hanging on a wall" is applied at the nearest free slot instead
        # (_lw6ker_cursor_update_apply_pos -> find_free_slot_near). Blocking the
        # cursor on walls is both unfaithful and catastrophic -- a straight-line
        # chaser wedges itself in the first concave corner and never moves again.
        self._cursor[:, 0] = np.clip(nxt[:, 0], 0, self.H - 1)
        self._cursor[:, 1] = np.clip(nxt[:, 1], 0, self.W - 1)
        self._push_cursors([t for t in range(2) if t not in self.external_teams])
        self.game.step()
        return self.obs(), None, self.score()

    def close(self):
        self.game.close()
