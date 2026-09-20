"""Batched fluxwar environment.

No gym interface on purpose: the whole batch lives in GPU memory and advances with one
call, so the one-env-per-step API would only force a copy. `step` is differentiable
whenever the incoming action is.
"""
from __future__ import annotations

import torch

from . import core
from .core import Params, score
from .maps import make_maps, spawn_points
from .observation import observation

_COMPILED = {}


def compiled_tick():
    if "tick" not in _COMPILED:
        _COMPILED["tick"] = torch.compile(core.tick, dynamic=False)
    return _COMPILED["tick"]


class FluxWar:
    def __init__(self, cfg, device="cuda", seed: int = 0, B: int | None = None,
                 compile_tick: bool = False, checkpoint: bool = False):
        self.cfg = cfg
        self.p = Params.from_cfg(cfg)
        self.device = torch.device(device)
        self.dtype = getattr(torch, cfg.sim.dtype)
        self.B = int(B if B is not None else cfg.sim.B)
        self.H, self.W = int(cfg.sim.H), int(cfg.sim.W)
        self.seed = seed
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(seed)
        self._tick = compiled_tick() if compile_tick else core.tick
        # Recompute the tick in the backward pass instead of storing its activations.
        # The capacity fixed point in `advect` keeps ~30 intermediate grids alive per
        # tick per team; over a 32-tick BPTT window at B = 128 that is ~20 GB and the
        # 3090 runs out. Checkpointing trades one extra forward per tick for it.
        self.checkpoint = bool(checkpoint)
        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self, maps: torch.Tensor | None = None, cursors: torch.Tensor | None = None):
        cfg = self.cfg
        B, H, W = self.B, self.H, self.W
        walls = make_maps(B, cfg, self.device, self.gen) if maps is None else maps.clone()
        self.walls = walls.to(self.dtype)
        pts = (spawn_points(self.walls, self.gen) if cursors is None else cursors).to(self.dtype)
        self.cursor = pts.clone()

        cap = float(cfg.sim.get("capacity", 0.0) or 0.0)
        ys = torch.arange(H, device=self.device, dtype=self.dtype).view(1, 1, H, 1)
        xs = torch.arange(W, device=self.device, dtype=self.dtype).view(1, 1, 1, W)
        d2 = (ys - pts[..., 0].view(B, 2, 1, 1)) ** 2 + (xs - pts[..., 1].view(B, 2, 1, 1)) ** 2
        mass = float(cfg.map.spawn_mass)
        if cap > 0:
            # A filled disk at capacity, like M0's one-fighter-per-cell start. A
            # gaussian blob of the same mass peaks well above capacity, and the first
            # few ticks are then spent spraying that overflow outwards.
            n_cells = max(int(round(mass / cap)), 1)
            far = d2 + self.p.LARGE * (1.0 - self.walls.unsqueeze(1))
            idx = far.flatten(2).topk(min(n_cells, H * W), dim=2, largest=False).indices
            blob = torch.zeros(B, 2, H * W, device=self.device, dtype=self.dtype)
            blob.scatter_(2, idx, 1.0)
            blob = blob.view(B, 2, H, W) * self.walls.unsqueeze(1)
        else:
            blob = torch.exp(-d2 / (2.0 * float(cfg.map.spawn_radius) ** 2))
            blob = blob * self.walls.unsqueeze(1)
        blob = blob / blob.sum(dim=(2, 3), keepdim=True).clamp_min(1e-12)
        self.density = blob * mass

        # Health stock: mass times mean health fraction. Everyone starts at full.
        self.stock = self.density.clone() if self.p.track_health else None
        self.pot = torch.full((B, 2, H, W), self.p.LARGE, device=self.device, dtype=self.dtype)
        self.relax_field(k=H + W)  # warm start so tick 0 is not garbage
        self.t = 0
        self.mass0 = self.mass()
        return self.obs()

    # ------------------------------------------------------------------ state
    def state(self):
        return (self.density, self.pot, self.cursor, self.stock)

    def load_state(self, s):
        self.density, self.pot, self.cursor, self.stock = s

    def detach_(self):
        self.density = self.density.detach()
        self.pot = self.pot.detach()
        self.cursor = self.cursor.detach()
        if self.stock is not None:
            self.stock = self.stock.detach()

    # ------------------------------------------------------------------ tick
    def relax_field(self, k: int | None = None):
        seed = core.cursor_seed(self.cursor, self.walls, self.p.seed_radius, self.p.LARGE)
        self.pot = core.relax(self.pot, seed, self.walls.unsqueeze(1),
                              k=int(self.p.K_RELAX if k is None else k), p=self.p)

    def step(self, action: torch.Tensor):
        """action: [B, 2, 2] cursor velocity per team, expected in [-1, 1].

        Returns (obs, reward, score) with reward = the change in team 0's share.
        """
        s_prev = score(self.density)
        args = (self.density, self.pot, self.cursor, self.walls, action, self.p,
                self.stock)
        if self.checkpoint and torch.is_grad_enabled():
            out = torch.utils.checkpoint.checkpoint(self._tick, *args,
                                                    use_reentrant=False)
        else:
            out = self._tick(*args)
        if self.stock is None:
            self.density, self.pot, self.cursor = out
        else:
            self.density, self.pot, self.cursor, self.stock = out
        self.t += 1
        s = score(self.density)
        return self.obs(), s - s_prev, s

    # ------------------------------------------------------------------ views
    def obs(self, team: int = 0) -> torch.Tensor:
        """[B, 3, H, W]. Team 1's view is the mirror image, so one policy plays both
        sides. Built by sim.observation so the density sim, the verbatim reference and
        any bridge into a real LW6 build all agree on the representation."""
        return observation(self.density[:, team], self.density[:, 1 - team], self.walls)

    def score(self):
        return score(self.density)

    def mass(self):
        return self.density.sum(dim=(1, 2, 3))

    def health(self):
        """Mean health fraction per cell per team, or None when not tracked."""
        if self.stock is None:
            return None
        return (self.stock / self.density.clamp_min(1e-9)).clamp(0.0, 1.0)
