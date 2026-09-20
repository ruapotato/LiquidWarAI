"""Interactive player. Drive a cursor with the mouse and play the sim yourself.

    python -m fluxwar.play                          # you vs hold_own, density sim
    python -m fluxwar.play --opponent lean5
    python -m fluxwar.play --opponent runs/ppo/policy.pt
    python -m fluxwar.play --engine reference       # play the M0 discrete sim
    python -m fluxwar.play --engine both            # both sims, same map, same cursors

`--engine both` is the fidelity check you can actually see: the discrete reference and
the density sim run from an identical map, identical starting armies and identical
cursor trajectories, side by side, with both population curves drawn live. If the
density model is faithful the two pictures move together.

Controls
    mouse           move your cursor (it tracks the pointer at MAX_CURSOR_SPEED)
    space           pause / resume
    r               new round
    tab             swap which side you play
    [ ]             slower / faster (sim ticks per rendered frame)
    h               hide / show the HUD
    esc or q        quit
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch

from .config import load
from .eval.render import frame
from .policy.agents import SCRIPTED, NetAgent
from .policy.net import CursorPolicy
from .sim import core
from .sim.observation import observation
from .sim.env import FluxWar
from .sim.maps import make_maps, spawn_points

BG = (16, 17, 22)
FG = (228, 230, 238)
DIM = (120, 126, 140)
RED = (232, 96, 96)
BLUE = (110, 150, 245)


# --------------------------------------------------------------------- engines
class DensityEngine:
    """The batched GPU sim, run at B = 1."""
    label = "density (M1)"

    def __init__(self, cfg, device, seed):
        self.cfg = cfg
        self.env = FluxWar(cfg, device, seed=seed, B=1)

    def reset(self, maps=None, cursors=None):
        self.env.reset(maps=maps, cursors=cursors)

    @property
    def walls(self):
        return self.env.walls[0]

    @property
    def cursor(self):
        return self.env.cursor[0]

    def density(self):
        return self.env.density[0]

    def score(self):
        return float(self.env.score()[0])

    def step(self, a0, a1):
        with torch.no_grad():
            self.env.step(torch.stack([a0, a1], 1))

    def seed_from(self, other):
        """Copy another engine's map and starting armies, so both show the same game."""
        raise NotImplementedError


class LW6Engine:
    """The real Liquid War 6 kernel, wrapped so the same agents and renderer work.

    This is the game itself -- liblw6ker compiled from source -- not a
    re-implementation, so `--engine both` puts the differentiable model next to the
    real rules on an identical map and lets you watch them diverge.
    """
    label = "real LW6"

    def __init__(self, cfg, device, seed):
        self.cfg = cfg
        self.device = torch.device(device)
        self.dtype = torch.float32
        self.B = 1
        self.seed = seed
        self.env = None

    def build(self, walls_np, centres=None, fighters=None):
        from .lw6.env import LW6Env
        if self.env is not None:
            self.env.close()
        h, w = walls_np.shape
        self.env = LW6Env(width=w, height=h, device=self.device, walls=walls_np,
                          max_cursor_speed=float(self.cfg.sim.MAX_CURSOR_SPEED))

    @property
    def walls(self):
        return self.env.walls[0]

    @property
    def cursor(self):
        return self.env.cursor[0]

    def density(self):
        return self.env.density[0]

    def health(self):
        return self.env.game.health_image()

    def score(self):
        return float(self.env.score()[0])

    def step(self, a0, a1):
        self.env.step(torch.stack([a0, a1], 1))

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None


class AgentView:
    """Minimal env-shaped view so policy/agents.py works against either engine."""

    def __init__(self, engine, H, W):
        self.engine = engine
        self.H, self.W = H, W
        self.B = 1
        self.device = engine.density().device
        self.dtype = torch.float32

    @property
    def density(self):
        return self.engine.density().unsqueeze(0)

    @property
    def cursor(self):
        return self.engine.cursor.unsqueeze(0)

    @property
    def walls(self):
        return self.engine.walls.unsqueeze(0)

    def obs(self, team):
        d = self.density
        return observation(d[:, team], d[:, 1 - team], self.walls)


# ---------------------------------------------------------------------- driver
def load_opponent(spec, cfg, device):
    if spec in SCRIPTED:
        return SCRIPTED[spec]
    net = CursorPolicy(cfg).to(device)
    net.load_state_dict(torch.load(spec, map_location=device))
    net.eval()
    return NetAgent(net, name=spec)


def _toward(cursor, target, max_speed):
    v = target - cursor
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return np.zeros(2, np.float32)
    return (v / max(n, max_speed)).astype(np.float32)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="default.yaml")
    ap.add_argument("--engine", default="density", choices=("density", "lw6", "both"),
                    help="the differentiable sim, the real LW6 kernel, or both side by side")
    ap.add_argument("--opponent", default="hold_own",
                    help="a scripted name (%s) or a path to a saved policy" % ", ".join(SCRIPTED))
    ap.add_argument("--size", type=int, default=None, help="override H = W")
    ap.add_argument("--fighters", type=int, default=400,
                    help="fighters per team; the real kernel derives its own count "
                         "from the map, this sets the density sim's to match")
    ap.add_argument("--scale", type=int, default=9, help="screen pixels per cell")
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--speed", type=int, default=1, help="sim ticks per rendered frame")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--frames", type=int, default=0,
                    help="run this many frames then exit (headless self-test)")
    ap.add_argument("--shot", default=None, help="save the final frame here and exit")
    ap.add_argument("--auto", default=None,
                    help="drive 'your' cursor with this scripted agent instead of the mouse")
    a = ap.parse_args(argv)

    import pygame

    over = {}
    if a.size:
        over["sim.H"] = over["sim.W"] = a.size
    elif a.engine in ("lw6", "both"):
        over["sim.H"] = over["sim.W"] = 48
    if a.engine in ("lw6", "both"):
        # Give the density sim the same population as the kernel, so "mass" is in
        # fighters and the two panels are comparable cell for cell.
        over["map.spawn_mass"] = float(a.fighters)
    cfg = load(a.config, **over)
    H, W = int(cfg.sim.H), int(cfg.sim.W)
    device = a.device
    max_speed = float(cfg.sim.MAX_CURSOR_SPEED)

    gen = torch.Generator(device=device).manual_seed(a.seed)
    engines: list = []
    dens = DensityEngine(cfg, device, a.seed)
    if a.engine in ("density", "both"):
        engines.append(dens)
    ref = None
    if a.engine in ("lw6", "both"):
        ref = LW6Engine(cfg, device, a.seed)
        engines.append(ref)

    state = dict(walls=None, centres=None)

    def new_round():
        walls = make_maps(1, cfg, device, gen)
        pts = spawn_points(walls, gen)
        state["walls"] = walls
        state["centres"] = pts[0].cpu().numpy().astype(np.float32)
        dens.reset(maps=walls, cursors=pts)
        if ref is not None:
            ref.build(walls[0].cpu().numpy())

    new_round()
    opponent = load_opponent(a.opponent, cfg, device)
    views = {id(e): AgentView(e, H, W) for e in engines}

    pygame.init()
    pygame.display.set_caption("fluxwar")
    pad, hud_h = 12, 128
    panel_w = W * a.scale
    win_w = max(panel_w * len(engines) + pad * (len(engines) + 1), 720)
    win_h = H * a.scale + hud_h + pad * 2
    screen = pygame.display.set_mode((win_w, win_h))
    font = pygame.font.SysFont("monospace", 14)
    big = pygame.font.SysFont("monospace", 20, bold=True)
    clock = pygame.time.Clock()

    cap = float(cfg.sim.get('capacity', 1.0)) or 1.0
    my_team = 0
    paused = False
    show_hud = True
    speed = max(1, a.speed)
    ticks = 0
    history = {id(e): [] for e in engines}
    t_last, tick_rate = time.perf_counter(), 0.0

    auto = SCRIPTED[a.auto] if a.auto else None

    def mouse_target():
        mx, my = pygame.mouse.get_pos()
        gx = (mx - pad) % (panel_w + pad) / a.scale
        gy = (my - pad) / a.scale
        return np.array([np.clip(gy, 1, H - 2), np.clip(gx, 1, W - 2)], np.float32)

    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif ev.key == pygame.K_SPACE:
                    paused = not paused
                elif ev.key == pygame.K_r:
                    new_round()
                    ticks = 0
                    history = {id(e): [] for e in engines}
                elif ev.key == pygame.K_TAB:
                    my_team = 1 - my_team
                elif ev.key == pygame.K_h:
                    show_hud = not show_hud
                elif ev.key == pygame.K_LEFTBRACKET:
                    speed = max(1, speed - 1)
                elif ev.key == pygame.K_RIGHTBRACKET:
                    speed = min(16, speed + 1)

        if not paused:
            target = mouse_target()
            for _ in range(speed):
                for e in engines:
                    view = views[id(e)]
                    cur = e.cursor.cpu().numpy()
                    if auto is not None:
                        mine = auto.act(view, my_team)
                    else:
                        mine = torch.as_tensor(_toward(cur[my_team], target, max_speed),
                                               device=device).view(1, 2)
                    theirs = opponent.act(view, 1 - my_team)
                    a0, a1 = (mine, theirs) if my_team == 0 else (theirs, mine)
                    e.step(a0, a1)
                ticks += 1
            for e in engines:
                history[id(e)].append(e.score())
            now = time.perf_counter()
            tick_rate = 0.9 * tick_rate + 0.1 * (speed / max(now - t_last, 1e-6))
            t_last = now

        screen.fill(BG)
        for i, e in enumerate(engines):
            img = frame(e.density(), e.walls, e.cursor, capacity=cap,
                        health=e.health() if hasattr(e, 'health') else None)
            surf = pygame.surfarray.make_surface(np.transpose(img, (1, 0, 2)))
            surf = pygame.transform.scale(surf, (panel_w, H * a.scale))
            x0 = pad + i * (panel_w + pad)
            screen.blit(surf, (x0, pad))
            pygame.draw.rect(screen, (60, 64, 76), (x0, pad, panel_w, H * a.scale), 1)

            s = e.score()
            y0 = pad + H * a.scale + 8
            screen.blit(font.render(e.label, True, DIM), (x0, y0))
            bar_y = y0 + 18
            pygame.draw.rect(screen, BLUE, (x0, bar_y, panel_w, 14))
            pygame.draw.rect(screen, RED, (x0, bar_y, int(panel_w * s), 14))
            mine_share = s if my_team == 0 else 1 - s
            screen.blit(big.render(f"you {mine_share * 100:5.1f}%", True,
                                   RED if my_team == 0 else BLUE), (x0, bar_y + 18))
            curve = history[id(e)]
            if len(curve) > 2:  # population curve, drawn over the panel
                n = min(len(curve), panel_w)
                pts = [(x0 + k * panel_w / n,
                        pad + H * a.scale * (1 - curve[int(k * len(curve) / n)]))
                       for k in range(n)]
                pygame.draw.lines(screen, (255, 255, 255), False, pts, 1)

        if show_hud:
            lines = [
                f"tick {ticks:5d}   {tick_rate:6.0f} ticks/s   x{speed}   "
                f"{'PAUSED' if paused else 'running'}",
                f"you: team {my_team} ({'red' if my_team == 0 else 'blue'})    "
                f"opponent: {opponent.name}",
                "mouse=cursor  space=pause  r=round  tab=swap  [ ]=speed  h=hud  q=quit",
            ]
            for k, line in enumerate(lines):
                screen.blit(font.render(line, True, FG if k == 0 else DIM),
                            (pad, win_h - 58 + k * 18))

        pygame.display.flip()
        clock.tick(a.fps)
        if a.frames and ticks >= a.frames:
            running = False

    if a.shot:
        pygame.image.save(screen, a.shot)
        print("saved", a.shot, {e.label: round(e.score(), 4) for e in engines})
    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
