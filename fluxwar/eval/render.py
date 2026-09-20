"""Density field -> mp4, for actually watching what the policies do."""
from __future__ import annotations

import numpy as np
import torch


TEAM_COLOUR = np.array([[232, 96, 96], [110, 150, 245]], np.float32) / 255.0
WALL_COLOUR = np.array([0.16, 0.17, 0.20], np.float32)


def frame(density, walls, cursor, capacity: float = 1.0, gamma: float = 0.6,
          health=None) -> np.ndarray:
    """One [H, W, 3] uint8 image.

    A cell is drawn the way the original game draws one: its *hue* is who owns it and
    its *brightness* is how contested it is. A cell held outright is a saturated team
    colour; a cell where the two teams are level is dark. So liquid in contact with
    enemy liquid darkens, and then comes back up in the other team's colour as
    ownership flips -- which is the visual signature of the mechanic. Total density
    sets the alpha against the background, so a thin skirmish line reads as faint and
    a packed blob as solid.
    """
    d = density.detach().float().cpu().numpy()
    w = walls.detach().float().cpu().numpy()
    total = d[0] + d[1]
    occupancy = np.clip(total / max(capacity, 1e-9), 0.0, 1.0) ** gamma
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(total > 1e-12, d[0] / np.maximum(total, 1e-12), 0.5)
    hue = TEAM_COLOUR[1] + (TEAM_COLOUR[0] - TEAM_COLOUR[1]) * share[..., None]
    if health is None:
        # In the density model a cell has no health of its own; how contested it is
        # plays the same role. A cell held outright is saturated, a cell where the two
        # teams are level is dark.
        contest = 0.35 + 0.65 * np.abs(2.0 * share - 1.0)
    else:
        # In the discrete model each fighter carries real health, so show that: a
        # fighter under attack darkens, then comes back up in the other team's colour
        # as it flips. This is the visual signature of the mechanic.
        h = health.detach().float().cpu().numpy() if hasattr(health, "detach") else np.asarray(health)
        contest = 0.35 + 0.65 * np.clip(h, 0.0, 1.0)
    img = hue * contest[..., None] * occupancy[..., None]
    img = img + WALL_COLOUR * (1.0 - occupancy)[..., None] * 0.55
    img[w < 0.5] = WALL_COLOUR

    c = cursor.detach().float().cpu().numpy()
    for t, col in enumerate(((1.0, 0.82, 0.82), (0.82, 0.88, 1.0))):
        y = min(max(int(round(c[t, 0])), 1), w.shape[0] - 2)
        x = min(max(int(round(c[t, 1])), 1), w.shape[1] - 2)
        img[y - 1 : y + 2, x - 1 : x + 2] = col
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


@torch.no_grad()
def record(agent_a, agent_b, cfg, path, *, episode_len=512, device="cuda", seed=0,
           game=0, scale=6, fps=30):
    import imageio.v2 as imageio

    from ..sim.env import FluxWar

    env = FluxWar(cfg, device, seed=seed, B=max(game + 1, 4))
    frames = []
    for _ in range(episode_len):
        action = torch.stack([agent_a.act(env, 0), agent_b.act(env, 1)], 1)
        env.step(action)
        f = frame(env.density[game], env.walls[game], env.cursor[game],
                  capacity=float(cfg.sim.get('capacity', 1.0)) or 1.0)
        frames.append(np.kron(f, np.ones((scale, scale, 1), np.uint8)))
    imageio.mimsave(path, frames, fps=fps, macro_block_size=1)
    return path, float(env.score()[game])
