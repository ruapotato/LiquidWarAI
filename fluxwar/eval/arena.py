"""Head-to-head evaluation with sides swapped, plus a small Elo helper."""
from __future__ import annotations

import math

import torch

from ..sim.env import FluxWar


@torch.no_grad()
def play(agent_a, agent_b, cfg, *, games: int, episode_len: int, device="cuda", seed: int = 0,
         swap: bool = False):
    """`games` parallel games. Returns final team-A scores, [games]."""
    env = FluxWar(cfg, device, seed=seed, B=games)
    agent_a.reset(env)
    agent_b.reset(env)
    ta, tb = (1, 0) if swap else (0, 1)
    for _ in range(episode_len):
        a = agent_a.act(env, ta)
        b = agent_b.act(env, tb)
        action = torch.stack([a, b], 1) if not swap else torch.stack([b, a], 1)
        env.step(action)
    s = env.score()
    return s if not swap else 1.0 - s


@torch.no_grad()
def head_to_head(agent_a, agent_b, cfg, *, games=200, episode_len=512, device="cuda", seed=0):
    """Half the games with A as team 0, half swapped, to control for map asymmetry."""
    half = games // 2
    s1 = play(agent_a, agent_b, cfg, games=half, episode_len=episode_len, device=device,
              seed=seed, swap=False)
    s2 = play(agent_a, agent_b, cfg, games=games - half, episode_len=episode_len,
              device=device, seed=seed, swap=True)
    s = torch.cat([s1, s2])
    # Games where the two armies never make contact end at exactly 0.5; scoring those
    # as losses (or wins) on float noise would make a pure turtle look like a 84%
    # winner. They are draws.
    eps = 1e-3
    win = (s > 0.5 + eps).float()
    draw = (s - 0.5).abs().le(eps).float()
    return dict(
        win_rate=float(win.mean()),
        draw_rate=float(draw.mean()),
        loss_rate=float(1.0 - win.mean() - draw.mean()),
        decisive_win_rate=float(win.sum() / max(float((1 - draw).sum()), 1.0)),
        mean_share=float(s.mean()),
        win_rate_side0=float((s1 > 0.5 + eps).float().mean()),
        win_rate_side1=float((s2 > 0.5 + eps).float().mean()),
        games=games,
    )


def elo_update(ra, rb, score_a, k=32.0):
    ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
    return ra + k * (score_a - ea), rb + k * ((1 - score_a) - (1 - ea))


def round_robin(agents, cfg, *, games=64, episode_len=512, device="cuda"):
    """Elo over all pairs. Non-transitivity shows up as a cycle in the pairwise table."""
    names = [a.name for a in agents]
    elo = {n: 1200.0 for n in names}
    table = {}
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            r = head_to_head(agents[i], agents[j], cfg, games=games,
                             episode_len=episode_len, device=device)
            table[(names[i], names[j])] = r["decisive_win_rate"]
            elo[names[i]], elo[names[j]] = elo_update(elo[names[i]], elo[names[j]], r["mean_share"])
    return elo, table
