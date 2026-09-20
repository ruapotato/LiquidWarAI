"""Agent interface: given an env and a team index, produce that team's cursor velocity.

Everything is batched -- an agent sees the whole batch of games at once.
"""
from __future__ import annotations

import torch


class Agent:
    name = "agent"

    def act(self, env, team: int) -> torch.Tensor:  # -> [B, 2] in [-1, 1]
        raise NotImplementedError

    def reset(self, env):
        pass


def _centroid(d: torch.Tensor) -> torch.Tensor:
    """Mass centroid of [B, H, W]. Returns [B, 2] as (y, x)."""
    B, H, W = d.shape
    ys = torch.arange(H, device=d.device, dtype=d.dtype).view(1, H)
    xs = torch.arange(W, device=d.device, dtype=d.dtype).view(1, W)
    m = d.sum((1, 2)).clamp_min(1e-12)
    return torch.stack([(d.sum(2) * ys).sum(1) / m, (d.sum(1) * xs).sum(1) / m], dim=1)


def _toward(cursor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    v = target - cursor
    n = v.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return v / torch.maximum(n, torch.ones_like(n))


class ChaseEnemy(Agent):
    """The scripted baseline: drive the cursor at the enemy's centre of mass."""
    name = "chase_enemy"

    def act(self, env, team):
        return _toward(env.cursor[:, team], _centroid(env.density[:, 1 - team]))


class HoldOwn(Agent):
    """Sit on your own centre of mass: keeps the army balled up."""
    name = "hold_own"

    def act(self, env, team):
        return _toward(env.cursor[:, team], _centroid(env.density[:, team]))


class Stationary(Agent):
    name = "stationary"

    def act(self, env, team):
        return torch.zeros(env.B, 2, device=env.device, dtype=env.dtype)


class NetAgent(Agent):
    """Wraps a CursorPolicy. `stochastic=False` for evaluation."""
    name = "net"

    def __init__(self, net, stochastic: bool = False, name: str | None = None):
        self.net = net
        self.stochastic = stochastic
        if name:
            self.name = name

    def act(self, env, team):
        """Whatever the policy's action mode produces -- a velocity or an absolute
        target -- the env interprets it, so this does not need to know which."""
        obs = env.obs(team)
        if self.stochastic:
            action, _raw, _logp, _v = self.net.sample(obs)
            return action
        return self.net.act_deterministic(obs)


class Lean(Agent):
    """Park the cursor `k` cells from your own centroid, towards the enemy.

    A one-parameter family spanning the obvious strategies: k = 0 is HoldOwn (turtle),
    large k approaches ChaseEnemy (charge). Used to probe whether the game has any
    strategic structure at all, or whether one end of the axis just dominates.
    """

    def __init__(self, k: float):
        self.k = float(k)
        self.name = f"lean{k:g}"

    def act(self, env, team):
        own = _centroid(env.density[:, team])
        foe = _centroid(env.density[:, 1 - team])
        v = foe - own
        v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return _toward(env.cursor[:, team], own + self.k * v)


SCRIPTED = {a.name: a for a in (ChaseEnemy(), HoldOwn(), Stationary())}
SCRIPTED.update({f"lean{k:g}": Lean(k) for k in (0, 2, 5, 10, 20)})
