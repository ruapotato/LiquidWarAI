"""Policy network, with two ways of controlling a cursor.

`velocity` (the human constraint): output a direction, the cursor moves at most
`MAX_CURSOR_SPEED` cells per round. This is what a hand can do, and it is the action
space every policy here used until now.

`target` (the bot constraint): output an absolute position and the cursor goes there.
This is what LW6's own `mod-brute` does -- `cursor.pos.x = lw6sys_random(shape.w)`,
anywhere on the map, every round -- and it is a strictly larger action space. A
velocity policy cannot express it, which is most of why capping brute's cursor turned
a 0.04 rout into a coin flip.

The head is spatial: a 1x1 convolution on the trunk's 16x16 feature map gives one
logit per 4x4 block of the board, and the action is a categorical draw over those
cells. That makes it discrete, so it has no gradient path for M3 -- the velocity head
stays for the differentiable arm.

Both modes go through `sample` / `evaluate_actions` / `act_deterministic`, so PPO does
not need to know which it is holding.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CursorPolicy(nn.Module):
    def __init__(self, cfg, action_mode: str = "velocity"):
        super().__init__()
        if action_mode not in ("velocity", "target"):
            raise ValueError(f"unknown action_mode {action_mode!r}")
        self.action_mode = action_mode
        ch = list(cfg.policy.channels)
        hid = int(cfg.policy.hidden)
        layers, prev = [], 3
        for c in ch:
            layers += [nn.Conv2d(prev, c, 3, stride=2, padding=1), nn.SiLU()]
            prev = c
        self.trunk = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(2)
        self.head = nn.Sequential(nn.Linear(prev * 4, hid), nn.SiLU())
        self.v = nn.Linear(hid, 1)
        # Velocity head. Kept even in target mode: it costs ~250 parameters and
        # keeps one state_dict shape for both.
        self.mu = nn.Linear(hid, 2)
        self.log_std = nn.Parameter(torch.full((2,), -0.5))
        nn.init.zeros_(self.mu.bias)
        self.mu.weight.data.mul_(0.01)
        # Target head: one logit per cell of the second-to-last feature map, which
        # at the default channel list is 16x16 for a 64x64 board -- one action per
        # 4x4 block. Tapping the last map instead would give 8x8, too coarse to aim.
        # Index and channel count both derive from the trunk so a different
        # `policy.channels` stays consistent; hard-coding either silently mismatches.
        self._spatial_depth = (len(ch) - 1) * 2    # after the second-to-last conv+SiLU
        self.spatial = nn.Conv2d(ch[len(ch) - 2], 1, 1)
        nn.init.zeros_(self.spatial.bias)
        self.spatial.weight.data.mul_(0.01)

    # ------------------------------------------------------------------ core
    def _features(self, obs):
        mid = self.trunk[: self._spatial_depth](obs)
        deep = self.trunk[self._spatial_depth :](mid)
        h = self.head(self.pool(deep).flatten(1))
        return mid, h

    def forward(self, obs):
        _mid, h = self._features(obs)
        return self.mu(h), self.v(h).squeeze(-1)

    def _dist(self, obs):
        mid, h = self._features(obs)
        value = self.v(h).squeeze(-1)
        if self.action_mode == "velocity":
            mu = self.mu(h)
            std = self.log_std.exp().expand_as(mu)
            return torch.distributions.Normal(mu, std), value, None
        logits = self.spatial(mid)                       # [B, 1, h, w]
        shape = logits.shape[-2:]
        return torch.distributions.Categorical(logits=logits.flatten(1)), value, shape

    # --------------------------------------------------------------- actions
    def _to_action(self, raw, shape):
        """Map a raw sample to what the environment takes, [B, 2] per team.

        velocity: tanh-squashed direction in [-1, 1].
        target:   normalised position in [0, 1], cell centres of the logit grid.
        """
        if self.action_mode == "velocity":
            return torch.tanh(raw)
        h, w = shape
        y = torch.div(raw, w, rounding_mode="floor").float()
        x = (raw % w).float()
        return torch.stack([(y + 0.5) / h, (x + 0.5) / w], dim=-1)

    def sample(self, obs):
        """Returns (action, raw, log_prob, value). `raw` is what evaluate_actions wants."""
        dist, value, shape = self._dist(obs)
        raw = dist.sample()
        logp = dist.log_prob(raw)
        if self.action_mode == "velocity":
            logp = logp.sum(-1)
        return self._to_action(raw, shape), raw, logp, value

    def evaluate_actions(self, obs, raw):
        dist, value, _shape = self._dist(obs)
        logp = dist.log_prob(raw)
        ent = dist.entropy()
        if self.action_mode == "velocity":
            logp, ent = logp.sum(-1), ent.sum(-1)
        return logp, ent, value

    def act_deterministic(self, obs):
        dist, _value, shape = self._dist(obs)
        if self.action_mode == "velocity":
            return torch.tanh(dist.mean)
        return self._to_action(dist.logits.argmax(-1), shape)

    # kept for the older call sites
    def dist(self, obs):
        d, v, _ = self._dist(obs)
        return d, v


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
