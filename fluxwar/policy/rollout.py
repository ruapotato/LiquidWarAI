"""Shared self-play plumbing for M2 and M3: the frozen-opponent checkpoint pool."""
from __future__ import annotations

import copy
import random

import torch

from .agents import SCRIPTED, NetAgent


class OpponentPool:
    """Last N checkpoints plus the scripted agents, sampled uniformly.

    Keeping the scripted agents in the pool stops self-play from drifting into a
    private equilibrium that loses to the thing we actually measure against.
    """

    def __init__(self, cfg, net, size: int = 8, include_scripted=("chase_enemy", "hold_own")):
        self.size = int(size)
        self.ckpts: list = []
        self.scripted = [SCRIPTED[n] for n in include_scripted]
        self.snapshot(net)

    def snapshot(self, net):
        clone = copy.deepcopy(net).eval()
        for p in clone.parameters():
            p.requires_grad_(False)
        self.ckpts.append(clone)
        if len(self.ckpts) > self.size:
            self.ckpts.pop(0)

    def sample(self, rng: random.Random):
        choices = self.scripted + [NetAgent(c, stochastic=True, name="ckpt") for c in self.ckpts]
        return rng.choice(choices)
