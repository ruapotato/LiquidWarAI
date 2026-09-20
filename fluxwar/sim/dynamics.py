"""The combat rule exactly as the original project spec wrote it, kept for the record.

Nothing in the simulator uses this. It is here because the spec's rule is subtly and
completely broken, and the demonstration is worth keeping:
tests/test_conservation.py::test_spec_combat_is_degenerate.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def combat_spec(density: torch.Tensor, rate_dt: float) -> torch.Tensor:
    """net = RATE * (dB * pA - dA * pB), with pX = conv2d(dX, 3x3 ones, no centre).

    Antisymmetric pointwise, and therefore looks fine. But for any symmetric kernel K,

        sum_x dB(x) * (K dA)(x)  ==  sum_x dA(x) * (K dB)(x)

    because both sides enumerate the same set of adjacent (A, B) pairs. The net
    transfer summed over the grid is identically zero: each team's total is conserved
    separately and the score is a constant 0.5 for the whole episode, no matter what
    either player does. Measured drift on a random state is ~1e-5, i.e. float noise.

    The replacement is sim/core.py::combat, derived from the LW6 attack rule.
    """
    B, T, H, W = density.shape
    k = torch.ones(1, 1, 3, 3, device=density.device, dtype=density.dtype)
    k[0, 0, 1, 1] = 0.0
    pressure = F.conv2d(density.reshape(B * T, 1, H, W), k, padding=1).reshape(B, T, H, W)
    a, b = density[:, 0], density[:, 1]
    net = rate_dt * (b * pressure[:, 0] - a * pressure[:, 1])
    return torch.stack([a + net, b - net], dim=1)
