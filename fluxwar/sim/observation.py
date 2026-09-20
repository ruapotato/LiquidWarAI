"""The single definition of what a policy sees and what it emits.

Both the density sim, the verbatim reference wrapper, and any bridge into a real
Liquid War 6 build must produce observations through this function, so a policy cannot
quietly learn a representation that the real game cannot supply.

Everything here is derivable from LW6's own state: per-cell team occupancy comes from
`_lw6ker_map_state_get_fighter_id` plus `fighters[id].team_color`, and the wall mask
from whether a cell has a zone id at all.
"""
from __future__ import annotations

import torch

N_CHANNELS = 3


def observation(own: torch.Tensor, enemy: torch.Tensor, walls: torch.Tensor) -> torch.Tensor:
    """[B, 3, H, W] = (own density, enemy density, walls), mass-normalised.

    own/enemy: [B, H, W] fighters per cell for each side, from the acting team's point
    of view. walls: [B, H, W], 1.0 open, 0.0 wall.

    Normalising by mean density per cell rather than by a fixed constant is what makes
    the same weights usable at a different map size or population: the network sees
    occupancy relative to the average, not raw counts.
    """
    total = (own + enemy).sum(dim=(1, 2), keepdim=True).clamp_min(1e-12)
    scale = total / (own.shape[-2] * own.shape[-1])
    return torch.stack([own / scale, enemy / scale, walls], dim=1)


def action_to_cursor_target(cursor_yx, action, max_cursor_speed):
    """Turn a policy action into the absolute cursor target LW6's bot API wants.

    `lw6bot_next_move` returns an (x, y) for the cursor to be placed at, so a velocity
    action becomes current position plus velocity. Note the axis order: this codebase
    is (y, x) throughout and LW6's API is (x, y).
    """
    target = cursor_yx + action * max_cursor_speed
    return target
