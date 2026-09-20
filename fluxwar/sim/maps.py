"""Procedural map generation. Returns walls: [B, H, W], 1.0 = open, 0.0 = wall."""
from __future__ import annotations

import torch


def _connected_component_mask(open_mask: torch.Tensor, seed_yx) -> torch.Tensor:
    """Flood fill from seed over an [H, W] bool mask, on whatever device it lives on."""
    reach = torch.zeros_like(open_mask)
    reach[seed_yx[0], seed_yx[1]] = open_mask[seed_yx[0], seed_yx[1]]
    for _ in range(open_mask.shape[0] * 2 + open_mask.shape[1] * 2):
        prev = reach
        grown = reach.clone()
        grown[1:, :] |= reach[:-1, :]
        grown[:-1, :] |= reach[1:, :]
        grown[:, 1:] |= reach[:, :-1]
        grown[:, :-1] |= reach[:, 1:]
        reach = grown & open_mask
        if bool((reach == prev).all()):
            break
    return reach


def _blobs(H, W, wall_frac, gen, device):
    """Low-frequency noise thresholded into cave-like walls."""
    small = torch.rand(1, 1, max(H // 8, 2), max(W // 8, 2), generator=gen, device=device)
    field = torch.nn.functional.interpolate(small, size=(H, W), mode="bicubic", align_corners=False)[0, 0]
    thresh = torch.quantile(field.flatten(), float(wall_frac))
    return (field > thresh).float()


def _rooms(H, W, wall_frac, gen, device):
    """Axis-aligned walls with doorways: produces real geodesic detours."""
    walls = torch.ones(H, W, device=device)
    n_walls = max(int(wall_frac * 8), 1)
    for _ in range(n_walls):
        horizontal = bool(torch.rand(1, generator=gen, device=device) < 0.5)
        if horizontal:
            y = int(torch.randint(H // 8, H - H // 8, (1,), generator=gen, device=device))
            x0 = int(torch.randint(0, W // 2, (1,), generator=gen, device=device))
            x1 = int(torch.randint(W // 2, W, (1,), generator=gen, device=device))
            walls[y : y + max(H // 32, 1), x0:x1] = 0.0
            door = int(torch.randint(x0, max(x0 + 1, x1), (1,), generator=gen, device=device))
            walls[y : y + max(H // 32, 1), door : door + max(W // 10, 2)] = 1.0
        else:
            x = int(torch.randint(W // 8, W - W // 8, (1,), generator=gen, device=device))
            y0 = int(torch.randint(0, H // 2, (1,), generator=gen, device=device))
            y1 = int(torch.randint(H // 2, H, (1,), generator=gen, device=device))
            walls[y0:y1, x : x + max(W // 32, 1)] = 0.0
            door = int(torch.randint(y0, max(y0 + 1, y1), (1,), generator=gen, device=device))
            walls[door : door + max(H // 10, 2), x : x + max(W // 32, 1)] = 1.0
    return walls


def make_maps(B: int, cfg, device="cuda", generator=None) -> torch.Tensor:
    """Batch of maps. Border cells are always walls. Guaranteed single connected region."""
    H, W = cfg.sim.H, cfg.sim.W
    kind = cfg.map.kind
    gen = generator
    out = torch.empty(B, H, W, device=device)
    for b in range(B):  # generation is setup-time only; the tick loop is fully batched
        if kind == "open":
            walls = torch.ones(H, W, device=device)
        elif kind == "blobs":
            walls = _blobs(H, W, cfg.map.wall_frac, gen, device)
        elif kind == "rooms":
            walls = _rooms(H, W, cfg.map.wall_frac, gen, device)
        else:
            raise ValueError(f"unknown map kind {kind}")
        walls[0, :] = 0.0
        walls[-1, :] = 0.0
        walls[:, 0] = 0.0
        walls[:, -1] = 0.0
        # Keep only the connected component containing the centre-ish largest region.
        om = walls > 0.5
        if not bool(om[H // 2, W // 2]):
            idx = int(om.flatten().float().argmax())
            seed = (idx // W, idx % W)
        else:
            seed = (H // 2, W // 2)
        walls = _connected_component_mask(om, seed).float()
        out[b] = walls
    return out


def spawn_points(walls: torch.Tensor, generator=None, sep_frac=(0.35, 0.95)):
    """Two open cells per map, separated by a random fraction of the map's diameter.

    Always spawning at the diameter means the first ~150 ticks of every episode are
    dead time with no contact and therefore no reward; drawing the separation gives
    the learner short games as well as long ones. Returns [B, 2, 2] float (y, x).
    """
    B, H, W = walls.shape
    device = walls.device
    pts = torch.zeros(B, 2, 2, device=device)
    ys = torch.arange(H, device=device).view(H, 1).expand(H, W)
    xs = torch.arange(W, device=device).view(1, W).expand(H, W)
    lo, hi = sep_frac
    for b in range(B):
        om = walls[b] > 0.5
        oy, ox = ys[om].float(), xs[om].float()
        n = oy.numel()
        i = int(torch.randint(0, n, (1,), generator=generator, device=device))
        p0 = torch.stack([oy[i], ox[i]])
        d = ((oy - p0[0]) ** 2 + (ox - p0[1]) ** 2).sqrt()
        frac = lo + (hi - lo) * float(torch.rand(1, generator=generator, device=device))
        j = int((d - frac * float(d.max())).abs().argmin())
        pts[b, 0] = p0
        pts[b, 1] = torch.stack([oy[j], ox[j]])
    return pts
