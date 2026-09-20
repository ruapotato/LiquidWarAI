"""M3 prerequisite: gradients actually flow from the score back to the cursor.

The likeliest silent failure in the whole project is seeding the potential field with a
hard index, which has exactly zero gradient w.r.t. the cursor.
"""
import torch

from fluxwar.config import load
from fluxwar.sim.core import Params, cursor_seed, relax, tick
from fluxwar.sim.env import FluxWar


def _open_walls(B, H, W, device):
    w = torch.ones(B, H, W, device=device)
    w[:, 0], w[:, -1], w[:, :, 0], w[:, :, -1] = 0, 0, 0, 0
    return w


def test_potential_has_nonzero_cursor_gradient(device):
    cfg = load(**{"sim.soft_field": True})
    p = Params.from_cfg(cfg)
    walls = _open_walls(2, 24, 24, device)
    cursor = torch.tensor([[[8.3, 9.7], [16.2, 15.1]]] * 2, device=device, requires_grad=True)
    seed = cursor_seed(cursor, walls, p.seed_radius, p.LARGE)
    pot = torch.full((2, 2, 24, 24), p.LARGE, device=device)
    pot = relax(pot, seed, walls.unsqueeze(1), k=16, p=p)
    pot.sum().backward()
    g = cursor.grad.abs()
    assert torch.isfinite(g).all()
    assert g.max().item() > 1e-6, "d(potential)/d(cursor) is zero -- the seed is not differentiable"


def _fd_check(steps=24, eps=1e-5, **overrides):
    """Analytical vs central-difference d(score)/d(action). Returns the max rel error.

    float64 on CPU: in float32 the score is ~0.5 and a finite-difference step moves it
    by ~1e-8, below the representable resolution -- the numeric derivative comes back
    quantised to powers of two and tells you nothing.
    """
    cfg = load(**{"sim.soft_field": True, "sim.K_RELAX": 4, "sim.dtype": "float64",
                  **overrides})
    p = Params.from_cfg(cfg)
    H = W = 20
    walls = _open_walls(1, H, W, "cpu").double()
    ys = torch.arange(H).view(1, 1, H, 1).double()
    xs = torch.arange(W).view(1, 1, 1, W).double()
    # Armies overlapping and already fighting, at a density near `capacity`. Place
    # them further apart and d(score)/d(action) over a short window is ~1e-9, which is
    # the float64 noise floor -- the test then measures nothing.
    centres = torch.tensor([[[9.0, 9.0], [10.0, 11.0]]]).double()
    d2 = ((ys - centres[..., 0].view(1, 2, 1, 1)) ** 2
          + (xs - centres[..., 1].view(1, 2, 1, 1)) ** 2)
    dens = torch.exp(-d2 / 8.0) * walls.unsqueeze(1)
    dens = dens / dens.amax(dim=(2, 3), keepdim=True)
    pot0 = torch.full((1, 2, H, W), p.LARGE, dtype=torch.float64)

    def rollout(action):
        d, pot, cur = dens.clone(), pot0.clone(), centres.clone()
        for _ in range(steps):
            d, pot, cur = tick(d, pot, cur, walls, action, p)
        return d[:, 0].sum() / d.sum()

    a = torch.tensor([[[0.4, -0.3], [-0.2, 0.5]]], dtype=torch.float64, requires_grad=True)
    rollout(a).backward()
    ana = a.grad.clone().flatten()
    num = torch.zeros_like(ana)
    with torch.no_grad():
        for i in range(ana.numel()):
            for sign in (+1, -1):
                b = a.detach().clone().flatten()
                b[i] += sign * eps
                num[i] += sign * rollout(b.view_as(a)) / (2 * eps)
    mask = ana.abs() > 1e-9
    assert mask.any(), "analytical gradient is identically zero"
    return ((ana[mask] - num[mask]).abs() / ana[mask].abs()).max().item(), ana, num


def test_score_gradient_finite_difference():
    """Checked at the M3 (soft.yaml) temperature, where the FD is actually trustworthy."""
    torch.set_default_dtype(torch.float64)
    try:
        rel, ana, num = _fd_check(**{"sim.temp": 0.5})
        assert rel < 1e-3, f"gradient mismatch: analytical {ana.tolist()} vs numeric {num.tolist()}"
    finally:
        torch.set_default_dtype(torch.float32)


def test_score_gradient_finite_difference_at_default_temp():
    """Same check at the fidelity-tuned temp = 0.1.

    The bar is looser on purpose. At temp = 0.1 the advection softmax is nearly a step
    function, so its curvature goes as 1/temp**2 and the central difference carries a
    truncation error of order eps**2 / temp**3 -- about 1% of a gradient this size.
    The analytical value is the accurate one here; this asserts they still agree to
    the precision the finite difference can actually deliver.
    """
    torch.set_default_dtype(torch.float64)
    try:
        rel, ana, num = _fd_check()
        assert rel < 0.05, f"gradient mismatch: analytical {ana.tolist()} vs numeric {num.tolist()}"
    finally:
        torch.set_default_dtype(torch.float32)


def test_gradients_are_finite_through_a_bptt_window(device):
    cfg = load(**{"sim.B": 4, "sim.soft_field": True})
    env = FluxWar(cfg, device, seed=4)
    a = torch.zeros(env.B, 2, 2, device=device, requires_grad=True)
    total = 0.0
    for _ in range(32):
        _, r, _ = env.step(torch.tanh(a))
        total = total + r.sum()
    total.backward()
    assert torch.isfinite(a.grad).all()
    assert a.grad.abs().max().item() > 0
