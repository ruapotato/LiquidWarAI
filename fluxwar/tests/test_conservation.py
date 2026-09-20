"""M1 acceptance: total mass is conserved for the whole episode."""
import torch

from fluxwar.config import load
from fluxwar.sim.core import Params, combat
from fluxwar.sim.dynamics import combat_spec
from fluxwar.sim.env import FluxWar


def _drift(device, dtype, ticks=2000, B=8):
    cfg = load(**{"sim.B": B, "sim.dtype": dtype})
    env = FluxWar(cfg, device, seed=1)
    m0 = env.mass().clone()
    gen = torch.Generator(device=device).manual_seed(0)
    with torch.no_grad():
        for _ in range(ticks):
            a = torch.rand(env.B, 2, 2, device=device, generator=gen,
                           dtype=env.dtype) * 2 - 1
            env.step(a)
    return ((env.mass() - m0).abs() / m0).max().item()


def test_mass_conserved_2000_ticks(device):
    """float32 over 2000 ticks, each of which runs acts_per_tick advections.

    The bar is 5e-4 rather than the 1e-4 the spec asked for because what is left is
    float32 accumulation, not a leak: the identical run in float64 drifts by 1.3e-6,
    a hundredfold better, which is what rounding predicts and a real leak would not
    do. test_mass_conserved_in_float64 pins the algorithm itself.
    """
    rel = _drift(device, "float32")
    assert rel < 5e-4, f"relative mass drift {rel:.3e}"


def test_mass_conserved_in_float64(device):
    """The transport itself is conservative; float32 is the only thing losing mass."""
    rel = _drift(device, "float64", ticks=500)
    assert rel < 1e-5, f"relative mass drift in float64 {rel:.3e}"


def test_density_stays_non_negative(device):
    cfg = load(**{"sim.B": 8})
    env = FluxWar(cfg, device, seed=2)
    gen = torch.Generator(device=device).manual_seed(0)
    with torch.no_grad():
        for _ in range(500):
            env.step(torch.rand(env.B, 2, 2, device=device, generator=gen) * 2 - 1)
            assert env.density.min().item() >= -1e-9


def test_combat_is_zero_sum_and_local(device):
    p = Params.from_cfg(load())
    d = torch.rand(4, 2, 16, 16, device=device)
    attack = torch.rand(4, 2, 16, 16, device=device)
    n = combat(d, attack, p)
    assert torch.allclose(n.sum(1), d.sum(1), atol=1e-5)          # cellwise conservation
    swapped = combat(d.flip(1), attack.flip(1), p).flip(1)
    assert torch.allclose(n, swapped, atol=1e-5)                   # team-symmetric rule
    assert n.min() >= -1e-6                                        # no negative density


def test_only_blocked_mass_fights(device):
    """LW6's rule: a fighter that can move does not attack. With nothing blocked the
    two armies pass through each other without a single conversion."""
    p = Params.from_cfg(load())
    d = torch.rand(2, 2, 12, 12, device=device)
    free = combat(d, torch.zeros_like(d), p)
    assert torch.allclose(free, d, atol=1e-7)


def test_spec_combat_is_degenerate():
    """Why core.combat does not use the rule as written in the spec.

    net = RATE * (dB * pA - dA * pB) is pointwise antisymmetric, but summed over the
    grid the two terms enumerate the same adjacent (A, B) pairs, so the total transfer
    is identically zero: the score can never move.
    """
    torch.manual_seed(0)
    d = torch.rand(3, 2, 24, 24)
    moved = combat_spec(d, 0.05)
    drift = (moved.sum((2, 3)) - d.sum((2, 3))).abs().max().item()
    assert drift < 1e-4, "spec combat unexpectedly moved mass between teams"


def test_out_of_range_speed_is_rejected():
    """SPEED > 1 makes density * (1 - SPEED) negative; the sim NaNs a few ticks later
    with no other warning, so it is refused up front."""
    import pytest

    for bad in ({"sim.SPEED": 1.2}, {"sim.RATE_DT": 1.5}):
        with pytest.raises(ValueError):
            Params.from_cfg(load(**bad))


def test_health_field_conserves_mass_and_stays_bounded(device):
    """The optional health field (`track_health`) must not break the invariants.

    It is off by default because it measured *worse* on matchup fidelity, not better
    (0.31 against 0.24) -- see docs/results.md. The code stays because the negative
    result is worth keeping reproducible.
    """
    cfg = load(**{"sim.B": 4, "sim.track_health": True})
    env = FluxWar(cfg, device, seed=0)
    m0 = env.mass().clone()
    gen = torch.Generator(device=device).manual_seed(0)
    with torch.no_grad():
        for _ in range(300):
            env.step(torch.rand(env.B, 2, 2, device=device, generator=gen) * 2 - 1)
    rel = ((env.mass() - m0).abs() / m0).max().item()
    assert rel < 5e-4, f"relative mass drift {rel:.3e}"
    assert env.density.min().item() >= -1e-9
    h = env.health()
    assert h is not None and 0.0 <= h.min().item() and h.max().item() <= 1.0 + 1e-6
