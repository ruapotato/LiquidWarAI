"""The two cursor action spaces.

`velocity` is the human constraint: a direction, bounded by MAX_CURSOR_SPEED.
`target` is the bot constraint: an absolute position, which is what LW6's own
`mod-brute` takes (`cursor.pos.x = lw6sys_random(shape.w)` every round). The second
is strictly larger, and capping brute down to the first turned a 0.04 rout into a
coin flip -- so the difference is worth testing rather than assuming.
"""
import numpy as np
import pytest
import torch

from fluxwar.config import load
from fluxwar.policy.net import CursorPolicy
from fluxwar.sim.env import FluxWar


@pytest.mark.parametrize("mode", ["velocity", "target"])
def test_policy_round_trips_its_own_actions(mode):
    cfg = load()
    net = CursorPolicy(cfg, action_mode=mode)
    obs = torch.randn(4, 3, cfg.sim.H, cfg.sim.W)
    action, raw, logp, value = net.sample(obs)
    logp2, ent, value2 = net.evaluate_actions(obs, raw)
    assert action.shape == (4, 2)
    assert torch.allclose(logp, logp2, atol=1e-5)
    assert torch.allclose(value, value2, atol=1e-5)
    assert ent.shape == (4,) and float(ent.mean()) > 0
    lo, hi = (-1.0, 1.0) if mode == "velocity" else (0.0, 1.0)
    assert action.min() >= lo - 1e-6 and action.max() <= hi + 1e-6


@pytest.mark.parametrize("channels", [[32, 64], [32, 64, 64], [16, 32, 64, 64]])
def test_target_head_follows_the_trunk(channels):
    """The spatial head taps the second-to-last feature map, and both its index and
    its channel count have to come from the trunk. Hard-coding either one mismatches
    as soon as `policy.channels` changes."""
    cfg = load()
    cfg.policy.channels = channels
    net = CursorPolicy(cfg, action_mode="target")
    obs = torch.randn(2, 3, cfg.sim.H, cfg.sim.W)
    action, raw, logp, _ = net.sample(obs)
    assert action.shape == (2, 2)
    grid = net.spatial(net.trunk[: net._spatial_depth](obs)).shape[-2:]
    assert int(raw.max()) < grid[0] * grid[1]


def test_target_mode_places_the_cursor_anywhere(device):
    """One step must be able to cross the map; a velocity action cannot."""
    cfg = load(**{"sim.B": 2, "sim.action_mode": "target"})
    env = FluxWar(cfg, device, seed=0)
    before = env.cursor.clone()
    corner = torch.zeros(2, 2, 2, device=device)      # normalised (0, 0)
    env.step(corner)
    assert float(env.cursor.abs().max()) < 1.0, "target action did not move the cursor"
    assert float((env.cursor - before).abs().max()) > cfg.sim.MAX_CURSOR_SPEED


def test_velocity_mode_respects_the_speed_limit(device):
    cfg = load(**{"sim.B": 2, "sim.action_mode": "velocity"})
    env = FluxWar(cfg, device, seed=0)
    before = env.cursor.clone()
    env.step(torch.ones(2, 2, 2, device=device))
    moved = (env.cursor - before).abs().max().item()
    assert moved <= float(cfg.sim.MAX_CURSOR_SPEED) + 1e-5, f"cursor jumped {moved}"


@pytest.mark.parametrize("mode", ["velocity", "target"])
def test_exported_policy_matches_pytorch(tmp_path, mode):
    """The deployed C forward pass must agree with the one that was trained."""
    import ctypes

    lw6 = pytest.importorskip("fluxwar.lw6.kernel")
    if not lw6.library_path().exists():
        pytest.skip("liblw6ker.so not built")
    from fluxwar.deploy.export import export

    cfg = load()
    torch.manual_seed(0)
    net = CursorPolicy(cfg, action_mode=mode).eval()
    for p in net.parameters():
        p.data.normal_(0, 0.3)
    path = tmp_path / "p.flxw"
    export(net, path, int(cfg.sim.H), int(cfg.sim.W), 4.0, mode)

    lib = ctypes.CDLL(str(lw6.library_path()))
    lib.fw_net_load.argtypes = [ctypes.c_char_p]
    lib.fw_net_load.restype = ctypes.c_void_p
    lib.fw_net_forward.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
                                   ctypes.c_int, ctypes.c_int,
                                   ctypes.POINTER(ctypes.c_float)]
    lib.fw_net_forward.restype = ctypes.c_int
    lib.fw_net_action_mode.argtypes = [ctypes.c_void_p]
    lib.fw_net_action_mode.restype = ctypes.c_int
    handle = lib.fw_net_load(str(path).encode())
    assert handle, "C side could not load the exported weights"
    assert lib.fw_net_action_mode(handle) == (1 if mode == "target" else 0)

    rng = np.random.default_rng(0)
    for _ in range(3):
        obs = (rng.random((3, cfg.sim.H, cfg.sim.W)) * 2).astype(np.float32)
        out = (ctypes.c_float * 2)()
        assert lib.fw_net_forward(handle, obs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                  cfg.sim.H, cfg.sim.W, out)
        with torch.no_grad():
            want = net.act_deterministic(torch.from_numpy(obs).unsqueeze(0)).numpy()[0]
        assert np.abs(np.array([out[0], out[1]]) - want).max() < 1e-5


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["velocity", "target"])
def test_native_opponent_is_not_overwritten_by_the_action(mode):
    """A native LW6 bot must actually get to move.

    `LW6Env.step` writes cursors for both teams out of the action array. When team 1
    is driven by a C bot writing straight into the kernel, that overwrites whatever
    the bot just chose -- and the failure is silent and flattering: a policy
    "trained against mod-brute" hit a 1.0 win rate in 156k steps because brute's
    cursor was being parked every round. An idle learner should be destroyed.
    """
    lw6 = pytest.importorskip("fluxwar.lw6.kernel")
    if not lw6.library_path().exists():
        pytest.skip("liblw6ker.so not built")
    from fluxwar.lw6.vecenv import LW6VecEnv

    cfg = load(**{"sim.H": 48, "sim.W": 48, "sim.action_mode": mode})
    env = LW6VecEnv(cfg, num_envs=4, workers=2, seed=1, episode_len=1000,
                    opponent="brute", device="cpu")
    try:
        idle = torch.zeros(env.B, 2, 2)
        if mode == "target":
            idle[:, 0, :] = 0.5      # park the learner mid-map, not in a corner
        for _ in range(400):
            env.step(idle)
        share = float(env.score().mean())
    finally:
        env.close()
    assert share < 0.25, (
        f"an idle learner kept {share:.3f} of the population against mod-brute; "
        "the native opponent is probably not being driven")
