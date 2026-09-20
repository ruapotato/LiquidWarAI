"""M3: optimise the policy by backpropagating through the simulator.

Same network as M2. The RL objective is replaced by direct minimisation of

    loss = -sum_t gamma^t (s_t - s_{t-1})

over a truncated window, with the simulator state detached at window boundaries.

`alpha` interpolates between the analytical gradient and a REINFORCE estimator of the
same objective. Suh et al. (ICML 2022) found analytical simulator gradients can be
worse than zeroth-order ones in contact-rich systems; combat here is a contact
process, so the interpolation is the escape hatch that paper's results argue for.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn

from ..config import load
from ..eval.arena import head_to_head
from ..sim.env import FluxWar
from .agents import SCRIPTED, NetAgent
from .net import CursorPolicy, n_params
from .rollout import OpponentPool


def train(cfg, device="cuda", out="runs/bptt", log_every=10, eval_every=50, max_seconds=None):
    torch.manual_seed(0)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    hp = cfg.bptt
    # The differentiable field is required: hard min() gives a subgradient that is zero
    # almost everywhere w.r.t. the cursor, so the analytical path would be silently dead.
    if not cfg.sim.soft_field:
        cfg.sim.soft_field = True
    net = CursorPolicy(cfg).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=float(hp.lr))
    # No torch.compile here: it does not compose with gradient checkpointing -- the
    # compiled region still keeps its intermediates, so the window OOMs at the batch
    # sizes that fit without it. Checkpointing is the one that matters for M3.
    env = FluxWar(cfg, device, seed=0, B=int(hp.num_envs),
                  checkpoint=bool(hp.get("checkpoint", True)))
    pool = OpponentPool(cfg, net, size=int(hp.pool_size))
    rng = random.Random(0)
    opponent = pool.sample(rng)
    baseline = SCRIPTED["chase_enemy"]

    Hb, B = int(hp.H_BPTT), int(hp.num_envs)
    gamma, alpha = float(hp.gamma), float(hp.alpha)
    iters = int(hp.total_env_steps) // (Hb * B)
    history, t_start, env_steps, ep_t = [], time.perf_counter(), 0, 0

    for it in range(iters):
        env.detach_()
        loss = 0.0
        reinforce = 0.0
        disc = 1.0
        for t in range(Hb):
            obs = env.obs(0)
            dist, _ = net.dist(obs)
            pre = dist.rsample()                       # reparameterised: gradient flows
            logp = dist.log_prob(pre).sum(-1)
            a0 = torch.tanh(pre)
            with torch.no_grad():
                a1 = opponent.act(env, 1)
            _, r, _ = env.step(torch.stack([a0, a1], 1))
            loss = loss - disc * r.mean()
            reinforce = reinforce - (logp * (disc * r).detach()).mean()
            disc *= gamma
            ep_t += 1
        env_steps += Hb * B

        opt.zero_grad(set_to_none=True)
        (alpha * loss + (1.0 - alpha) * reinforce).backward()
        gn = nn.utils.clip_grad_norm_(net.parameters(), float(hp.max_grad_norm))
        opt.step()

        if ep_t >= int(cfg.sim.episode_len):
            env.reset()
            opponent = pool.sample(rng)
            ep_t = 0
        if it % int(hp.pool_every) == 0 and it > 0:
            pool.snapshot(net)

        wall = time.perf_counter() - t_start
        if it % log_every == 0:
            rec = dict(iter=it, env_steps=env_steps, wall=wall, loss=loss.item(),
                       grad_norm=gn.item())
            history.append(rec)
            print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v)
                              for k, v in rec.items()}), flush=True)
        if it % eval_every == 0 and it > 0:
            r = head_to_head(NetAgent(net.eval()), baseline, cfg, games=64,
                             episode_len=int(cfg.eval.episode_len), device=device)
            net.train()
            rec = dict(iter=it, env_steps=env_steps, wall=wall, eval_win_rate=r["win_rate"],
                       eval_share=r["mean_share"])
            history.append(rec)
            print(json.dumps(rec), flush=True)
            torch.save(net.state_dict(), out / "policy.pt")
            (out / "history.json").write_text(json.dumps(history))
        if max_seconds and wall > max_seconds:
            print(f"# stopping at wall clock {wall:.0f}s", flush=True)
            break

    torch.save(net.state_dict(), out / "policy.pt")
    (out / "history.json").write_text(json.dumps(history))
    return net, history


def gradient_norm_report(cfg, device="cuda", windows=40):
    """Distribution of per-window gradient norms.

    Explosion is the failure the project brief expects. The one actually seen here is
    the opposite: a heavy tail *and* a substantial fraction of windows where the
    gradient is exactly zero. The objective is the change in population share, which
    is identically zero while the two armies are out of contact, so a truncated
    window that contains no fighting carries no signal at all -- not a small one, none.
    PPO does not care, because its value function bootstraps across the gap; BPTT gets
    nothing from those windows. In a real run 8 of 89 windows came back at exactly
    zero.
    """
    cfg = load() if cfg is None else cfg
    cfg.sim.soft_field = True
    net = CursorPolicy(cfg).to(device)
    env = FluxWar(cfg, device, seed=0, B=int(cfg.bptt.num_envs), checkpoint=True)
    opp = SCRIPTED["hold_own"]
    norms = []
    for _ in range(windows):
        env.detach_()
        loss, disc = 0.0, 1.0
        for _ in range(int(cfg.bptt.H_BPTT)):
            dist, _ = net.dist(env.obs(0))
            a0 = torch.tanh(dist.rsample())
            with torch.no_grad():
                a1 = opp.act(env, 1)
            _, r, _ = env.step(torch.stack([a0, a1], 1))
            loss = loss - disc * r.mean()
            disc *= float(cfg.bptt.gamma)
        net.zero_grad(set_to_none=True)
        loss.backward()
        norms.append(float(torch.nn.utils.clip_grad_norm_(net.parameters(), 1e9)))
    n = torch.tensor(norms)
    return dict(median=float(n.median()), p90=float(n.quantile(0.9)),
                p99=float(n.quantile(0.99)), max=float(n.max()),
                zero_fraction=float((n == 0).float().mean()),
                finite=float(torch.isfinite(n).float().mean()),
                dynamic_range=float(n.max() / n[n > 0].min()) if (n > 0).any() else 0.0,
                norms=norms)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="default.yaml")
    ap.add_argument("--out", default="runs/bptt")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--grad-report", action="store_true")
    a = ap.parse_args()
    over = {}
    if a.steps:
        over["bptt.total_env_steps"] = a.steps
    if a.alpha is not None:
        over["bptt.alpha"] = a.alpha
    cfg = load(a.config, **over)
    if a.grad_report:
        print(json.dumps({k: v for k, v in gradient_norm_report(cfg).items() if k != "norms"}))
    else:
        net, _ = train(cfg, out=a.out, max_seconds=a.max_seconds)
        print("params", n_params(net))
