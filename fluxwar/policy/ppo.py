"""M2: PPO baseline. The control condition for M3 -- without it M3 is uninterpretable.

Standard PPO, no custom tricks. The learner always plays team 0; the opponent is drawn
from the frozen checkpoint pool and plays team 1. Reward is the per-tick change in team
0's share of the population, which is dense and exactly zero-sum.

Two environments, chosen with --env:

* `density` -- the differentiable GPU sim. Fast (~17k env-steps/s here) and the same
  environment M3 trains on, so the M4 comparison is like for like.
* `lw6` -- real Liquid War 6 games in worker processes, ~7k env-steps/s. Slower, but
  there is no sim-to-real gap at all: a policy trained here is trained on the game it
  will be deployed into. The policy still runs on the GPU; only the physics are on CPU.
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


def gae(rewards, values, last_value, gamma, lam):
    T, B = rewards.shape
    adv = torch.zeros_like(rewards)
    run = torch.zeros(B, device=rewards.device)
    nxt = last_value
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * nxt - values[t]
        run = delta + gamma * lam * run
        adv[t] = run
        nxt = values[t]
    return adv, adv + values


def make_env(cfg, kind, device, num_envs, seed=0, opponent=None):
    """`density` is the differentiable GPU sim; `lw6` is real games in subprocesses.

    `opponent` names one of LW6's own bots to drive team 1 natively inside each
    worker. That matters for mod-brute: it returns an absolute cursor position and
    routing it through the parent as a velocity would cap the strongest opponent
    available to one cell per round, which is most of its strength.
    """
    if kind == "lw6":
        from ..lw6.vecenv import LW6VecEnv
        env = LW6VecEnv(cfg, num_envs=num_envs, seed=seed, device=device,
                        opponent=opponent)
        return env, True      # workers reset their own games
    if opponent:
        raise ValueError("native LW6 opponents need --env lw6")
    return FluxWar(cfg, device, seed=seed, B=num_envs, compile_tick=True), False


@torch.no_grad()
def evaluate(net, baseline, cfg, env_kind, device, games=64, episode_len=400,
             opponent=None, _cache={}):
    """Win rate against a fixed scripted baseline, in the environment being trained on.

    Evaluating an LW6-trained policy in the density sim would measure the wrong thing,
    so the eval environment follows --env. The LW6 eval env is built once and reused;
    spinning up worker processes every fifty iterations costs more than the eval.
    """
    if env_kind != "lw6" and opponent is None:
        from ..eval.arena import head_to_head
        r = head_to_head(NetAgent(net.eval()), baseline, cfg, games=games,
                         episode_len=int(cfg.eval.episode_len), device=device)
        net.train()
        return r["win_rate"], r["mean_share"]

    from ..lw6.vecenv import LW6VecEnv
    if "env" not in _cache:
        # Fewer workers than training uses: this runs between iterations, and the
        # native opponent (mod-brute forward-simulates a sandbox each round) is the
        # expensive part.
        _cache["env"] = LW6VecEnv(cfg, num_envs=games, workers=6, seed=12345,
                                  device=device, episode_len=episode_len,
                                  opponent=opponent)
    env = _cache["env"]
    native = opponent is not None
    net.eval()
    # Stop one short of the episode length: the workers auto-reset on the last step
    # and the score snaps back to 0.5, which reads as "every game was a draw".
    for _ in range(episode_len - 1):
        a0 = net.act_deterministic(env.obs(0))
        a1 = torch.zeros_like(a0) if native else baseline.act(env, 1)
        env.step(torch.stack([a0, a1], 1))
    s = env.score()
    env.step(torch.zeros(env.B, 2, 2, device=env.device))   # roll the envs over
    net.train()
    eps = 1e-3
    return float((s > 0.5 + eps).float().mean()), float(s.mean())


def train(cfg, device="cuda", out="runs/ppo", log_every=10, eval_every=50,
          max_seconds=None, env_kind="density", native_opponent=None, init=None):
    torch.manual_seed(0)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    hp = cfg.ppo
    action_mode = str(cfg.sim.get("action_mode", "velocity"))
    net = CursorPolicy(cfg, action_mode=action_mode).to(device)
    if init:
        # Warm start, for a curriculum: learn against a beatable opponent first, then
        # against the one that matters. Against an opponent far stronger than the
        # policy the reward is dominated by the opponent's choices, and PPO never
        # gets a foothold -- the velocity agent sat flat for a million steps below
        # where a hand-written chaser already was.
        missing = net.load_state_dict(torch.load(init, map_location=device),
                                      strict=False)
        print(f"# warm start from {init} ({missing})", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=float(hp.lr))
    env, auto_reset = make_env(cfg, env_kind, device, int(hp.num_envs),
                               opponent=native_opponent)
    is_native = native_opponent is not None
    B = env.B                       # lw6 rounds num_envs down to a multiple of workers
    # With a native opponent the workers drive team 1 themselves, so the checkpoint
    # pool is unused; it exists for self-play only. Do not name it `opponent`: an
    # earlier version did, shadowing the bot name and handing a Python object to the
    # C loader as a module name.
    pool = OpponentPool(cfg, net, size=int(hp.pool_size))
    rng = random.Random(0)
    pool_opponent = pool.sample(rng)
    baseline = SCRIPTED["chase_enemy"]

    T = int(hp.rollout_len)
    steps_per_iter = T * B
    iters = int(hp.total_env_steps) // steps_per_iter
    history = []
    t_start = time.perf_counter()
    env_steps = 0
    ep_t = 0

    for it in range(iters):
        obs_buf = torch.empty(T, B, 3, cfg.sim.H, cfg.sim.W, device=device)
        raw_buf = None      # shape depends on the action mode; allocated on first use
        logp_buf = torch.empty(T, B, device=device)
        val_buf = torch.empty(T, B, device=device)
        rew_buf = torch.empty(T, B, device=device)

        with torch.no_grad():
            for t in range(T):
                obs = env.obs(0)
                a0, raw, logp, v = net.sample(obs)
                if raw_buf is None:
                    raw_buf = torch.empty((T,) + raw.shape, dtype=raw.dtype, device=device)
                if is_native:
                    # The opponent is driven inside the worker processes, natively.
                    a1 = torch.zeros_like(a0)
                else:
                    a1 = pool_opponent.act(env, 1)
                _, r, _ = env.step(torch.stack([a0, a1], 1))
                obs_buf[t], raw_buf[t], logp_buf[t], val_buf[t], rew_buf[t] = \
                    obs, raw, logp, v, r
                ep_t += 1
                if ep_t >= int(cfg.sim.episode_len):
                    if not auto_reset:
                        env.reset()
                    if not is_native:
                        pool_opponent = pool.sample(rng)
                    ep_t = 0
            last_v = net.evaluate_actions(env.obs(0), raw_buf[-1])[2]
        env_steps += steps_per_iter

        adv, ret = gae(rew_buf, val_buf, last_v, float(hp.gamma), float(hp.gae_lambda))
        adv = (adv - adv.mean()) / adv.std().clamp_min(1e-8)

        flat = lambda x: x.reshape(T * B, *x.shape[2:])
        obs_f, raw_f, logp_f, adv_f, ret_f = map(flat, (obs_buf, raw_buf, logp_buf, adv, ret))
        n = T * B
        mb = n // int(hp.minibatches)
        stats = {}
        for _ in range(int(hp.epochs)):
            perm = torch.randperm(n, device=device)
            for k in range(0, n, mb):
                idx = perm[k : k + mb]
                logp, ent, v = net.evaluate_actions(obs_f[idx], raw_f[idx])
                ratio = (logp - logp_f[idx]).exp()
                a = adv_f[idx]
                pg = -torch.min(ratio * a,
                                ratio.clamp(1 - float(hp.clip), 1 + float(hp.clip)) * a).mean()
                vf = 0.5 * (v - ret_f[idx]).pow(2).mean()
                ent = ent.mean()
                loss = pg + float(hp.vf_coef) * vf - float(hp.ent_coef) * ent
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gn = nn.utils.clip_grad_norm_(net.parameters(), float(hp.max_grad_norm))
                opt.step()
                stats = dict(pg=pg.item(), vf=vf.item(), ent=ent.item(), grad_norm=gn.item())

        if not is_native and it % int(hp.pool_every) == 0 and it > 0:
            pool.snapshot(net)

        wall = time.perf_counter() - t_start
        if it % log_every == 0:
            rec = dict(iter=it, env_steps=env_steps, wall=wall,
                       mean_reward=float(rew_buf.sum(0).mean()), **stats)
            history.append(rec)
            print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v)
                              for k, v in rec.items()}), flush=True)
        if it % eval_every == 0 and it > 0:
            wr, share = evaluate(net, baseline, cfg, env_kind, device,
                                 opponent=native_opponent)
            rec = dict(iter=it, env_steps=env_steps, wall=wall, eval_win_rate=wr,
                       eval_share=share)
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="default.yaml")
    ap.add_argument("--out", default="runs/ppo")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--env", default="density", choices=("density", "lw6"),
                    help="train on the differentiable sim or on real LW6 games")
    ap.add_argument("--num-envs", type=int, default=None)
    ap.add_argument("--opponent", default=None,
                    help="an LW6 bot (brute, follow) driven natively in the workers")
    ap.add_argument("--action-mode", default=None, choices=("velocity", "target"))
    ap.add_argument("--init", default=None,
                    help="warm start from a checkpoint, for a curriculum")
    a = ap.parse_args()
    over = {}
    if a.steps:
        over["ppo.total_env_steps"] = a.steps
    if a.num_envs:
        over["ppo.num_envs"] = a.num_envs
    if a.action_mode:
        over["sim.action_mode"] = a.action_mode
    cfg = load(a.config, **over)
    net, _ = train(cfg, out=a.out, max_seconds=a.max_seconds, env_kind=a.env,
                   native_opponent=a.opponent, init=a.init)
    print("params", n_params(net))
