"""A batched Liquid War 6 environment: many real games across worker processes.

The point of this is to remove the sim-to-real gap from training entirely. LW6's
kernel cannot be batched on a GPU -- the fighter update is sequential and
order-dependent -- but one game runs at ~1900 rounds/s per process, so a dozen
processes reach ~20k env-steps/s, which is the same order as the differentiable GPU
sim. The policy still runs on the GPU in the parent; only the physics are on CPU.

Observations move through shared memory rather than pickles: at 384 envs a step is
~19 MB of float32, which is fine to memcpy and expensive to serialise.
"""
from __future__ import annotations

import multiprocessing as mp
import os

import numpy as np
import torch

from ..sim.observation import N_CHANNELS
from .env import LW6Env

_CMD_STEP, _CMD_RESET, _CMD_CLOSE = 0, 1, 2


def _worker(idx, n_local, cfg_dict, obs_buf, act_buf, score_buf, cur_buf, done_buf,
            pipe, seed0, episode_len, opponent):
    """Run `n_local` real games, stepping them in lockstep with the parent."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from ..config import _wrap
    from ..sim.maps import make_maps
    from .kernel import reset_process_state, silence

    reset_process_state()   # do not reuse the parent's context across the fork
    silence()

    cfg = _wrap(cfg_dict)
    H, W = int(cfg.sim.H), int(cfg.sim.W)
    obs = np.frombuffer(obs_buf, np.float32).reshape(-1, N_CHANNELS, H, W)
    act = np.frombuffer(act_buf, np.float32).reshape(-1, 2, 2)
    sc = np.frombuffer(score_buf, np.float32).reshape(-1)
    cur = np.frombuffer(cur_buf, np.float32).reshape(-1, 2, 2)
    done = np.frombuffer(done_buf, np.float32).reshape(-1)
    lo = idx * n_local

    envs, steps, opps = [], [], []

    def attach_opponent(env):
        """Team 1 is driven natively, inside the worker.

        LW6's own bots keep C state per game and return an absolute cursor position.
        Routing that through the parent as a velocity both costs a round trip and
        cripples them -- mod-brute repositions anywhere on the map, and capping it to
        one cell per round turns the strongest opponent available into a weak one.
        """
        if not opponent:
            return None
        from ..eval.deployed import NativeBot
        bot = NativeBot(opponent)
        bot.attach(env.game, 1)
        env.external_teams.add(1)   # step() must not overwrite what the bot chooses
        return bot

    def build(i):
        gen = torch.Generator(device="cpu").manual_seed(seed0 + (lo + i) * 7919
                                                        + len(steps) * 104729)
        walls = make_maps(1, cfg, "cpu", gen)[0].numpy()
        return LW6Env(width=W, height=H, device="cpu", walls=walls,
                      max_cursor_speed=float(cfg.sim.MAX_CURSOR_SPEED),
                      action_mode=str(cfg.sim.get("action_mode", "velocity")))

    for i in range(n_local):
        steps.append(0)
        envs.append(build(i))
        opps.append(attach_opponent(envs[i]))
        envs[i].obs_into(obs[lo + i], 0)
        sc[lo + i] = float(envs[i].score())
        cur[lo + i] = envs[i]._cursor
        done[lo + i] = 0.0
    pipe.send(True)

    while True:
        cmd = pipe.recv()
        if cmd == _CMD_CLOSE:
            for e in envs:
                e.close()
            pipe.send(True)
            return
        for i, e in enumerate(envs):
            done[lo + i] = 0.0
            if cmd == _CMD_STEP:
                if opps[i] is not None:
                    opps[i].move()
                e.step(torch.from_numpy(act[lo + i]).unsqueeze(0))
                steps[i] += 1
                if steps[i] >= episode_len:
                    e.close()
                    envs[i] = e = build(i)
                    opps[i] = attach_opponent(e)
                    steps[i] = 0
                    done[lo + i] = 1.0   # the parent must not treat the score reset
                                         # back to 0.5 as a reward
            e.obs_into(obs[lo + i], 0)
            sc[lo + i] = float(e.score())
            cur[lo + i] = e._cursor
        pipe.send(True)


class LW6VecEnv:
    """`num_envs` real LW6 games, stepped together.

    `obs()` returns team 0's view; team 1's is the same tensor with the first two
    channels swapped, so the opponent needs no extra work.
    """

    def __init__(self, cfg, num_envs=256, workers=None, seed=0, device="cuda",
                 episode_len=None, opponent=None):
        self.cfg = cfg
        self.H, self.W = int(cfg.sim.H), int(cfg.sim.W)
        self.device = torch.device(device)
        workers = workers or min(mp.cpu_count(), 16)
        self.n_local = max(1, num_envs // workers)
        self.workers = workers
        self.B = self.n_local * workers
        self.episode_len = int(episode_len or cfg.sim.episode_len)
        self.opponent = opponent

        ctx = mp.get_context("fork")
        self._obs_buf = mp.RawArray("f", self.B * N_CHANNELS * self.H * self.W)
        self._act_buf = mp.RawArray("f", self.B * 4)
        self._score_buf = mp.RawArray("f", self.B)
        self._cur_buf = mp.RawArray("f", self.B * 4)
        self._done_buf = mp.RawArray("f", self.B)
        self.obs_np = np.frombuffer(self._obs_buf, np.float32).reshape(
            self.B, N_CHANNELS, self.H, self.W)
        self.act_np = np.frombuffer(self._act_buf, np.float32).reshape(self.B, 2, 2)
        self.score_np = np.frombuffer(self._score_buf, np.float32).reshape(self.B)
        self.cursor_np = np.frombuffer(self._cur_buf, np.float32).reshape(self.B, 2, 2)
        self.done_np = np.frombuffer(self._done_buf, np.float32).reshape(self.B)

        cfg_dict = _to_plain(cfg)
        self._pipes, self._procs = [], []
        for k in range(workers):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker,
                            args=(k, self.n_local, cfg_dict, self._obs_buf,
                                  self._act_buf, self._score_buf, self._cur_buf,
                                  self._done_buf, child, seed + k * 1009,
                                  self.episode_len, opponent),
                            daemon=True)
            p.start()
            child.close()
            self._pipes.append(parent)
            self._procs.append(p)
        for p in self._pipes:
            p.recv()
        self._prev_score = self.score_np.copy()

    # ------------------------------------------------------------------ views
    def obs(self, team: int = 0) -> torch.Tensor:
        o = torch.from_numpy(self.obs_np).to(self.device)
        if team == 1:
            o = torch.cat([o[:, 1:2], o[:, 0:1], o[:, 2:3]], dim=1)
        return o

    def score(self) -> torch.Tensor:
        return torch.from_numpy(self.score_np.copy()).to(self.device)

    @property
    def density(self):
        """Approximate: the observation is mass-normalised, which is all the scripted
        agents need (they only take centroids)."""
        o = torch.from_numpy(self.obs_np).to(self.device)
        return torch.stack([o[:, 0], o[:, 1]], dim=1)

    @property
    def cursor(self):
        return torch.from_numpy(self.cursor_np.copy()).to(self.device)

    @property
    def dtype(self):
        return torch.float32

    # ------------------------------------------------------------------- tick
    def step(self, action: torch.Tensor):
        """action: [B, 2, 2] cursor velocities. Returns (obs, reward, score) with
        reward the per-round change in team 0's share."""
        self.act_np[:] = action.detach().to("cpu", torch.float32).numpy()
        for p in self._pipes:
            p.send(_CMD_STEP)
        for p in self._pipes:
            p.recv()
        s = self.score_np.copy()
        # An env that just auto-reset jumps back to 0.5; that is not a reward.
        r = (s - self._prev_score) * (1.0 - self.done_np)
        self._prev_score = s
        return self.obs(), torch.from_numpy(r).to(self.device), \
            torch.from_numpy(s).to(self.device)

    def close(self):
        for p in self._pipes:
            try:
                p.send(_CMD_CLOSE)
            except (BrokenPipeError, OSError):
                pass
        for p in self._procs:
            p.join(timeout=5)


def _to_plain(cfg):
    if isinstance(cfg, dict):
        return {k: _to_plain(v) for k, v in cfg.items()}
    if isinstance(cfg, list):
        return [_to_plain(v) for v in cfg]
    return cfg
