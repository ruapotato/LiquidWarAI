"""Liquid War 6's own bots, as opponents.

`mod-follow` and `mod-brute` are the AI the game ships with. They are compiled into
liblw6ker.so alongside the kernel (see lw6/build.py) and called here through their
internal entry points rather than through LW6's dynamic module loader, which would
drag in libtool and the pilot library for no benefit.

Beating these is a more meaningful claim than beating a baseline written by the same
person who wrote the environment.
"""
from __future__ import annotations

import ctypes

import numpy as np
import torch

from ..policy.agents import Agent
from .kernel import context, load


class BotParam(ctypes.Structure):
    """lw6bot_param_t."""
    _fields_ = [("speed", ctypes.c_float), ("iq", ctypes.c_int),
                ("cursor_id", ctypes.c_uint16)]


class BotData(ctypes.Structure):
    """lw6bot_data_t: the game state plus the constant parameters."""
    _fields_ = [("game_state", ctypes.c_void_p), ("param", BotParam)]


_DECLARED = False


def _declare(lib):
    global _DECLARED
    if _DECLARED:
        return
    P = ctypes.c_void_p
    for mod in ("follow", "brute", "nn"):
        init = getattr(lib, f"_mod_{mod}_init")
        init.argtypes = [P, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p),
                         ctypes.POINTER(BotData)]
        init.restype = P
        nxt = getattr(lib, f"_mod_{mod}_next_move")
        nxt.argtypes = [P, P, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                        ctypes.POINTER(BotData)]
        nxt.restype = ctypes.c_int
        quit_ = getattr(lib, f"_mod_{mod}_quit")
        quit_.argtypes = [P, P]
        quit_.restype = None
    _DECLARED = True


class LW6Bot(Agent):
    """One of LW6's shipped bots, playing through the same Agent interface.

    Only usable against `lw6.env.LW6Env`: these bots read the kernel's game state
    directly, so there is nothing to point them at in the density sim.
    """

    def __init__(self, kind="follow", iq=100, speed=1.0, name=None):
        if kind not in ("follow", "brute", "nn"):
            raise ValueError(f"unknown LW6 bot {kind!r}")
        self.kind = kind
        self.iq = int(iq)
        self.speed = float(speed)
        self.name = name or f"lw6_{kind}"
        self.lib = load()
        _declare(self.lib)
        self.ctx = context(self.lib)
        self._ctxs = {}

    def _bot_for(self, env, team):
        key = (id(env), team)
        if key not in self._ctxs:
            data = BotData()
            data.game_state = env.game.game_state
            data.param = BotParam(self.speed, self.iq, env.game.cursor_ids[team])
            handle = getattr(self.lib, f"_mod_{self.kind}_init")(
                self.ctx, 0, None, ctypes.byref(data))
            if not handle:
                raise RuntimeError(f"_mod_{self.kind}_init failed")
            self._ctxs[key] = (handle, data)
        return self._ctxs[key]

    def act(self, env, team: int) -> torch.Tensor:
        """The bots return an absolute cursor target; this project's action is a
        velocity, so convert and clamp to one cell per round."""
        if not hasattr(env, "game"):
            raise TypeError("LW6 bots can only play the real kernel (lw6.env.LW6Env)")
        handle, data = self._bot_for(env, team)
        data.game_state = env.game.game_state      # the pointer may change between games
        x, y = ctypes.c_int(0), ctypes.c_int(0)
        ok = getattr(self.lib, f"_mod_{self.kind}_next_move")(
            self.ctx, handle, ctypes.byref(x), ctypes.byref(y), ctypes.byref(data))
        cur = env._cursor[team]
        if not ok:
            return torch.zeros(1, 2)
        # LW6 is (x, y); this project is (y, x) throughout. The bots return an
        # absolute target, so convert to the velocity this project's envs take, capped
        # at one cell per round.
        target = np.array([y.value, x.value], np.float32)
        delta = target - cur
        n = float(np.linalg.norm(delta))
        if n < 1e-6:
            return torch.zeros(1, 2)
        return torch.from_numpy((delta / max(n, 1.0)).astype(np.float32)).unsqueeze(0)

    def reset(self, env):
        self._ctxs.clear()


def lw6_bots(iq=100):
    return {f"lw6_{k}": LW6Bot(k, iq=iq) for k in ("follow", "brute")}


class ModNNBot(LW6Bot):
    """The deployed policy, running through LW6's own bot interface.

    Same code path a real Liquid War 6 build would use: mod-nn reads the kernel state,
    builds the observation in C, evaluates the exported weights in C, and returns a
    cursor position. Playing this against `NetAgent` on the same weights is the
    end-to-end check that deployment matches training.
    """

    def __init__(self, weights, name="mod_nn"):
        import os
        os.environ["FLUXWAR_POLICY"] = str(weights)
        super().__init__("nn", name=name)
