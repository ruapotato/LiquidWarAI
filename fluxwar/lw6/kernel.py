"""ctypes binding to the real Liquid War 6 game kernel.

This is not a port and not an approximation -- it is LW6's own `liblw6ker` compiled
from source and called directly, so a game played through here is the game. It is the
arbiter for everything the differentiable sim claims, and the natural target for a
deployed policy.

Batch size is 1 by design: LW6's fighter update is sequential and order-dependent, so
games cannot be batched without changing results. Run many across processes.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

_LIB = None
_CTX = None

# lw6sys log levels, from sys.h
LOG_ERROR, LOG_WARNING, LOG_NOTICE, LOG_INFO, LOG_DEBUG = 0, 1, 2, 3, 4


class XYZ(ctypes.Structure):
    """lw6sys_xyz_t: three bit-fields packed into one 32-bit word (14/14/4)."""
    _fields_ = [("x", ctypes.c_int32, 14), ("y", ctypes.c_int32, 14),
                ("z", ctypes.c_int32, 4)]


class Cursor(ctypes.Structure):
    """lw6ker_cursor_t."""
    _fields_ = [
        ("node_id", ctypes.c_uint64),
        ("cursor_id", ctypes.c_uint16),
        ("letter", ctypes.c_char),
        ("enabled", ctypes.c_int),
        ("team_color", ctypes.c_int),
        ("pos", XYZ),
        ("fire", ctypes.c_int),
        ("fire2", ctypes.c_int),
        ("apply_pos", XYZ),
        ("pot_offset", ctypes.c_int32),
    ]


class Fighter(ctypes.Structure):
    """lw6ker_fighter_t. Three 32-bit words: (team_color:8, last_direction:8,
    health:16), (act_counter:16, pad:16), then the packed xyz."""
    _fields_ = [
        ("team_color", ctypes.c_uint32, 8),
        ("last_direction", ctypes.c_int32, 8),
        ("health", ctypes.c_int32, 16),
        ("act_counter", ctypes.c_int32, 16),
        ("pad", ctypes.c_int32, 16),
        ("pos", XYZ),
    ]


# numpy view of the same 12 bytes, so the whole fighter array can be read in one go
# instead of probing every cell on the map. Bit-fields are unpacked with shifts.
FIGHTER_DTYPE = np.dtype([("w0", "<u4"), ("w1", "<u4"), ("w2", "<u4")])


def library_path() -> Path:
    env = os.environ.get("FLUXWAR_LW6_LIB")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "build" / "lw6" / "liblw6ker.so"


def load(path=None):
    """Load liblw6ker.so and declare the signatures actually used here."""
    global _LIB
    # The kernel spreads each team's gradient under `#pragma omp parallel for`. For a
    # single game that buys nothing, and left at the default it spawns a thread per
    # core inside every worker of a process pool. An intermittent segfault was seen
    # with it on; one thread is both faster here and stable.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if _LIB is not None:
        return _LIB
    path = Path(path) if path else library_path()
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build it with:\n"
            f"    python -m fluxwar.lw6.build --out build/lw6")
    lib = ctypes.CDLL(str(path))
    P = ctypes.c_void_p
    sig = {
        "lw6sys_context_new": ([], P),
        "lw6sys_context_free": ([P], None),
        "lw6sys_log_set_level": ([P, ctypes.c_int], None),
        "lw6sys_debug_set": ([P, ctypes.c_int], None),
        "lw6sys_log_set_file": ([P, ctypes.c_char_p], None),
        "lw6map_builtin_custom": ([P, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                   ctypes.c_int], P),
        "lw6map_free": ([P, P], None),
        "lw6ker_game_struct_new": ([P, P, P], P),
        "lw6ker_game_struct_free": ([P, P], None),
        "lw6ker_game_state_new": ([P, P, P], P),
        "lw6ker_game_state_free": ([P, P], None),
        "lw6ker_cursor_reset": ([P, ctypes.POINTER(Cursor)], None),
        "lw6ker_game_state_register_node": ([P, P, ctypes.c_uint64], ctypes.c_int),
        "lw6ker_game_state_get_w": ([P, P], ctypes.c_int),
        "lw6ker_game_state_get_h": ([P, P], ctypes.c_int),
        "lw6ker_game_state_add_cursor": ([P, P, ctypes.c_uint64, ctypes.c_uint16,
                                          ctypes.c_int], ctypes.c_int),
        "lw6ker_game_state_get_cursor": ([P, P, ctypes.POINTER(Cursor),
                                          ctypes.c_uint16], ctypes.c_int),
        "lw6ker_game_state_set_cursor": ([P, P, ctypes.POINTER(Cursor)], ctypes.c_int),
        "lw6ker_game_state_do_round": ([P, P], None),
        "lw6ker_game_state_get_rounds": ([P, P], ctypes.c_uint32),
        "lw6ker_game_state_get_team_info": ([P, P, ctypes.c_int,
                                             ctypes.POINTER(ctypes.c_int32),
                                             ctypes.POINTER(ctypes.c_int32)], ctypes.c_int),
        "lw6ker_game_state_get_nb_active_fighters": ([P, P], ctypes.c_int32),
        "lw6ker_game_state_get_fighter_id": ([P, P, ctypes.c_int32, ctypes.c_int32,
                                              ctypes.c_int32], ctypes.c_int32),
        "lw6ker_game_state_get_fighter_ro_by_id": ([P, P, ctypes.c_int32],
                                                   ctypes.POINTER(Fighter)),
        "lw6ker_game_struct_get_zone_id": ([P, P, ctypes.c_int32, ctypes.c_int32,
                                            ctypes.c_int32], ctypes.c_int32),
        "fluxwar_set_walls": ([P, P, ctypes.POINTER(ctypes.c_uint8), ctypes.c_int,
                               ctypes.c_int], ctypes.c_int),
        "fluxwar_read_fighters": ([P, P, ctypes.POINTER(ctypes.c_int32), ctypes.c_int],
                                  ctypes.c_int),
        "fluxwar_read_walls": ([P, P, ctypes.POINTER(ctypes.c_uint8), ctypes.c_int,
                                ctypes.c_int], None),
        "lw6ker_game_state_is_over": ([P, P], ctypes.c_int),
        "lw6ker_game_state_get_winner": ([P, P, ctypes.c_int], ctypes.c_int),
    }
    for name, (argtypes, restype) in sig.items():
        fn = getattr(lib, name)
        fn.argtypes = argtypes
        fn.restype = restype
    _LIB = lib
    return lib


def context(lib=None):
    """The process-wide lw6sys context.

    Deliberately a singleton: creating and freeing a context per game segfaults on the
    second one. lw6sys keeps process-global state (logging, memory bookkeeping), so
    one context per process is what the library expects.
    """
    global _CTX
    if _CTX is None:
        lib = lib or load()
        _CTX = lib.lw6sys_context_new()
        if not _CTX:
            raise RuntimeError("lw6sys_context_new failed")
        # The kernel logs at DEBUG by default, several lines per round. Left on it
        # dominates the runtime completely -- a 2400-round benchmark that should take
        # two seconds ran for ten minutes writing mutex traces to stderr.
        silence(lib, _CTX)
    return _CTX


def silence(lib=None, ctx=None):
    """Turn the kernel's logging off.

    It defaults to INFO and logs several lines per round; left on it dominates the
    runtime completely and buries stderr. Both switches matter: `lw6sys_log` prints
    when the level allows it *or* when the debug flag is set.
    """
    lib = lib or load()
    ctx = ctx or context(lib)
    lib.lw6sys_log_set_level(ctx, LOG_ERROR)
    lib.lw6sys_debug_set(ctx, 0)
    # Pin the log file. Left unset, the first message makes lw6sys fall back to
    # `lw6sys_get_default_log_file`, which outside a real installation returns
    # garbage: this dropped 18 MB of logs into the working directory under filenames
    # made of raw bytes.
    lib.lw6sys_log_set_file(ctx, os.devnull.encode())


def reset_process_state():
    """Forget the cached library handle and context.

    A forked worker inherits the parent's module globals, including a context pointer
    that belongs to the parent's address space. Calling this first makes the child
    build its own.
    """
    global _LIB, _CTX
    _LIB, _CTX = None, None


class LW6Game:
    """One real LW6 game, two teams, driven a round at a time."""

    NODE_ID = 0x1234123412341234

    def __init__(self, width=64, height=64, noise_percent=10, lib_path=None,
                 walls=None):
        self.lib = load(lib_path)
        self.ctx = context(self.lib)
        self.level = self.lib.lw6map_builtin_custom(self.ctx, width, height, 1,
                                                    noise_percent)
        if not self.level:
            raise RuntimeError("lw6map_builtin_custom failed")
        if walls is not None:
            # Must happen before the game_struct is built: the struct bakes the zone
            # decomposition out of the level body.
            w = np.ascontiguousarray((np.asarray(walls) > 0.5).astype(np.uint8))
            if w.shape != (height, width):
                raise ValueError(f"walls must be {(height, width)}, got {w.shape}")
            ok = self.lib.fluxwar_set_walls(
                self.ctx, self.level,
                w.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), width, height)
            if not ok:
                raise RuntimeError("fluxwar_set_walls failed")
        self.game_struct = self.lib.lw6ker_game_struct_new(self.ctx, self.level, None)
        self.game_state = self.lib.lw6ker_game_state_new(self.ctx, self.game_struct, None)
        if not self.game_state:
            raise RuntimeError("lw6ker_game_state_new failed")
        self.W = int(self.lib.lw6ker_game_state_get_w(self.ctx, self.game_state))
        self.H = int(self.lib.lw6ker_game_state_get_h(self.ctx, self.game_state))
        # A cursor belongs to a node, and the node has to exist first -- add_cursor
        # just returns 0 otherwise, with no other complaint.
        if not self.lib.lw6ker_game_state_register_node(self.ctx, self.game_state,
                                                        self.NODE_ID):
            raise RuntimeError("register_node failed")
        self._walls = None
        self._fbuf = None
        self.cursor_ids = (0x1234, 0x2345)
        for team, cid in enumerate(self.cursor_ids):
            if not self.lib.lw6ker_game_state_add_cursor(self.ctx, self.game_state,
                                                         self.NODE_ID, cid, team):
                raise RuntimeError(f"add_cursor failed for team {team}")
        # Teams spawn where the map says; place the cursors on their own armies so a
        # game that is never driven is at least well formed.
        for team in range(2):
            c = Cursor()
            self.lib.lw6ker_cursor_reset(self.ctx, ctypes.byref(c))
            c.node_id = self.NODE_ID
            c.cursor_id = self.cursor_ids[team]
            c.pos.x = self.W // 4 if team == 0 else 3 * self.W // 4
            c.pos.y = self.H // 2
            self.lib.lw6ker_game_state_set_cursor(self.ctx, self.game_state,
                                                  ctypes.byref(c))

    # ------------------------------------------------------------------ state
    def cursor(self, team: int) -> Cursor:
        c = Cursor()
        self.lib.lw6ker_game_state_get_cursor(self.ctx, self.game_state, ctypes.byref(c),
                                              self.cursor_ids[team])
        return c

    def set_cursor(self, team: int, x: int, y: int):
        c = self.cursor(team)
        c.node_id = self.NODE_ID
        c.cursor_id = self.cursor_ids[team]
        c.pos.x = int(np.clip(x, 0, self.W - 1))
        c.pos.y = int(np.clip(y, 0, self.H - 1))
        self.lib.lw6ker_game_state_set_cursor(self.ctx, self.game_state, ctypes.byref(c))

    def team_fighters(self, team: int) -> int:
        nb_cursors = ctypes.c_int32()
        nb_fighters = ctypes.c_int32()
        self.lib.lw6ker_game_state_get_team_info(self.ctx, self.game_state, team,
                                                 ctypes.byref(nb_cursors),
                                                 ctypes.byref(nb_fighters))
        return int(nb_fighters.value)

    def score(self) -> float:
        a, b = self.team_fighters(0), self.team_fighters(1)
        return a / max(a + b, 1)

    def fighter_arrays(self):
        """(team, health, y, x) for every active fighter, in one call into C.

        Probing `get_fighter_id` for each cell costs thousands of ctypes calls per
        round and drops throughput from ~5000 rounds/s to 67.
        """
        n = int(self.lib.lw6ker_game_state_get_nb_active_fighters(self.ctx,
                                                                  self.game_state))
        if n <= 0:
            z = np.zeros(0, np.int32)
            return z, z, z, z
        if self._fbuf is None or len(self._fbuf) < n * 4:
            self._fbuf = np.zeros(max(n, 1) * 4, np.int32)
        got = self.lib.fluxwar_read_fighters(
            self.ctx, self.game_state,
            self._fbuf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            len(self._fbuf) // 4)
        if got < 0:
            raise RuntimeError("fluxwar_read_fighters: buffer too small")
        rows = self._fbuf[: got * 4].reshape(got, 4)
        return rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]

    def density(self) -> np.ndarray:
        """[2, H, W] fighters per cell, for the two teams this wrapper plays."""
        team, _, y, x = self.fighter_arrays()
        img = np.zeros((2, self.H, self.W), np.float32)
        keep = team < 2
        np.add.at(img, (team[keep], y[keep], x[keep]), 1.0)
        return img

    def health_image(self) -> np.ndarray:
        """[H, W] health in [0, 1], for rendering fighters darkening before they flip."""
        _, health, y, x = self.fighter_arrays()
        img = np.zeros((self.H, self.W), np.float32)
        if health.size:
            img[y, x] = health / 10000.0
        return img

    def walls(self) -> np.ndarray:
        """[H, W], 1.0 where fighters may stand -- the game's own notion of walkable
        (a cell is open exactly when the kernel assigns it a zone)."""
        if self._walls is None:
            buf = np.zeros(self.H * self.W, np.uint8)
            self.lib.fluxwar_read_walls(
                self.ctx, self.game_struct,
                buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), self.W, self.H)
            self._walls = buf.reshape(self.H, self.W).astype(np.float32)
        return self._walls

    # ------------------------------------------------------------------- tick
    def step(self):
        self.lib.lw6ker_game_state_do_round(self.ctx, self.game_state)

    @property
    def rounds(self) -> int:
        return int(self.lib.lw6ker_game_state_get_rounds(self.ctx, self.game_state))

    def is_over(self) -> bool:
        return bool(self.lib.lw6ker_game_state_is_over(self.ctx, self.game_state))

    def close(self):
        if getattr(self, "game_state", None):
            self.lib.lw6ker_game_state_free(self.ctx, self.game_state)
            self.game_state = None
        if getattr(self, "game_struct", None):
            self.lib.lw6ker_game_struct_free(self.ctx, self.game_struct)
            self.game_struct = None
        if getattr(self, "level", None):
            self.lib.lw6map_free(self.ctx, self.level)
            self.level = None
        self.ctx = None   # shared; never freed per game

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
