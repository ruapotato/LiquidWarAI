"""Throughput benchmark for the batched density sim (M1 acceptance)."""
from __future__ import annotations

import argparse
import time

import torch

from ..config import load
from ..sim.env import FluxWar


def bench(cfg, device="cuda", ticks=300, warmup=40, compile_tick=True):
    env = FluxWar(cfg, device, seed=0, compile_tick=compile_tick)
    act = torch.zeros(env.B, 2, 2, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            env.step(act)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(ticks):
            env.step(act)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    return dict(ticks_per_sec=ticks / dt, env_steps_per_sec=ticks * env.B / dt,
                B=env.B, H=cfg.sim.H, W=cfg.sim.W, seconds=dt,
                peak_mem_gb=torch.cuda.max_memory_allocated() / 2**30)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="default.yaml")
    ap.add_argument("--B", type=int, default=None)
    ap.add_argument("--ticks", type=int, default=300)
    ap.add_argument("--no-compile", action="store_true")
    a = ap.parse_args()
    cfg = load(a.config, **({"sim.B": a.B} if a.B else {}))
    r = bench(cfg, ticks=a.ticks, compile_tick=not a.no_compile)
    print(f"B={r['B']} {r['H']}x{r['W']}  {r['ticks_per_sec']:.1f} ticks/s  "
          f"{r['env_steps_per_sec']:.3e} env-steps/s  peak {r['peak_mem_gb']:.2f} GiB")
