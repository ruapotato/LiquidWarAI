"""M4: PPO vs BPTT -- win rate against a fixed baseline, against environment steps and
against wall-clock seconds, plus a side-swapped head-to-head between the two policies.

Wall clock is the axis that matters here and the two methods are not close on it: the
differentiable rollout has to keep (or recompute) a window of activations, so M3 runs
at a few hundred env-steps/s where PPO on the same simulator runs at tens of
thousands. A per-step advantage for the analytical gradient has to be large to pay
for that.

`--real` additionally benchmarks both inside real Liquid War 6, which is the number
that decides whether either policy is worth deploying.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..config import load
from ..policy.agents import SCRIPTED, NetAgent
from ..policy.net import CursorPolicy
from .arena import head_to_head, round_robin


def load_policy(cfg, path, device="cuda", name="net"):
    net = CursorPolicy(cfg).to(device)
    net.load_state_dict(torch.load(path, map_location=device))
    net.eval()
    return NetAgent(net, name=name)


def curves(history_path):
    h = json.loads(Path(history_path).read_text())
    return [(r["env_steps"], r["wall"], r["eval_win_rate"], r["eval_share"])
            for r in h if "eval_win_rate" in r]


def plot(ppo_hist, bptt_hist, out="runs/compare.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for name, hist, colour in (("PPO (M2)", ppo_hist, "#1f77b4"), ("BPTT (M3)", bptt_hist, "#d62728")):
        c = curves(hist)
        if not c:
            continue
        steps, wall, wr, _ = zip(*c)
        ax[0].plot(steps, wr, "-o", ms=3, color=colour, label=name)
        ax[1].plot(wall, wr, "-o", ms=3, color=colour, label=name)
    ax[0].set_xlabel("environment steps")
    ax[1].set_xlabel("wall clock (s)")
    for a in ax:
        a.set_ylabel("win rate vs chase_enemy")
        a.axhline(0.8, ls="--", lw=0.8, color="gray")
        a.grid(alpha=0.25)
        a.legend()
    fig.savefig(out, dpi=130)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="default.yaml")
    ap.add_argument("--ppo", default="runs/ppo")
    ap.add_argument("--bptt", default="runs/bptt")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--real", action="store_true",
                    help="also benchmark both policies inside real Liquid War 6")
    a = ap.parse_args()
    cfg = load(a.config)
    dev = "cuda"
    ppo = load_policy(cfg, Path(a.ppo) / "policy.pt", dev, "ppo")
    bptt = load_policy(cfg, Path(a.bptt) / "policy.pt", dev, "bptt")

    report = {}
    for name, agent in (("ppo", ppo), ("bptt", bptt)):
        for opp in ("chase_enemy", "hold_own", "stationary"):
            report[f"{name}_vs_{opp}"] = head_to_head(
                agent, SCRIPTED[opp], cfg, games=a.games,
                episode_len=int(cfg.eval.episode_len), device=dev)
    report["ppo_vs_bptt"] = head_to_head(ppo, bptt, cfg, games=a.games,
                                         episode_len=int(cfg.eval.episode_len), device=dev)
    if a.real:
        from .transfer import head_to_head_reference
        for name, path in (("ppo", Path(a.ppo) / "policy.pt"),
                           ("bptt", Path(a.bptt) / "policy.pt")):
            for opp in ("chase_enemy", "hold_own"):
                report[f"real_lw6_{name}_vs_{opp}"] = head_to_head_reference(
                    str(path), opp, cfg, games=16, episode_len=400)
    elo, table = round_robin([ppo, bptt, SCRIPTED["chase_enemy"], SCRIPTED["hold_own"],
                              SCRIPTED["lean2"], SCRIPTED["lean5"]], cfg, games=64,
                             episode_len=int(cfg.eval.episode_len), device=dev)
    report["elo"] = elo
    report["pairwise"] = {f"{k[0]}|{k[1]}": v for k, v in table.items()}
    Path("runs/compare.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    try:
        print("plot:", plot(Path(a.ppo) / "history.json", Path(a.bptt) / "history.json"))
    except Exception as e:  # plotting is a convenience, not the result
        print("plot failed:", e)


if __name__ == "__main__":
    main()
