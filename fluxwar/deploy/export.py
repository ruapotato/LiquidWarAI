"""Export a trained policy to a flat binary the LW6 bot plugin can read.

The plugin is C with no dependencies -- linking libtorch into a game bot module is
not reasonable -- so the weights go out in a trivial format and `csrc/mod_nn.c`
implements the forward pass directly. The network is small enough (~90k parameters,
three strided convolutions and two linear layers) that this is a couple of hundred
lines and runs in microseconds.

Format, all little-endian:

    magic   "FLXW"            4 bytes
    version uint32            3
    in_h    uint32            training input height
    in_w    uint32            training input width
    cursor_speed float32      cells per round the policy was trained with
    action_mode uint32        0 = velocity (bounded speed), 1 = target (place anywhere)
    n_tensors uint32
    then per tensor:
        name_len uint32, name bytes
        ndim uint32, dims uint32[ndim]
        float32 data, C order
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import torch

from ..config import load
from ..policy.net import CursorPolicy

MAGIC = b"FLXW"
VERSION = 3


def export(net: CursorPolicy, path, in_h=64, in_w=64, cursor_speed=1.0,
           action_mode='velocity'):
    tensors = [(k, v.detach().cpu().contiguous().float())
               for k, v in net.state_dict().items()]
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<IIIfII", VERSION, in_h, in_w, float(cursor_speed),
                            1 if action_mode == "target" else 0, len(tensors)))
        for name, t in tensors:
            raw = name.encode()
            f.write(struct.pack("<I", len(raw)))
            f.write(raw)
            f.write(struct.pack("<I", t.dim()))
            for d in t.shape:
                f.write(struct.pack("<I", d))
            f.write(t.numpy().tobytes())
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="default.yaml")
    ap.add_argument("--policy", required=True)
    # NB: input size and cursor speed are read from --config, not from the
    # checkpoint, so pass the config the policy was *trained* with. Exporting a
    # policy trained at 1 cell/round against a config that says 4 makes the deployed
    # bot move four times too far.
    ap.add_argument("--out", default="build/policy.flxw")
    ap.add_argument("--action-mode", default=None, choices=("velocity", "target"))
    ap.add_argument("--cursor-speed", type=float, default=None,
                    help="override MAX_CURSOR_SPEED, for a checkpoint trained under a "
                         "config you no longer have")
    a = ap.parse_args()
    cfg = load(a.config)
    mode = a.action_mode or str(cfg.sim.get("action_mode", "velocity"))
    net = CursorPolicy(cfg, action_mode=mode)
    net.load_state_dict(torch.load(a.policy, map_location="cpu"))
    net.eval()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    speed = a.cursor_speed if a.cursor_speed is not None else float(cfg.sim.MAX_CURSOR_SPEED)
    export(net, a.out, int(cfg.sim.H), int(cfg.sim.W), speed, mode)
    total = sum(p.numel() for p in net.parameters())
    print(f"wrote {a.out}  ({total} parameters, {Path(a.out).stat().st_size} bytes, "
          f"{int(cfg.sim.H)}x{int(cfg.sim.W)} input, {mode}, cursor speed {speed})")


if __name__ == "__main__":
    main()
