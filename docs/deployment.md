# Getting a trained policy into Liquid War 6

This is built and tested, not just specified.

## What exists

* `fluxwar/deploy/export.py` writes the policy to a flat binary (`.flxw`).
* `fluxwar/lw6/csrc/fluxwar_nn.c` evaluates it in plain C — three strided 3x3
  convolutions, an adaptive average pool, two linear layers, ~90k parameters, no
  dependencies. Linking libtorch into a game plugin is not reasonable. It is checked
  against PyTorch on random inputs and agrees to **1.2e-7**.
* `fluxwar/lw6/csrc/mod_nn.c` is a Liquid War 6 bot backend implementing LW6's own
  interface (`init` / `next_move` / `quit` / `repr` plus the module pedigree), so the
  game can load it the way it loads `mod-brute` or `mod-follow`. It builds the
  observation from the kernel's state, downsamples to the training resolution, and
  returns a cursor position.
* `fluxwar/eval/deployed.py` plays real LW6 games with bots driving cursors *the way
  the game does* — absolute positions written straight into the kernel — which is the
  end-to-end check.

```bash
python -m fluxwar.deploy.export --policy runs/ppo/policy.pt --out build/policy.flxw
python -m fluxwar.eval.deployed --weights build/policy.flxw --against brute follow
```

Two details that bite:

* **This project is (y, x); LW6's API is (x, y).**
* **LW6 cursor positions are integers.** The policy emits a velocity that is usually a
  fraction of a cell, so mod-nn keeps its position in floating point and rounds only
  on output. Rounding each move independently means a cursor asked to move 0.46 cells
  per round never moves at all.

## Results, and the asymmetry behind them

Against LW6's shipped bots, in real games, driven the way the game drives them:

| opponent | mod-nn share |
| --- | --- |
| `mod-follow` | 0.79 |
| `mod-stationary` | 0.67 |
| `mod-brute`, uncapped | 0.04 |

`mod-brute` looks unbeatable until you read it. It keeps a sandbox copy of the game
state, rolls it forward `nb_rounds_to_anticipate` rounds to score candidate cursor
positions, and — the important part — **teleports**: `cursor.pos.x =
lw6sys_random(sys_context, shape.w)`, anywhere on the map, every round. No human hand
can do that, and this project's policies cannot either, since they emit a velocity
bounded by `MAX_CURSOR_SPEED`. Uncapped, mod-brute takes 0.92 against mod-follow and
0.996 against a stationary cursor.

Capping every bot's cursor to the same mobility (`--cap`) gives a like-for-like
comparison:

| mod-brute's cursor cap | mod-nn share |
| --- | --- |
| 1 cell/round (what the policy has) | **0.44**, 50% wins |
| 2 cells/round | 0.42 |
| 4 cells/round | 0.17 |
| uncapped | 0.04 |

So the honest summary: the policy is about even with the strongest bot Liquid War 6
ships when both move a cursor at the same speed, and loses as that bot's mobility
advantage grows. The remaining gap is an action-space asymmetry as much as a policy
one — closing it means either giving the policy an absolute-target action, or
accepting the speed limit as the definition of fair play.

## Installing it in a real Liquid War 6 build

Not done here: the full game needs SDL, guile, libcurl and libjpeg, which are not
available on this machine, so only the kernel is built. To finish:

1. Copy `fluxwar/lw6/csrc/{mod_nn.c,fluxwar_nn.c,fluxwar_nn.h}` into
   `src/lib/bot/mod-nn/` of an LW6 source tree.
2. Add a `Makefile.am` modelled on `src/lib/bot/mod-follow/Makefile.am`, and the
   module to `src/lib/bot/bot-register.c`.
3. Build, and point `FLUXWAR_POLICY` at the exported `.flxw` file.

The C half is already exercised against the real kernel through
`fluxwar/eval/deployed.py`, so what is untested is the packaging, not the bot.
