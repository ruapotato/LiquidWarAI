# Running the real Liquid War 6 kernel

This project does not only imitate Liquid War 6 -- it links against it. `liblw6ker` is
compiled from the GNU LW6 sources and driven from Python, so "the reference" is the
game, not a re-implementation of it.

```bash
python -m fluxwar.lw6.build --out build/lw6      # clones the sources if needed
python -m fluxwar.eval.fidelity                  # density sim vs the real thing
python -m fluxwar.policy.ppo --env lw6           # train on real games
```

## What gets built, and why it is possible

Only the kernel: `ker` (the rules), `map` (the level representation) and `sys`
(portability helpers). `ker` includes nothing outside `map` and `sys`, and `sys` needs
nothing beyond libc and pthread, so none of the game's real dependencies (SDL, guile,
curl, libjpeg) are involved and autotools is not needed. About 100 source files,
compiled in a couple of seconds.

Five things had to be got right, none of them obvious, all of them in
`fluxwar/lw6/build.py`:

* **`-DHAVE_CONFIG_H`** or `config.h` is never included and every file silently
  compiles against its fallback defaults.
* **`-DLW6_AMD64=1`.** Without it the spinlocks fall through to a debug mutex path
  that logs at INFO on every lock. A 250-round run took ten minutes and wrote tens of
  megabytes to stderr.
* **`sys-testandsetamd64.s`** must be assembled in: the lock-free primitive that
  `-DLW6_AMD64` selects is hand-written assembly, not in any `.c` file. Filtering test
  files by "-test" anywhere in the name also drops `sys-testandset.c`, which is where
  its C wrapper lives, and the library then fails to load.
* **`-DLW6_OPTIMIZE=1`** plus one stub. Upstream does not link with this flag --
  `sys-context.c` calls `_lw6sys_bazooka_context_init` unconditionally while
  `sys-bazooka.c` only defines it when the flag is *off* -- but without it the
  "bazooka" memory tracker is compiled in and aborts here with "more bytes freed than
  malloced". `csrc/fluxwar_helpers.c` supplies the missing symbol as a no-op.
* **No `-fopenmp`.** The kernel parallelises the gradient spread, which is worthless
  for a single game, and the library would share `libgomp` with PyTorch.

## Talking to it

`fluxwar/lw6/kernel.py` is the ctypes layer and `fluxwar/lw6/env.py` wraps one game in
the same interface the density sim exposes, so the same agents play both.
`fluxwar/lw6/csrc/fluxwar_helpers.c` is compiled in alongside so that struct layouts
are resolved by the compiler rather than guessed from headers.

Things worth knowing before extending it:

* **The context is a process-wide singleton.** Creating and freeing one per game
  segfaults on the second. A forked worker must call `reset_process_state()` so it
  builds its own rather than inheriting the parent's pointer.
* **Logging defaults to INFO** and must be turned off with *both* `lw6sys_log_set_level`
  and `lw6sys_debug_set` -- `lw6sys_log` prints when either allows it.
* **A cursor must belong to a registered node.** `lw6ker_game_state_add_cursor` just
  returns 0 otherwise, with no other complaint.
* **`lw6map_builtin_custom` always returns the same board.** `noise_percent` changes
  the texture, not the walls, so without `fluxwar_set_walls` every "random" game is
  the same map. Pass `walls=` to `LW6Game` to inject one.
* **Read the fighter array in one call.** `fluxwar_read_fighters` copies every active
  fighter at once; probing `get_fighter_id` per cell costs thousands of ctypes calls
  and drops throughput from ~5000 rounds/s to 67.
* **Cursors may sit on walls.** LW6 applies them at the nearest free slot
  (`find_free_slot_near`). Blocking a cursor on walls is unfaithful *and* fatal: a
  straight-line chaser wedges in the first concave corner and never moves again.

## Training on real games

`fluxwar/lw6/vecenv.py` runs many real games across worker processes and hands
observations to the parent through shared memory. One game runs at ~1900 rounds/s, and
a dozen workers reach ~6k env-steps/s -- slower than the differentiable sim's ~90k,
but with no sim-to-real gap at all. The policy still runs on the GPU; only the physics
are on CPU.

LW6's fighter update is sequential and order-dependent, so a game cannot be batched on
a GPU and cannot be differentiated. That is the whole reason the density sim exists,
and the reason `eval/transfer.py` reports both columns side by side.
