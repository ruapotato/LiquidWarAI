/*
  Helpers for driving the Liquid War 6 kernel from Python.

  These exist so this project never has to guess LW6 struct offsets: this file
  includes LW6's own headers, so the compiler resolves them.

  Part of the fluxwar project; AGPLv3, linked against GPLv3 LW6 code.
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <string.h>
#include "map/map.h"
#include "ker/ker.h"

/*
 * Write an arbitrary wall mask into a level: 1 = open, 0 = wall, row-major, w*h
 * bytes. Needed because lw6map_builtin_custom always produces the same board --
 * noise_percent changes the texture, not the walls -- so without this every game is
 * played on one identical map. Call before building the game_struct.
 */
int
fluxwar_set_walls (lw6sys_context_t * sys_context, lw6map_level_t * level,
                   const unsigned char *walls, int w, int h)
{
  int x, y, z;
  lw6map_layer_t *layer;

  if (!level || !walls)
    return 0;
  if (level->body.shape.w != w || level->body.shape.h != h)
    return 0;

  for (z = 0; z < level->body.shape.d; ++z)
    {
      layer = &(level->body.layers[z]);
      if (!layer->data)
        continue;
      for (y = 0; y < h; ++y)
        for (x = 0; x < w; ++x)
          layer->data[y * w + x] = walls[y * w + x] ? 1 : 0;
    }
  lw6map_body_fix_checksum (sys_context, &(level->body));
  return 1;
}

/* Shape of a level's body, so Python does not have to know the struct layout. */
void
fluxwar_level_shape (const lw6map_level_t * level, int *w, int *h, int *d)
{
  *w = level->body.shape.w;
  *h = level->body.shape.h;
  *d = level->body.shape.d;
}

/* The rules block, for reading or overriding individual game rules. */
lw6map_rules_t *
fluxwar_level_rules (lw6map_level_t * level)
{
  return &(level->param.rules);
}

/*
 * Copy every active fighter into a flat int32 array of (team, health, y, x) rows.
 * Probing get_fighter_id per cell costs thousands of ctypes calls per round; this is
 * one.  Returns the number of rows written, or -1 if `max_rows` is too small.
 */
int
fluxwar_read_fighters (lw6sys_context_t * sys_context,
                       const lw6ker_game_state_t * game_state, int *out, int max_rows)
{
  int i, n;
  const lw6ker_fighter_t *f;

  n = lw6ker_game_state_get_nb_active_fighters (sys_context, game_state);
  if (n > max_rows)
    return -1;
  for (i = 0; i < n; ++i)
    {
      f = lw6ker_game_state_get_fighter_ro_by_id (sys_context, game_state, i);
      out[i * 4 + 0] = f->team_color;
      out[i * 4 + 1] = f->health;
      out[i * 4 + 2] = f->pos.y;
      out[i * 4 + 3] = f->pos.x;
    }
  return n;
}

/* Wall mask as the kernel sees it: 1 where a fighter may stand. */
void
fluxwar_read_walls (lw6sys_context_t * sys_context,
                    const lw6ker_game_struct_t * game_struct, unsigned char *out,
                    int w, int h)
{
  int x, y;
  for (y = 0; y < h; ++y)
    for (x = 0; x < w; ++x)
      out[y * w + x] =
        (lw6ker_game_struct_get_zone_id (sys_context, game_struct, x, y, 0) >= 0) ? 1 : 0;
}

/*
 * LW6 does not link with --enable-optimize: sys-context.c calls
 * _lw6sys_bazooka_context_init unconditionally, but sys-bazooka.c only defines it
 * when LW6_OPTIMIZE is *off*. Without LW6_OPTIMIZE the "bazooka" memory tracker is
 * compiled in and miscounts here ("more bytes freed than malloced"), so define the
 * flag and supply the one symbol it removes. With LW6_OPTIMIZE set, every other
 * bazooka entry point is already a no-op, so this is too.
 */
#ifdef LW6_OPTIMIZE
void
_lw6sys_bazooka_context_init (lw6sys_context_t * sys_context, void *bazooka_context)
{
  (void) sys_context;
  (void) bazooka_context;
}
#endif

/*
 * The public API has no way to get from a game_state back to its game_struct, but a
 * bot needs it: the wall mask lives in the struct, and `lw6bot_data_t` only carries
 * the state. The internal header knows the layout.
 */
#include "ker/ker-internal.h"

const lw6ker_game_struct_t *
fluxwar_game_state_struct (const lw6ker_game_state_t * game_state)
{
  return (const lw6ker_game_struct_t *) (((const _lw6ker_game_state_t *) game_state)
                                         ->game_struct);
}
