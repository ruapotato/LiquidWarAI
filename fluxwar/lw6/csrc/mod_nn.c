/*
  mod-nn: a Liquid War 6 bot backend that plays with a trained fluxwar policy.

  Implements LW6's bot interface (src/lib/bot/bot.h), so the game can load this the
  same way it loads mod-brute or mod-follow. The network is evaluated by
  fluxwar_nn.c, which has no dependencies -- linking libtorch into a game module is
  not reasonable -- and reads weights exported by fluxwar/deploy/export.py.

  The weight file is found via the FLUXWAR_POLICY environment variable, falling back
  to ./policy.flxw.

  Part of the fluxwar project; AGPLv3, linked against GPLv3 LW6 code.
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "bot/bot.h"
#include "ker/ker.h"
#include "map/map.h"

#include "fluxwar_nn.h"

/* Defined in fluxwar_helpers.c: the public API cannot get from a game_state to its
   game_struct, and the wall mask lives in the struct. */
extern const lw6ker_game_struct_t *fluxwar_game_state_struct (const lw6ker_game_state_t *
                                                              game_state);

typedef struct
{
  fw_net *net;
  int in_h, in_w;
  float cursor_speed;
  int action_mode;
  float *obs;
  int team_color;
  const lw6ker_game_struct_t *game_struct;
  /*
   * LW6 cursor positions are integers, but the policy emits a velocity that is
   * usually a fraction of a cell. Rounding each move independently throws that away
   * and a cursor asked to move 0.46 cells per round never moves at all. Keep the
   * position in floating point and hand LW6 the rounded value.
   */
  float fx, fy;
  int have_pos;
} _mod_nn_context_t;

/*
 * Build the observation exactly as fluxwar/sim/observation.py defines it: own
 * density, enemy density, walls, with the two density channels divided by the mean
 * total density per cell. That normalisation is what lets one set of weights work at
 * a map size and population the policy never trained on.
 *
 * LW6 maps are usually far larger than the training resolution, so cells are binned
 * down by nearest neighbour rather than fed at native size: the trunk is fully
 * convolutional and would accept any shape, but its features are scale-dependent.
 */
static void
_build_obs (lw6sys_context_t * sys_context, _mod_nn_context_t * ctx,
            const lw6ker_game_state_t * game_state, int team_color)
{
  int w = lw6ker_game_state_get_w (sys_context, game_state);
  int h = lw6ker_game_state_get_h (sys_context, game_state);
  int oh = ctx->in_h, ow = ctx->in_w;
  int x, y, ox, oy, i;
  int plane = oh * ow;
  float total = 0.0f, scale;

  ctx->game_struct = fluxwar_game_state_struct (game_state);

  for (i = 0; i < 3 * plane; ++i)
    ctx->obs[i] = 0.0f;

  for (y = 0; y < h; ++y)
    {
      oy = (y * oh) / h;
      for (x = 0; x < w; ++x)
        {
          int fid;
          ox = (x * ow) / w;
          if (lw6ker_game_struct_get_zone_id (sys_context, ctx->game_struct, x, y, 0) >= 0)
            ctx->obs[2 * plane + oy * ow + ox] = 1.0f;
          fid = lw6ker_game_state_get_fighter_id (sys_context, game_state, x, y, 0);
          if (fid >= 0)
            {
              const lw6ker_fighter_t *f =
                lw6ker_game_state_get_fighter_ro_by_id (sys_context, game_state, fid);
              int own = ((int) f->team_color == team_color);
              ctx->obs[(own ? 0 : 1) * plane + oy * ow + ox] += 1.0f;
              total += 1.0f;
            }
        }
    }

  scale = (total > 0.0f) ? (total / (float) plane) : 1.0f;
  for (i = 0; i < 2 * plane; ++i)
    ctx->obs[i] /= scale;
}

_mod_nn_context_t *
_mod_nn_init (lw6sys_context_t * sys_context, int argc, const char *argv[],
              lw6bot_data_t * data)
{
  _mod_nn_context_t *ctx = calloc (1, sizeof (_mod_nn_context_t));
  const char *path = getenv ("FLUXWAR_POLICY");

  if (!ctx)
    return NULL;
  ctx->net = fw_net_load (path ? path : "policy.flxw");
  if (!ctx->net)
    {
      lw6sys_log (sys_context, LW6SYS_LOG_WARNING,
                  _x_ ("mod-nn: cannot load policy weights from \"%s\"; "
                       "set FLUXWAR_POLICY"), path ? path : "policy.flxw");
      free (ctx);
      return NULL;
    }
  fw_net_input_shape (ctx->net, &ctx->in_h, &ctx->in_w);
  ctx->cursor_speed = fw_net_cursor_speed (ctx->net);
  ctx->action_mode = fw_net_action_mode (ctx->net);
  ctx->obs = malloc (sizeof (float) * 3 * ctx->in_h * ctx->in_w);
  if (!ctx->obs)
    {
      fw_net_free (ctx->net);
      free (ctx);
      return NULL;
    }
  ctx->team_color = -1;
  return ctx;
}

void
_mod_nn_quit (lw6sys_context_t * sys_context, _mod_nn_context_t * ctx)
{
  if (!ctx)
    return;
  fw_net_free (ctx->net);
  free (ctx->obs);
  free (ctx);
}

int
_mod_nn_next_move (lw6sys_context_t * sys_context, _mod_nn_context_t * ctx, int *x,
                   int *y, lw6bot_data_t * data)
{
  lw6ker_cursor_t cursor;
  float action[2];
  int w, h, cx, cy;

  if (!ctx || !data || !data->game_state)
    return 0;
  if (!lw6ker_game_state_get_cursor (sys_context, data->game_state, &cursor,
                                     data->param.cursor_id))
    return 0;

  _build_obs (sys_context, ctx, data->game_state, cursor.team_color);
  if (!fw_net_forward (ctx->net, ctx->obs, ctx->in_h, ctx->in_w, action))
    return 0;

  /*
   * The policy emits a velocity in [-1, 1]^2 as (dy, dx); LW6 wants an absolute
   * position as (x, y). One cell per round is MAX_CURSOR_SPEED in the training
   * config; scaling by the map's size ratio keeps the same real speed on a larger
   * map.
   */
  w = lw6ker_game_state_get_w (sys_context, data->game_state);
  h = lw6ker_game_state_get_h (sys_context, data->game_state);
  if (ctx->action_mode == 1)
    {
      /* The policy names an absolute position, the same freedom mod-brute takes. */
      cx = (int) lroundf (action[1] * (float) (w - 1));
      cy = (int) lroundf (action[0] * (float) (h - 1));
      *x = cx < 0 ? 0 : (cx >= w ? w - 1 : cx);
      *y = cy < 0 ? 0 : (cy >= h ? h - 1 : cy);
      return 1;
    }
  if (!ctx->have_pos)
    {
      ctx->fx = (float) cursor.pos.x;
      ctx->fy = (float) cursor.pos.y;
      ctx->have_pos = 1;
    }
  /* Scale by the map/training size ratio so the cursor covers the same fraction of
     the board per round whatever resolution the map is. */
  ctx->fx += action[1] * ctx->cursor_speed * ((float) w / (float) ctx->in_w);
  ctx->fy += action[0] * ctx->cursor_speed * ((float) h / (float) ctx->in_h);
  if (ctx->fx < 0.0f)
    ctx->fx = 0.0f;
  if (ctx->fy < 0.0f)
    ctx->fy = 0.0f;
  if (ctx->fx > (float) (w - 1))
    ctx->fx = (float) (w - 1);
  if (ctx->fy > (float) (h - 1))
    ctx->fy = (float) (h - 1);
  cx = (int) lroundf (ctx->fx);
  cy = (int) lroundf (ctx->fy);
  *x = cx;
  *y = cy;
  return 1;
}

char *
_mod_nn_repr (lw6sys_context_t * sys_context, _mod_nn_context_t * ctx, u_int32_t id)
{
  return lw6sys_new_sprintf (sys_context, "%u-%s", id, "nn");
}

/* ------------------------------------------------------------------ backend */

void
mod_nn_is_GPL_compatible ()
{
}

void
mod_nn_is_dlclose_safe ()
{
}

static void *
_init (lw6sys_context_t * sys_context, int argc, const char *argv[],
       lw6bot_data_t * data)
{
  return (void *) _mod_nn_init (sys_context, argc, argv, data);
}

static void
_quit (lw6sys_context_t * sys_context, void *bot_context)
{
  _mod_nn_quit (sys_context, (_mod_nn_context_t *) bot_context);
}

static int
_next_move (lw6sys_context_t * sys_context, void *bot_context, int *x, int *y,
            lw6bot_data_t * data)
{
  return _mod_nn_next_move (sys_context, (_mod_nn_context_t *) bot_context, x, y, data);
}

static char *
_repr (lw6sys_context_t * sys_context, void *bot_context, u_int32_t id)
{
  return _mod_nn_repr (sys_context, (_mod_nn_context_t *) bot_context, id);
}

lw6sys_module_pedigree_t *
mod_nn_get_pedigree (lw6sys_context_t * sys_context)
{
  lw6sys_module_pedigree_t *p =
    (lw6sys_module_pedigree_t *) LW6SYS_CALLOC (sys_context,
                                                sizeof (lw6sys_module_pedigree_t));
  if (p)
    {
      p->id = "nn";
      p->category = "bot";
      p->name = "Neural net";
      p->readme = "Plays with a convolutional policy trained by the fluxwar project.";
      p->version = VERSION;
      p->copyright = "AGPLv3";
      p->license = "AGPLv3+ (GNU AGPL version 3 or later)";
      p->date = __DATE__;
      p->time = __TIME__;
    }
  return p;
}

lw6bot_backend_t *
mod_nn_create_backend (lw6sys_context_t * sys_context)
{
  lw6bot_backend_t *backend =
    (lw6bot_backend_t *) LW6SYS_CALLOC (sys_context, sizeof (lw6bot_backend_t));
  if (backend)
    {
      backend->init = _init;
      backend->quit = _quit;
      backend->next_move = _next_move;
      backend->repr = _repr;
    }
  return backend;
}
