/* Self-contained inference for the fluxwar policy. See fluxwar_nn.c. AGPLv3. */
#ifndef FLUXWAR_NN_H
#define FLUXWAR_NN_H

typedef struct fw_net_s fw_net;

/* Load weights written by fluxwar/deploy/export.py. NULL on failure. */
fw_net *fw_net_load (const char *path);
void fw_net_free (fw_net *net);

/* The resolution the policy was trained at; feed it observations this size. */
void fw_net_input_shape (const fw_net *net, int *h, int *w);

/* Cells per round the policy was trained to move its cursor. Scale actions by it: a
   policy trained at 4 cells/round moves a quarter as far if you assume 1. */
float fw_net_cursor_speed (const fw_net *net);

/* 0 = velocity: out_action is a direction in [-1, 1], scale by the cursor speed.
   1 = target:   out_action is a normalised position in [0, 1], place the cursor
   there. The second is the action space LW6's own bots use. */
int fw_net_action_mode (const fw_net *net);

/*
 * obs: [3, h, w] row-major -- own density, enemy density, walls, normalised the way
 * fluxwar/sim/observation.py defines it. out_action: 2 floats, (dy, dx) in [-1, 1].
 * Returns 1 on success.
 */
int fw_net_forward (fw_net *net, const float *obs, int h, int w, float *out_action);

#endif
