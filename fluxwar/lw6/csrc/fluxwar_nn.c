/*
  Self-contained inference for the fluxwar policy, for use inside a Liquid War 6 bot.

  Linking libtorch into a game plugin is not reasonable, and the network is small --
  three strided 3x3 convolutions, an adaptive average pool to 2x2, and two linear
  layers, about 90k parameters -- so the forward pass is written out here. It reads
  the flat format written by fluxwar/deploy/export.py.

  No dependency on LW6: this half is pure C and is unit-tested against PyTorch.

  Part of the fluxwar project; AGPLv3.
*/

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fluxwar_nn.h"

#define MAX_TENSORS 32
#define NAME_MAX_LEN 64

typedef struct
{
  char name[NAME_MAX_LEN];
  int ndim;
  int dims[4];
  int count;
  float *data;
} fw_tensor;

struct fw_net_s
{
  fw_tensor t[MAX_TENSORS];
  int n;
  int in_h, in_w;
  float cursor_speed;
  int action_mode;              /* 0 velocity, 1 target */
  float *scratch_a, *scratch_b, *scratch_mid;
  int scratch_len;
};

static fw_tensor *
find (fw_net *net, const char *name)
{
  int i;
  for (i = 0; i < net->n; ++i)
    if (!strcmp (net->t[i].name, name))
      return &net->t[i];
  return NULL;
}

fw_net *
fw_net_load (const char *path)
{
  FILE *f = fopen (path, "rb");
  char magic[4];
  uint32_t version, n, i, j, len, ndim, d;
  fw_net *net;

  if (!f)
    return NULL;
  if (fread (magic, 1, 4, f) != 4 || memcmp (magic, "FLXW", 4))
    {
      fclose (f);
      return NULL;
    }
  net = calloc (1, sizeof (fw_net));
  if (fread (&version, 4, 1, f) != 1 || version != 3)
    goto fail;
  if (fread (&net->in_h, 4, 1, f) != 1 || fread (&net->in_w, 4, 1, f) != 1)
    goto fail;
  if (fread (&net->cursor_speed, 4, 1, f) != 1)
    goto fail;
  if (fread (&net->action_mode, 4, 1, f) != 1)
    goto fail;
  if (fread (&n, 4, 1, f) != 1 || n > MAX_TENSORS)
    goto fail;
  net->n = (int) n;
  for (i = 0; i < n; ++i)
    {
      fw_tensor *t = &net->t[i];
      if (fread (&len, 4, 1, f) != 1 || len >= NAME_MAX_LEN)
        goto fail;
      if (fread (t->name, 1, len, f) != len)
        goto fail;
      t->name[len] = '\0';
      if (fread (&ndim, 4, 1, f) != 1 || ndim > 4)
        goto fail;
      t->ndim = (int) ndim;
      t->count = 1;
      for (j = 0; j < ndim; ++j)
        {
          if (fread (&d, 4, 1, f) != 1)
            goto fail;
          t->dims[j] = (int) d;
          t->count *= (int) d;
        }
      t->data = malloc (sizeof (float) * t->count);
      if (fread (t->data, sizeof (float), t->count, f) != (size_t) t->count)
        goto fail;
    }
  fclose (f);
  /* Two buffers big enough for the widest activation, ping-ponged between layers. */
  net->scratch_len = 64 * net->in_h * net->in_w;
  net->scratch_a = malloc (sizeof (float) * net->scratch_len);
  net->scratch_b = malloc (sizeof (float) * net->scratch_len);
  /* The target head reads the *middle* feature map, so it has to survive the third
     convolution rather than being ping-ponged over. */
  net->scratch_mid = malloc (sizeof (float) * net->scratch_len);
  return net;

fail:
  fclose (f);
  fw_net_free (net);
  return NULL;
}

void
fw_net_free (fw_net *net)
{
  int i;
  if (!net)
    return;
  for (i = 0; i < net->n; ++i)
    free (net->t[i].data);
  free (net->scratch_a);
  free (net->scratch_b);
  free (net->scratch_mid);
  free (net);
}

void
fw_net_input_shape (const fw_net *net, int *h, int *w)
{
  *h = net->in_h;
  *w = net->in_w;
}

float
fw_net_cursor_speed (const fw_net *net)
{
  return net->cursor_speed;
}

int
fw_net_action_mode (const fw_net *net)
{
  return net->action_mode;
}

static float
silu (float x)
{
  return x / (1.0f + expf (-x));
}

/* 3x3 convolution, stride 2, padding 1 -- the only convolution this net uses. */
static void
conv3x3_s2 (const float *in, int cin, int h, int w, const fw_tensor *weight,
            const fw_tensor *bias, float *out, int *oh, int *ow)
{
  int cout = weight->dims[0];
  int H = (h + 1) / 2, W = (w + 1) / 2;
  int oc, oy, ox, ic, ky, kx;

  for (oc = 0; oc < cout; ++oc)
    {
      const float *wbase = weight->data + (size_t) oc * cin * 9;
      float b = bias->data[oc];
      for (oy = 0; oy < H; ++oy)
        for (ox = 0; ox < W; ++ox)
          {
            float acc = b;
            for (ic = 0; ic < cin; ++ic)
              {
                const float *k = wbase + (size_t) ic * 9;
                const float *plane = in + (size_t) ic * h * w;
                for (ky = 0; ky < 3; ++ky)
                  {
                    int iy = oy * 2 + ky - 1;
                    if (iy < 0 || iy >= h)
                      continue;
                    for (kx = 0; kx < 3; ++kx)
                      {
                        int ix = ox * 2 + kx - 1;
                        if (ix < 0 || ix >= w)
                          continue;
                        acc += k[ky * 3 + kx] * plane[iy * w + ix];
                      }
                  }
              }
            out[((size_t) oc * H + oy) * W + ox] = silu (acc);
          }
    }
  *oh = H;
  *ow = W;
}

/* AdaptiveAvgPool2d(2): average over each of the four quadrants, PyTorch's split. */
static void
adaptive_avg_pool2 (const float *in, int c, int h, int w, float *out)
{
  int ch, by, bx, y, x;
  for (ch = 0; ch < c; ++ch)
    for (by = 0; by < 2; ++by)
      for (bx = 0; bx < 2; ++bx)
        {
          int y0 = (by * h) / 2, y1 = ((by + 1) * h + 1) / 2;
          int x0 = (bx * w) / 2, x1 = ((bx + 1) * w + 1) / 2;
          float acc = 0.0f;
          int n = 0;
          for (y = y0; y < y1; ++y)
            for (x = x0; x < x1; ++x)
              {
                acc += in[((size_t) ch * h + y) * w + x];
                n++;
              }
          out[((size_t) ch * 2 + by) * 2 + bx] = n ? acc / (float) n : 0.0f;
        }
}

static void
linear (const float *in, const fw_tensor *weight, const fw_tensor *bias, float *out,
        int apply_silu)
{
  int o, i, out_n = weight->dims[0], in_n = weight->dims[1];
  for (o = 0; o < out_n; ++o)
    {
      float acc = bias->data[o];
      const float *row = weight->data + (size_t) o * in_n;
      for (i = 0; i < in_n; ++i)
        acc += row[i] * in[i];
      out[o] = apply_silu ? silu (acc) : acc;
    }
}

int
fw_net_forward (fw_net *net, const float *obs, int h, int w, float *out_action)
{
  float *a = net->scratch_a, *b = net->scratch_b, *mid = net->scratch_mid;
  float pooled[256], hidden[128];
  int ch, cw, mh, mw, i;
  fw_tensor *w0 = find (net, "trunk.0.weight"), *b0 = find (net, "trunk.0.bias");
  fw_tensor *w2 = find (net, "trunk.2.weight"), *b2 = find (net, "trunk.2.bias");
  fw_tensor *w4 = find (net, "trunk.4.weight"), *b4 = find (net, "trunk.4.bias");
  fw_tensor *hw = find (net, "head.0.weight"), *hb = find (net, "head.0.bias");
  fw_tensor *mw_ = find (net, "mu.weight"), *mb = find (net, "mu.bias");
  fw_tensor *sw = find (net, "spatial.weight"), *sb = find (net, "spatial.bias");

  if (!w0 || !b0 || !w2 || !b2 || !w4 || !b4 || !hw || !hb || !mw_ || !mb)
    return 0;
  if (net->action_mode == 1 && (!sw || !sb))
    return 0;
  if (64 * h * w > net->scratch_len)
    {
      net->scratch_len = 64 * h * w;
      net->scratch_a = realloc (net->scratch_a, sizeof (float) * net->scratch_len);
      net->scratch_b = realloc (net->scratch_b, sizeof (float) * net->scratch_len);
      net->scratch_mid = realloc (net->scratch_mid, sizeof (float) * net->scratch_len);
      a = net->scratch_a;
      b = net->scratch_b;
      mid = net->scratch_mid;
    }

  conv3x3_s2 (obs, 3, h, w, w0, b0, a, &ch, &cw);
  conv3x3_s2 (a, w0->dims[0], ch, cw, w2, b2, mid, &mh, &mw);
  conv3x3_s2 (mid, w2->dims[0], mh, mw, w4, b4, b, &ch, &cw);

  if (net->action_mode == 1)
    {
      /* Target head: one logit per cell of the middle feature map, take the best,
         return its centre as a normalised position. Argmax, not a sample: a
         deployed bot should play its best guess. */
      int best = 0, n = mh * mw, c, cin = sw->dims[1];
      float best_v = -1.0e30f;
      for (i = 0; i < n; ++i)
        {
          float acc = sb->data[0];
          for (c = 0; c < cin; ++c)
            acc += sw->data[c] * mid[(size_t) c * n + i];
          if (acc > best_v)
            {
              best_v = acc;
              best = i;
            }
        }
      out_action[0] = ((float) (best / mw) + 0.5f) / (float) mh;
      out_action[1] = ((float) (best % mw) + 0.5f) / (float) mw;
      return 1;
    }

  adaptive_avg_pool2 (b, w4->dims[0], ch, cw, pooled);
  linear (pooled, hw, hb, hidden, 1);
  linear (hidden, mw_, mb, out_action, 0);
  for (i = 0; i < mw_->dims[0]; ++i)
    out_action[i] = tanhf (out_action[i]);
  return 1;
}
