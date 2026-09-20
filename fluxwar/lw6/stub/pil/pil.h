/* Minimal stand-in for LW6's pil.h. The bot modules only pass lw6pil_pilot_t around
   as an opaque pointer, so the real pilot library -- and its network, message and
   connection dependencies -- is not needed to run LW6's bots headless. */
#ifndef FLUXWAR_PIL_STUB_H
#define FLUXWAR_PIL_STUB_H
typedef struct lw6pil_pilot_s lw6pil_pilot_t;
#endif
