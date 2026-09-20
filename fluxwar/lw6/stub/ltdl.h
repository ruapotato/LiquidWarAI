/* Minimal stand-in for libtool's ltdl.h. LW6's dyn.h needs only the lt_dlhandle
   type, and nothing compiled here loads a module at runtime. */
#ifndef FLUXWAR_LTDL_STUB_H
#define FLUXWAR_LTDL_STUB_H
typedef void *lt_dlhandle;
#endif
