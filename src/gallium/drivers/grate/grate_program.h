#ifndef GRATE_PROGRAM_H
#define GRATE_PROGRAM_H

#include "pipe/p_context.h"
#include "pipe/p_state.h"

#include "grate_compiler.h"

struct grate_shader_blob {
   uint32_t *commands;
   int num_commands;
};

struct grate_vertex_shader_state {
   struct pipe_shader_state base;
   struct grate_shader_blob blob;
   uint16_t output_mask;
   uint16_t output_semantic[16];
   uint32_t immediates[32][4];
   int num_immediates;
};

struct grate_fragment_shader_state {
   struct pipe_shader_state base;
   struct grate_shader_blob variants[GRATE_BLEND_COUNT];
   struct grate_fp_info info;
   struct nir_shader *nir; /* NIR compiler path, else base.tokens */
};

struct grate_context;

const struct grate_shader_blob *
grate_fs_variant(struct grate_context *context);

void
grate_context_program_init(struct pipe_context *pcontext);

#endif
