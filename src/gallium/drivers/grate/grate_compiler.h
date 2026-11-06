#ifndef GRATE_COMPILER_H
#define GRATE_COMPILER_H

#include "util/list.h"

#include <stdint.h>

struct tgsi_parse_context;
struct fp_instr;
struct fp_mfu_instr;

struct grate_vp_shader {
   struct list_head instructions;
   uint16_t output_mask;
   /* semantic (name << 8 | index) of each output register, for linking */
   uint16_t output_semantic[16];
   /* TGSI immediates, uploaded from the top of the 256 VP uniforms down */
   uint32_t immediates[32][4];
   int num_immediates;
};

struct grate_fp_info {
   struct {
      uint32_t src;
      uint32_t dst;
      uint16_t semantic; /* name << 8 | index */
      int reg;           /* TGSI input register */
   } inputs[16];
   int num_inputs;
   int color_input;
   int max_tram_row;
};

struct grate_fp_shader {
   struct list_head fp_instructions;
   struct list_head alu_instructions;
   struct list_head mfu_instructions;
   struct grate_fp_info info;
   int tex_temp; /* TEMP holding the last texture sample (in R0-R1), or -1 */
   int alu_buf_size;
   struct fp_instr *tex_instr; /* last TEX, its MFU and texcoord TRAM row */
   struct fp_mfu_instr *tex_mfu;
   int tex_row; /* rgba row-register groups carried per fragment (R0-R3, R4-R7..) */
};

void
grate_tgsi_to_vp(struct grate_vp_shader *vp, struct tgsi_parse_context *tgsi);

void
grate_tgsi_to_fp(struct grate_fp_shader *fp, struct tgsi_parse_context *tgsi);

/*
 * Blending is done in the fragment program: GR3D has no blend unit.
 * Supported: dst factor ONE_MINUS_SRC_ALPHA with src factor ONE or SRC_ALPHA,
 * chosen separately for RGB and alpha.
 */
enum grate_blend_mode {
   GRATE_BLEND_NONE = 0,
   GRATE_BLEND_OVER,          /* rgb: ONE,       a: ONE       (premultiplied) */
   GRATE_BLEND_OVER_RGB_SA,   /* rgb: SRC_ALPHA, a: ONE       */
   GRATE_BLEND_OVER_A_SA,     /* rgb: ONE,       a: SRC_ALPHA */
   GRATE_BLEND_OVER_SA,       /* rgb: SRC_ALPHA, a: SRC_ALPHA (straight)      */
   GRATE_BLEND_COUNT
};

void
grate_fp_append_blend(struct grate_fp_shader *fp, enum grate_blend_mode mode);

#endif
