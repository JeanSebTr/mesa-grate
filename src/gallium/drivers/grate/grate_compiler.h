#ifndef GRATE_COMPILER_H
#define GRATE_COMPILER_H

#include "util/list.h"

#include <stdint.h>

struct tgsi_parse_context;
struct grate_fp_shader;
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

   /* what the draw code uploads to uniform register 32 + i (FP_CONST(i)) */
   uint32_t uniforms[96]; /* 32 after compile(), see fp/nir.c */
   int num_uniforms;
};

#define GRATE_FP_MAX_UNIFORMS 32
/* uniform table entries: kind in bits 28..31, payload below */
#define GRATE_FP_UNIFORM_KIND(u)     ((u) >> 28)
#define GRATE_FP_UNIFORM_PAYLOAD(u)  ((u) & 0x0fffffff)
#define GRATE_FP_UNIFORM_KIND_CB     1 /* dword of constant buffer 0 */
#define GRATE_FP_UNIFORM_KIND_IMM    2 /* float bits >> 4 */
#define GRATE_FP_UNIFORM_KIND_FB_Y   3 /* gl_FragCoord.y from pos.y: 0 scale, 1 offset */
#define GRATE_FP_UNIFORM_KIND_CB_PAIR 4 /* two dwords as fx10 low, high */
#define GRATE_FP_UNIFORM_CB(dword) \
   ((GRATE_FP_UNIFORM_KIND_CB << 28) | (dword))
#define GRATE_FP_UNIFORM_IMM(f) \
   ((GRATE_FP_UNIFORM_KIND_IMM << 28) | (fui(f) >> 4))
#define GRATE_FP_UNIFORM_FB_YSCALE (GRATE_FP_UNIFORM_KIND_FB_Y << 28)
#define GRATE_FP_UNIFORM_FB_YOFF ((GRATE_FP_UNIFORM_KIND_FB_Y << 28) | 1)
#define GRATE_FP_UNIFORM_CB_PAIR(lo, hi) \
   ((GRATE_FP_UNIFORM_KIND_CB_PAIR << 28) | (lo) | (hi) << 14)

struct nir_shader;
bool
grate_nir_to_fp(struct grate_fp_shader *fp, struct nir_shader *s,
                const char **error);
void
grate_nir_fs_lower(struct nir_shader *s);

struct grate_fp_shader {
   struct list_head fp_instructions;
   struct list_head alu_instructions;
   struct list_head mfu_instructions;
   struct grate_fp_info info;
   int alu_buf_size;
};

void
grate_tgsi_to_vp(struct grate_vp_shader *vp, struct tgsi_parse_context *tgsi);

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
