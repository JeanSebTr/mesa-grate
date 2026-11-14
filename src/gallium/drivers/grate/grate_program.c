#include <stdio.h>
#include <string.h>

#include "util/u_dynarray.h"
#include "util/u_memory.h"

#include "nir/nir_to_tgsi.h"
#include "nir/tgsi_to_nir.h"
#include "compiler/nir/nir.h"
#include "compiler/nir/nir_serialize.h"
#include "util/blob.h"
#include "util/disk_cache.h"
#include "util/u_debug.h"
#include "tgsi/tgsi_dump.h"
#include "tgsi/tgsi_parse.h"
#include "tgsi/tgsi_scan.h"
#include "tgsi/tgsi_transform.h"

#include "host1x01_hardware.h"
#include "grate_common.h"
#include "grate_context.h"
#include "grate_screen.h"
#include "grate_program.h"
#include "grate_compiler.h"
#include "fp/fpir.h"
#include "vp/vpir.h"
#include "tgr_3d.xml.h"

struct grate_vs_tgsi_transform_context {
   struct tgsi_transform_context base;
   unsigned next_temp;
};

static void
grate_vs_tgsi_transform_instruction(struct tgsi_transform_context *ctx,
                                    struct tgsi_full_instruction *inst)
{
   struct grate_vs_tgsi_transform_context *gctx =
      (struct grate_vs_tgsi_transform_context *)ctx;

   int const_index = -1;
   for (unsigned i = 0; i < inst->Instruction.NumSrcRegs; i++) {
      if (inst->Src[i].Register.File != TGSI_FILE_CONSTANT)
         continue;

      if (const_index < 0)
         const_index = inst->Src[i].Register.Index;
      else if (inst->Src[i].Register.Index != const_index) {
         /* We already have a constant-reads in this instruction,
          * add extra instructions in order to read more
          */
         unsigned temp = gctx->next_temp++;
         tgsi_transform_temp_decl(ctx, temp);
         /* insert MOV to temp */
         tgsi_transform_op1_inst(ctx, TGSI_OPCODE_MOV,
                                 /* dst reg */
                                 TGSI_FILE_TEMPORARY, temp,
                                 TGSI_WRITEMASK_XYZW,
                                 /* src reg */
                                 TGSI_FILE_CONSTANT,
                                 inst->Src[i].Register.Index);

         /* rewrite instruction */
	 tgsi_transform_src_reg_xyzw(&inst->Src[i], TGSI_FILE_TEMPORARY, temp);
	 inst->Src[i].Register.File = TGSI_FILE_TEMPORARY;
         inst->Src[i].Register.Index = temp;
      }
   }

   ctx->emit_instruction(ctx, inst);
}

static struct tgsi_token *
grate_vs_tgsi_transform(const struct tgsi_token *tokens_in)
{
   const unsigned new_len = tgsi_num_tokens(tokens_in) + 100;

   struct tgsi_shader_info info;
   tgsi_scan_shader(tokens_in, &info);

   struct grate_vs_tgsi_transform_context ctx = {};
   ctx.base.transform_instruction = grate_vs_tgsi_transform_instruction;
   ctx.next_temp = info.file_max[TGSI_FILE_TEMPORARY] + 1;

   return tgsi_transform_shader(tokens_in, new_len, &ctx.base);
}


/* Mesa hands drivers NIR; grate still compiles TGSI. */
static const struct tgsi_token *
grate_shader_tokens(struct pipe_context *pcontext,
                    const struct pipe_shader_state *template)
{
   if (template->type == PIPE_SHADER_IR_NIR)
      return nir_to_tgsi(template->ir.nir, pcontext->screen);

   assert(template->type == PIPE_SHADER_IR_TGSI);
   return tgsi_dup_tokens(template->tokens);
}

static void *
grate_create_vs_state(struct pipe_context *pcontext,
                      const struct pipe_shader_state *template)
{
   struct grate_vertex_shader_state *so =
      CALLOC_STRUCT(grate_vertex_shader_state);

   if (!so)
      return NULL;

   const struct tgsi_token *tokens = grate_shader_tokens(pcontext, template);
   if (!tokens) {
      FREE(so);
      return NULL;
   }

   if (grate_debug & GRATE_DEBUG_TGSI) {
      fprintf(stderr, "DEBUG: TGSI:\n");
      tgsi_dump(tokens, 0);
      fprintf(stderr, "\n");
   }

   struct tgsi_token *new_tokens = grate_vs_tgsi_transform(tokens);
   FREE((void *)tokens);
   if (!new_tokens) {
      FREE(so);
      return NULL;
   }

   struct tgsi_parse_context parser;
   unsigned ok = tgsi_parse_init(&parser, new_tokens);
   assert(ok == TGSI_PARSE_OK);

   struct grate_vp_shader vp;
   grate_tgsi_to_vp(&vp, &parser);

   int num_instructions = list_length(&vp.instructions);
   assert(num_instructions < 256);
   int num_commands = 2 + num_instructions * 4;
   uint32_t *commands = MALLOC(num_commands * sizeof(uint32_t));
   if (!commands) {
      FREE(so);
      return NULL;
   }

   commands[0] = host1x_opcode_imm(TGR3D_VP_UPLOAD_INST_ID, 0);
   commands[1] = host1x_opcode_nonincr(TGR3D_VP_UPLOAD_INST,
                                       num_instructions * 4);

   struct vp_instr *last = list_last_entry(&vp.instructions, struct vp_instr, link);
   int offset = 2;
   list_for_each_entry(struct vp_instr, instr, &vp.instructions, link) {
      bool end_of_program = instr == last;
      grate_vp_pack(commands + offset, instr, end_of_program);
      offset += 4;
   }

   memcpy(so->immediates, vp.immediates, sizeof(so->immediates));
   so->num_immediates = vp.num_immediates;

   so->blob.commands = commands;
   so->blob.num_commands = num_commands;
   so->output_mask = vp.output_mask;
   memcpy(so->output_semantic, vp.output_semantic, sizeof(so->output_semantic));

   return so;
}

static void
grate_bind_vs_state(struct pipe_context *pcontext, void *so)
{
   grate_context(pcontext)->vshader = so;
}

static void
grate_delete_vs_state(struct pipe_context *pcontext, void *so)
{
   FREE(so);
}

static void
grate_fs_build_blob(struct grate_context *context, struct grate_fp_shader *fpp,
                    struct grate_shader_blob *blob);

/*
 * The on-disk cache holds what a compile produces for one blend mode: the
 * passes' command streams and linking info, and whether the shader was
 * supported. Keyed on the NIR, the blend mode and the chip; the cache
 * itself is per driver build (see grate_screen.c).
 */
#define GRATE_FS_CACHE_VERSION 1

static void
grate_fs_cache_key(struct grate_context *context,
                   struct grate_fragment_shader_state *fs,
                   enum grate_blend_mode mode, cache_key key)
{
   struct disk_cache *cache = grate_screen(context->base.screen)->disk_cache;
   struct blob b;
   blob_init(&b);
   blob_write_uint32(&b, GRATE_FS_CACHE_VERSION);
   blob_write_uint32(&b, mode);
   blob_write_uint32(&b, context->tegra114);
   nir_serialize(&b, fs->nir, true);
   disk_cache_compute_key(cache, b.data, b.size, key);
   blob_finish(&b);
}

static bool
grate_fs_cache_load(struct grate_context *context,
                    struct grate_fragment_shader_state *fs,
                    enum grate_blend_mode mode, const cache_key key)
{
   struct disk_cache *cache = grate_screen(context->base.screen)->disk_cache;
   size_t size;
   void *data = disk_cache_get(cache, key, &size);
   if (!data)
      return false;

   struct blob_reader r;
   blob_reader_init(&r, data, size);
   int num_segments = blob_read_uint32(&r);
   bool supported = blob_read_uint32(&r);
   struct grate_shader_blob blobs[GRATE_FP_MAX_SEGMENTS] = { 0 };
   struct grate_fp_info info[GRATE_FP_MAX_SEGMENTS];
   bool ok = num_segments >= 1 && num_segments <= GRATE_FP_MAX_SEGMENTS;
   for (int i = 0; ok && i < num_segments; i++) {
      blobs[i].num_commands = blob_read_uint32(&r);
      blobs[i].commands = MALLOC(blobs[i].num_commands * sizeof(uint32_t));
      blob_copy_bytes(&r, blobs[i].commands,
                      blobs[i].num_commands * sizeof(uint32_t));
      blob_copy_bytes(&r, &info[i], sizeof(info[i]));
   }
   ok &= !r.overrun;
   free(data);

   if (!ok) {
      for (int i = 0; i < GRATE_FP_MAX_SEGMENTS; i++)
         FREE(blobs[i].commands);
      return false;
   }
   if (!supported && mode == GRATE_BLEND_NONE)
      fprintf(stderr, "grate: fragment shader not supported (cached)\n");
   for (int i = 0; i < num_segments; i++) {
      fs->variants[mode][i] = blobs[i];
      if (mode == GRATE_BLEND_NONE)
         fs->info[i] = info[i];
   }
   if (mode == GRATE_BLEND_NONE)
      fs->num_segments = num_segments;
   return true;
}

static void
grate_fs_cache_store(struct grate_context *context,
                     struct grate_fragment_shader_state *fs,
                     enum grate_blend_mode mode, const cache_key key,
                     int num_segments, const struct grate_fp_info *info,
                     bool supported)
{
   struct disk_cache *cache = grate_screen(context->base.screen)->disk_cache;
   struct blob b;
   blob_init(&b);
   blob_write_uint32(&b, num_segments);
   blob_write_uint32(&b, supported);
   for (int i = 0; i < num_segments; i++) {
      const struct grate_shader_blob *sb = &fs->variants[mode][i];
      blob_write_uint32(&b, sb->num_commands);
      blob_write_bytes(&b, sb->commands, sb->num_commands * sizeof(uint32_t));
      blob_write_bytes(&b, &info[i], sizeof(info[i]));
   }
   if (!b.out_of_memory)
      disk_cache_put(cache, key, b.data, b.size, NULL);
   blob_finish(&b);
}

static bool
grate_fs_compile(struct grate_context *context,
                 struct grate_fragment_shader_state *fs,
                 enum grate_blend_mode mode)
{
   struct disk_cache *cache = grate_screen(context->base.screen)->disk_cache;
   struct grate_fp_shader fps[GRATE_FP_MAX_SEGMENTS];
   struct grate_fp_info info[GRATE_FP_MAX_SEGMENTS];
   int num_segments = 1;
   cache_key key;

   if (cache) {
      grate_fs_cache_key(context, fs, mode, key);
      if (grate_fs_cache_load(context, fs, mode, key))
         return true;
   }

   const char *error = NULL;
   bool supported = grate_nir_to_fp(fps, &num_segments, fs->nir, &error);
   if (!supported) {
      /* drawn as nothing, see fp/nir.c */
      fprintf(stderr, "grate: fragment shader not supported: %s\n", error);
      if (grate_debug & GRATE_DEBUG_FS)
         nir_print_shader(fs->nir, stderr);
   }
   /* blending happens in the last pass, the one that writes the colour */
   grate_fp_append_blend(&fps[num_segments - 1], mode);

   for (int i = 0; i < num_segments; i++) {
      info[i] = fps[i].info;
      grate_fs_build_blob(context, &fps[i], &fs->variants[mode][i]);
      if (mode == GRATE_BLEND_NONE)
         fs->info[i] = info[i];
   }
   if (mode == GRATE_BLEND_NONE)
      fs->num_segments = num_segments;

   if (cache)
      grate_fs_cache_store(context, fs, mode, key, num_segments, info,
                           supported);
   return true;
}

static void
grate_fs_build_blob(struct grate_context *context, struct grate_fp_shader *fpp,
                    struct grate_shader_blob *blob)
{
#define fp (*fpp)

   struct util_dynarray buf;
   util_dynarray_init(&buf, NULL);

#define PUSH(x) util_dynarray_append_typed(&buf, uint32_t, (x))
   PUSH(host1x_opcode_incr(TGR3D_ALU_BUFFER_SIZE, 1));
   /* SIZEx4 = (pseq cfg 0x12c - 1) / (size * 4), matches 0x501 */
   PUSH(TGR3D_VAL(ALU_BUFFER_SIZE, SIZE, fp.alu_buf_size - 1) |
        TGR3D_VAL(ALU_BUFFER_SIZE, SIZEx4, (0x12c - 1) / (fp.alu_buf_size * 4)));

   PUSH(host1x_opcode_imm(TGR3D_FP_PSEQ_QUAD_ID, 0));
   PUSH(host1x_opcode_imm(TGR3D_FP_UPLOAD_INST_ID_COMMON, 0));
   PUSH(host1x_opcode_imm(TGR3D_FP_UPLOAD_MFU_INST_ID, 0));
   PUSH(host1x_opcode_imm(TGR3D_FP_UPLOAD_ALU_INST_ID, 0));

   int num_fp_instrs = list_length(&fp.fp_instructions);
   assert(num_fp_instrs < 64);

   PUSH(host1x_opcode_incr(TGR3D_FP_PSEQ_ENGINE_INST, 1));
   PUSH(0x20006000 | num_fp_instrs);

   /* the (1-based) instruction that does the DW */
   int dw_exec_nb = 1, n = 0;
   list_for_each_entry(struct fp_instr, instr, &fp.fp_instructions, link) {
      n++;
      if (instr->dw.enable)
         dw_exec_nb = n;
   }

   PUSH(host1x_opcode_incr(TGR3D_FP_PSEQ_DW_CFG, 1));
   PUSH(TGR3D_VAL(FP_PSEQ_DW_CFG, PSEQ_TO_DW_EXEC_NB, dw_exec_nb));

   if (context->tegra114) {
      /* XXX: maybe not needed */
      PUSH(host1x_opcode_incr(0x547, 0x0002));
      PUSH(0xc0000000);
      PUSH(0x00000000);
   }

   PUSH(host1x_opcode_imm(TGR3D_FP_PSEQ_UPLOAD_INST_BUFFER_FLUSH, 0));

   PUSH(host1x_opcode_nonincr(TGR3D_FP_PSEQ_UPLOAD_INST, num_fp_instrs));
   list_for_each_entry(struct fp_instr, instr, &fp.fp_instructions, link)
      PUSH(instr->pseq);

   PUSH(host1x_opcode_nonincr(TGR3D_FP_UPLOAD_MFU_SCHED, num_fp_instrs));
   list_for_each_entry(struct fp_instr, instr, &fp.fp_instructions, link)
      PUSH(grate_fp_pack_sched(&instr->mfu_sched));

   int num_mfu_instrs = list_length(&fp.mfu_instructions);
   assert(num_mfu_instrs < 64); // TODO: not sure if this is really correct

   PUSH(host1x_opcode_nonincr(TGR3D_FP_UPLOAD_MFU_INST, num_mfu_instrs * 2));
   list_for_each_entry(struct fp_mfu_instr, instr, &fp.mfu_instructions, link) {
      uint32_t words[2];
      grate_fp_pack_mfu(words, instr);
      PUSH(words[0]);
      PUSH(words[1]);
   }

   PUSH(host1x_opcode_nonincr(TGR3D_FP_UPLOAD_TEX_INST, num_fp_instrs));
   list_for_each_entry(struct fp_instr, instr, &fp.fp_instructions, link)
      PUSH(grate_fp_pack_tex(&instr->tex));

   PUSH(host1x_opcode_nonincr(TGR3D_FP_UPLOAD_ALU_SCHED, num_fp_instrs));
   list_for_each_entry(struct fp_instr, instr, &fp.fp_instructions, link) {
      if (context->tegra114)
         PUSH(grate_fp_pack_alu_sched_t114(&instr->alu_sched));
      else
         PUSH(grate_fp_pack_sched(&instr->alu_sched));
   }

   int num_alu_instrs = list_length(&fp.alu_instructions);
   PUSH(host1x_opcode_nonincr(TGR3D_FP_UPLOAD_ALU_INST,
        num_alu_instrs * 4 * 2));
   list_for_each_entry(struct fp_alu_instr_packet, instr, &fp.alu_instructions, link) {
      for (int i = 0; i < 4; ++i) {
         uint32_t words[2];
         if (i == 3 && instr->has_imm) {
            /* not swapped, unlike instructions */
            uint64_t v = (uint64_t)instr->imm[0] << 4 |
                         (uint64_t)instr->imm[1] << 24 |
                         (uint64_t)instr->imm[2] << 44;
            PUSH((uint32_t)v);
            PUSH((uint32_t)(v >> 32));
            continue;
         }
         grate_fp_pack_alu(words, instr->slots + i);
         PUSH(words[0]);
         PUSH(words[1]);
      }
   }

   PUSH(host1x_opcode_nonincr(TGR3D_FP_UPLOAD_ALU_INST_COMPLEMENT, num_fp_instrs));
   list_for_each_entry(struct fp_instr, instr, &fp.fp_instructions, link)
      PUSH(0x00000000);

   PUSH(host1x_opcode_nonincr(TGR3D_FP_UPLOAD_DW_INST, num_fp_instrs));
   list_for_each_entry(struct fp_instr, instr, &fp.fp_instructions, link)
      PUSH(grate_fp_pack_dw(&instr->dw));

   uint32_t tram_setup = 0;
   tram_setup |= TGR3D_VAL(TRAM_SETUP, USED_TRAM_ROWS_NB, fp.info.max_tram_row);
   tram_setup |= TGR3D_VAL(TRAM_SETUP, DIV64, 64 / fp.info.max_tram_row);

   PUSH(host1x_opcode_incr(TGR3D_TRAM_SETUP, 1));
   PUSH(tram_setup);

#undef PUSH
   util_dynarray_trim(&buf);

   blob->num_commands = buf.size / sizeof(uint32_t);
   blob->commands = buf.data;
   if (getenv("GRATE_FP_DUMP"))
      for (int i = 0; i < blob->num_commands; i++)
         fprintf(stderr, "DUMP %08x\n", blob->commands[i]);

   list_for_each_entry_safe(struct fp_instr, i, &fp.fp_instructions, link)
      FREE(i);
   list_for_each_entry_safe(struct fp_alu_instr_packet, p, &fp.alu_instructions, link)
      FREE(p);
   list_for_each_entry_safe(struct fp_mfu_instr, m, &fp.mfu_instructions, link)
      FREE(m);
#undef fp
}

static void *
grate_create_fs_state(struct pipe_context *pcontext,
                      const struct pipe_shader_state *template)
{
   struct grate_context *context = grate_context(pcontext);
   struct grate_fragment_shader_state *so =
      CALLOC_STRUCT(grate_fragment_shader_state);

   if (!so)
      return NULL;

   /* Mesa's own utility shaders can still come as TGSI */
   so->nir = template->type == PIPE_SHADER_IR_NIR ? template->ir.nir :
             tgsi_to_nir(template->tokens, pcontext->screen, false);
   grate_nir_fs_lower(so->nir);
   if (grate_debug & GRATE_DEBUG_FS)
      nir_print_shader(so->nir, stderr);

   if (!grate_fs_compile(context, so, GRATE_BLEND_NONE)) {
      ralloc_free(so->nir);
      FREE(so);
      return NULL;
   }
   return so;
}

static enum grate_blend_mode
grate_blend_mode(const struct pipe_blend_state *blend)
{
   const struct pipe_rt_blend_state *rt = &blend->rt[0];

   if (!rt->blend_enable)
      return GRATE_BLEND_NONE;

   if (rt->rgb_func == PIPE_BLEND_ADD && rt->alpha_func == PIPE_BLEND_ADD) {
      bool rgb_one = rt->rgb_src_factor == PIPE_BLENDFACTOR_ONE;
      bool rgb_sa = rt->rgb_src_factor == PIPE_BLENDFACTOR_SRC_ALPHA;
      bool a_one = rt->alpha_src_factor == PIPE_BLENDFACTOR_ONE;
      bool a_sa = rt->alpha_src_factor == PIPE_BLENDFACTOR_SRC_ALPHA;

      if (rt->rgb_dst_factor == PIPE_BLENDFACTOR_ZERO &&
          rt->alpha_dst_factor == PIPE_BLENDFACTOR_ZERO && rgb_one && a_one)
         return GRATE_BLEND_NONE;

      if (rt->rgb_dst_factor == PIPE_BLENDFACTOR_INV_SRC_ALPHA &&
          rt->alpha_dst_factor == PIPE_BLENDFACTOR_INV_SRC_ALPHA &&
          (rgb_one || rgb_sa) && (a_one || a_sa))
         return GRATE_BLEND_OVER + rgb_sa + 2 * a_sa;
   }

   static bool warned;
   if (!warned) {
      fprintf(stderr, "grate: unsupported blend state, drawing unblended\n");
      warned = true;
   }
   return GRATE_BLEND_NONE;
}

const struct grate_shader_blob *
grate_fs_variant(struct grate_context *context)
{
   struct grate_fragment_shader_state *fs = context->fshader;
   enum grate_blend_mode mode =
      context->blend ? grate_blend_mode(context->blend) : GRATE_BLEND_NONE;

   if (!fs->variants[mode][0].commands)
      grate_fs_compile(context, fs, mode);

   return &fs->variants[mode][context->fs_pass];
}

static void
grate_bind_fs_state(struct pipe_context *pcontext, void *so)
{
   grate_context(pcontext)->fshader = so;
}

static void
grate_delete_fs_state(struct pipe_context *pcontext, void *so)
{
   struct grate_fragment_shader_state *fs = so;

   for (int i = 0; i < GRATE_BLEND_COUNT; i++)
      for (int j = 0; j < GRATE_FP_MAX_SEGMENTS; j++)
         FREE(fs->variants[i][j].commands);
   ralloc_free(fs->nir);
   FREE(so);
}

void
grate_context_program_init(struct pipe_context *pcontext)
{
   pcontext->create_vs_state = grate_create_vs_state;
   pcontext->bind_vs_state = grate_bind_vs_state;
   pcontext->delete_vs_state = grate_delete_vs_state;

   pcontext->create_fs_state = grate_create_fs_state;
   pcontext->bind_fs_state = grate_bind_fs_state;
   pcontext->delete_fs_state = grate_delete_fs_state;
}
