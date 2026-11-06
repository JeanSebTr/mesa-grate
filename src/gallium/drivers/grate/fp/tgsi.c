#include <stdlib.h>
#include "grate_compiler.h"
#include "fpir.h"

#include "tgsi/tgsi_parse.h"

#include "util/u_memory.h"

#define LINK_SRC(index) ((index) << 3)
#define LINK_DST(index, comp, type) (((comp) | (type) << 2) << ((index) * 4))
#define LINK_DST_NONE      0
#define LINK_DST_FX10_LOW  1
#define LINK_DST_FX10_HIGH 2
#define LINK_DST_FP20      3

static struct fp_alu_src_operand
fp_alu_src_row(int index)
{
   assert(index >= 0 && index < 16);
   struct fp_alu_src_operand src = {
      .index = index
   };
   return src;
}

static struct fp_alu_src_operand
fp_alu_src_reg(int index)
{
   assert(index >= 0 && index < 8);
   struct fp_alu_src_operand src = {
      .index = 16 + index
   };
   return src;
}

static struct fp_alu_src_operand
fp_alu_src_zero()
{
   struct fp_alu_src_operand src = {
      .index = 31,
      .datatype = FP_DATATYPE_FIXED10,
      .sub_reg_select_high = 0
   };
   return src;
}

static struct fp_alu_src_operand
fp_alu_src_one()
{
   struct fp_alu_src_operand src = {
      .index = 31,
      .datatype = FP_DATATYPE_FIXED10,
      .sub_reg_select_high = 1
   };
   return src;
}

static struct fp_alu_instr
fp_alu_sMOV(struct fp_alu_dst_operand dst, struct fp_alu_src_operand src)
{
   struct fp_alu_instr ret = {
      .op = FP_ALU_OP_MAD,
      .dst = dst,
      .src = {
         src,
         fp_alu_src_one(),
         fp_alu_src_zero(),
         fp_alu_src_one()
      }
   };
   return ret;
}

static struct fp_alu_dst_operand
fp_alu_dst(const struct tgsi_dst_register *dst, int subreg, bool saturate)
{
   struct fp_alu_dst_operand ret = { 0 };

   ret.index = dst->Index;
   if (dst->File == TGSI_FILE_OUTPUT) {
      ret.index = 2; // HACK: r2+r3 to match hard-coded store shader for now

      // fixed10
      // swizzle RGBA -> BGRA
      int o = subreg < 3 ? (2 - subreg) : 3;
      ret.index += o / 2;
      ret.write_low_sub_reg = (o % 2) == 0;
      ret.write_high_sub_reg = (o % 2) != 0;
   } else
      ret.index += subreg;

   ret.saturate = saturate;

   return ret;
}

static void
emit_vMOV(struct grate_fp_shader *fp, const struct tgsi_dst_register *dst,
          bool saturate, const struct tgsi_src_register *src)
{
   struct fp_instr *inst = CALLOC_STRUCT(fp_instr);
   list_inithead(&inst->link);

   struct fp_mfu_instr *mfu = NULL;
   if (src->File == TGSI_FILE_INPUT) {
      mfu = CALLOC_STRUCT(fp_mfu_instr);
      list_inithead(&mfu->link);
   }

   int swizzle[] = {
      src->SwizzleX,
      src->SwizzleY,
      src->SwizzleZ,
      src->SwizzleW
   };

   struct fp_alu_instr_packet *alu = CALLOC_STRUCT(fp_alu_instr_packet);
   int alu_instrs = 0;
   list_inithead(&alu->link);
   for (int i = 0; i < 4; ++i) {
      if ((dst->WriteMask & (1 << i)) == 0)
         continue;

      int comp = swizzle[i];

      struct fp_alu_src_operand src0 = { };
      if (src->File == TGSI_FILE_INPUT) {
         mfu->var[i].op = FP_VAR_OP_FP20;
         mfu->var[i].tram_row = src->Index;
         fp->info.max_tram_row = MAX2(fp->info.max_tram_row, src->Index + 1);
         src0 = fp_alu_src_row(comp);
      } else
         src0 = fp_alu_src_reg(src->Index + comp);

      alu->slots[alu_instrs++] = fp_alu_sMOV(fp_alu_dst(dst, i, saturate), src0);
   }
   inst->alu_sched.num_instructions = 1;
   inst->alu_sched.address = list_length(&fp->alu_instructions);

   inst->mfu_sched.address = list_length(&fp->mfu_instructions);
   if (mfu != NULL) {
      inst->mfu_sched.num_instructions = 1;
      list_addtail(&mfu->link, &fp->mfu_instructions);
   }

   if (dst->File == TGSI_FILE_OUTPUT) {
      inst->dw.enable = 1;
      inst->dw.index = 1 + dst->Index;
      inst->dw.stencil_write = 0;
      inst->dw.src_regs = FP_DW_REGS_R2_R3; // hard-coded for now
   }

   list_addtail(&alu->link, &fp->alu_instructions);
   list_addtail(&inst->link, &fp->fp_instructions);
}

/*
 * TEX: interpolate S, T from the texcoord varying into R0, R1 (MFU), sample
 * into R0-R1 as four fx10 (TEX). The hardware RGBA8888 layout is the render
 * target's, so a sample written straight back with DW needs no swizzle: the
 * texel sits in R0-R1 as B, G, R, A like the color output in R2-R3.
 */
static void
emit_TEX(struct grate_fp_shader *fp, const struct tgsi_full_instruction *tinst)
{
   const struct tgsi_dst_register *dst = &tinst->Dst[0].Register;
   const struct tgsi_src_register *src = &tinst->Src[0].Register;
   unsigned sampler = tinst->Src[1].Register.Index;
   int swizzle[] = { src->SwizzleX, src->SwizzleY };

   assert(src->File == TGSI_FILE_INPUT);
   assert(tinst->Texture.Texture == TGSI_TEXTURE_2D ||
          tinst->Texture.Texture == TGSI_TEXTURE_RECT);

   struct fp_instr *inst = CALLOC_STRUCT(fp_instr);
   list_inithead(&inst->link);

   struct fp_mfu_instr *mfu = CALLOC_STRUCT(fp_mfu_instr);
   list_inithead(&mfu->link);
   for (int i = 0; i < 2; ++i) {
      /* var[i] interpolates component i of the row into row register i */
      assert(swizzle[i] == i);
      mfu->var[i].op = FP_VAR_OP_FP20;
      mfu->var[i].tram_row = src->Index;
   }
   fp->info.max_tram_row = MAX2(fp->info.max_tram_row, src->Index + 1);

   inst->mfu_sched.num_instructions = 1;
   inst->mfu_sched.address = list_length(&fp->mfu_instructions);
   list_addtail(&mfu->link, &fp->mfu_instructions);

   inst->tex.enable = true;
   inst->tex.sampler = sampler;
   fp->tex_instr = inst;
   fp->tex_mfu = mfu;
   fp->tex_row = src->Index;

   inst->alu_sched.num_instructions = 0;
   inst->alu_sched.address = list_length(&fp->alu_instructions);

   if (dst->File == TGSI_FILE_OUTPUT) {
      assert(dst->WriteMask == TGSI_WRITEMASK_XYZW);
      inst->dw.enable = 1;
      inst->dw.index = 1 + dst->Index;
      inst->dw.src_regs = FP_DW_REGS_R0_R1;
   } else {
      assert(dst->File == TGSI_FILE_TEMPORARY);
      fp->tex_temp = dst->Index;
   }

   list_addtail(&inst->link, &fp->fp_instructions);
}

/* MOV OUT, TEMP holding a texture sample: store R0-R1 */
static void
emit_store_tex_temp(struct grate_fp_shader *fp, const struct tgsi_dst_register *dst)
{
   struct fp_instr *inst = CALLOC_STRUCT(fp_instr);
   list_inithead(&inst->link);
   inst->mfu_sched.address = list_length(&fp->mfu_instructions);
   inst->alu_sched.address = list_length(&fp->alu_instructions);
   inst->dw.enable = 1;
   inst->dw.index = 1 + dst->Index;
   inst->dw.src_regs = FP_DW_REGS_R0_R1;
   list_addtail(&inst->link, &fp->fp_instructions);
}

/*
 * MUL OUT, TEMP holding a texture sample, IN (varying), done the opentegra
 * way in the TEX instruction itself: everything interpolated in the first
 * batch (later batches can't interpolate reliably, and GR3D global registers
 * are shared between fragments). The linker packs the varying as fx10 BGRA
 * into the free z/w components of the texcoord TRAM row, so one ipl gives
 * the coordinates in R0-R1 (fp20) and the colour in R2-R3 (fx10), and the
 * texel (R0-R1, fx10 BGRA) is multiplied by it before the store.
 */
static void
emit_modulate_tex_temp(struct grate_fp_shader *fp, const struct tgsi_dst_register *dst,
                       const struct tgsi_src_register *in)
{
   struct fp_instr *inst = fp->tex_instr;
   struct grate_fp_info *info = &fp->info;
   int c = -1;

   for (int i = 0; i < info->num_inputs; i++)
      if (info->inputs[i].reg == (int)in->Index)
         c = i;
   assert(c >= 0);

   /*
    * Linker words are indexed by export component, each naming its TRAM
    * destination component and type: r -> w.lo, g -> z.hi, b -> z.lo,
    * a -> w.hi, so TRAM.z/w interpolate as fx10 B G / R A into R2/R3.
    */
   assert(in->SwizzleX == 0 && in->SwizzleY == 1 && in->SwizzleZ == 2 && in->SwizzleW == 3);
   info->inputs[c].src = (info->inputs[c].src & ~(0x3f << 9)) | fp->tex_row << 9;
   info->inputs[c].dst = LINK_DST(0, 3, LINK_DST_FX10_LOW) |
                         LINK_DST(1, 2, LINK_DST_FX10_HIGH) |
                         LINK_DST(2, 2, LINK_DST_FX10_LOW) |
                         LINK_DST(3, 3, LINK_DST_FX10_HIGH);

   /* the texcoords only fill x, y */
   for (int i = 0; i < info->num_inputs; i++)
      if (info->inputs[i].reg == fp->tex_row)
         info->inputs[i].dst = LINK_DST(0, 0, LINK_DST_FP20) | LINK_DST(1, 1, LINK_DST_FP20);

   fp->tex_mfu->var[2].op = FP_VAR_OP_FX10;
   fp->tex_mfu->var[2].tram_row = fp->tex_row;
   fp->tex_mfu->var[3].op = FP_VAR_OP_FX10;
   fp->tex_mfu->var[3].tram_row = fp->tex_row;

   struct fp_alu_instr_packet *p = CALLOC_STRUCT(fp_alu_instr_packet);
   list_inithead(&p->link);
   for (int i = 0; i < 4; i++) {
      struct fp_alu_dst_operand d = {
         .index = i / 2,
         .write_low_sub_reg = !(i & 1),
         .write_high_sub_reg = i & 1,
      };
      struct fp_alu_src_operand texel = {
         .index = i / 2,
         .datatype = FP_DATATYPE_FIXED10,
         .sub_reg_select_high = i & 1,
      };
      struct fp_alu_src_operand colour = texel;
      colour.index = 2 + i / 2;
      p->slots[i] = (struct fp_alu_instr) {
         .op = FP_ALU_OP_MAD,
         .dst = d,
         .src = { texel, colour, fp_alu_src_zero(), fp_alu_src_one() },
      };
   }
   inst->alu_sched.address = list_length(&fp->alu_instructions);
   inst->alu_sched.num_instructions = 1;
   list_addtail(&p->link, &fp->alu_instructions);

   inst->dw.enable = 1;
   inst->dw.index = 1 + dst->Index;
   inst->dw.src_regs = FP_DW_REGS_R0_R1;
}

static bool
src_is_identity(const struct tgsi_src_register *src)
{
   return src->SwizzleX == TGSI_SWIZZLE_X && src->SwizzleY == TGSI_SWIZZLE_Y &&
          src->SwizzleZ == TGSI_SWIZZLE_Z && src->SwizzleW == TGSI_SWIZZLE_W &&
          !src->Negate && !src->Absolute;
}

static void
emit_tgsi_instr(struct grate_fp_shader *fp, const struct tgsi_full_instruction *inst)
{
   bool saturate = inst->Instruction.Saturate != 0;

   switch (inst->Instruction.Opcode) {
   case TGSI_OPCODE_TEX:
      emit_TEX(fp, inst);
      break;

   case TGSI_OPCODE_MOV:
      if (inst->Dst[0].Register.File == TGSI_FILE_OUTPUT &&
          inst->Src[0].Register.File == TGSI_FILE_TEMPORARY &&
          (int)inst->Src[0].Register.Index == fp->tex_temp &&
          inst->Dst[0].Register.WriteMask == TGSI_WRITEMASK_XYZW &&
          src_is_identity(&inst->Src[0].Register)) {
         emit_store_tex_temp(fp, &inst->Dst[0].Register);
         break;
      }
      emit_vMOV(fp, &inst->Dst[0].Register, saturate,
                    &inst->Src[0].Register);
      break;

   case TGSI_OPCODE_MUL: {
      const struct tgsi_src_register *a = &inst->Src[0].Register;
      const struct tgsi_src_register *b = &inst->Src[1].Register;
      if (a->File == TGSI_FILE_INPUT) {
         const struct tgsi_src_register *t = a;
         a = b;
         b = t;
      }
      if (inst->Dst[0].Register.File == TGSI_FILE_OUTPUT &&
          inst->Dst[0].Register.WriteMask == TGSI_WRITEMASK_XYZW &&
          a->File == TGSI_FILE_TEMPORARY && (int)a->Index == fp->tex_temp &&
          src_is_identity(a) && b->File == TGSI_FILE_INPUT &&
          !b->Negate && !b->Absolute) {
         emit_modulate_tex_temp(fp, &inst->Dst[0].Register, b);
         break;
      }
      UNREACHABLE("unsupported MUL form");
   }

   default:
      UNREACHABLE("unsupported TGSI-opcode!");
   }
}

static struct fp_alu_src_operand
fp_alu_src_fx10(int reg, bool high)
{
   struct fp_alu_src_operand src = {
      .index = reg,
      .datatype = FP_DATATYPE_FIXED10,
      .sub_reg_select_high = high
   };
   return src;
}

/*
 * Colors are four fx10 in a register pair, B, G, R, A:
 * pair.l, pair.h, (pair + 1).l, (pair + 1).h
 */
void
grate_fp_append_blend(struct grate_fp_shader *fp, enum grate_blend_mode mode)
{
   struct fp_instr *store = NULL;

   if (mode == GRATE_BLEND_NONE)
      return;

   list_for_each_entry(struct fp_instr, instr, &fp->fp_instructions, link)
      if (instr->dw.enable)
         store = instr;

   if (!store)
      return;

   bool rgb_sa = mode == GRATE_BLEND_OVER_RGB_SA || mode == GRATE_BLEND_OVER_SA;
   bool a_sa = mode == GRATE_BLEND_OVER_A_SA || mode == GRATE_BLEND_OVER_SA;
   int s = store->dw.src_regs == FP_DW_REGS_R2_R3 ? 2 : 0;
   int d = 2 - s;

   struct fp_instr *inst = CALLOC_STRUCT(fp_instr);
   list_inithead(&inst->link);

   /* fetch the render target pixel into the other register pair */
   inst->pseq = 0x00810008 | (d ? 0x2 : 0x0) | ((store->dw.index & 0xf) << 16);
   inst->mfu_sched.address = list_length(&fp->mfu_instructions);
   inst->alu_sched.address = list_length(&fp->alu_instructions);
   inst->alu_sched.num_instructions = 1;

   /*
    * Row register writes are only visible to the next instruction, so the
    * second step (which reads the scaled dst) gets its own instruction.
    */
   struct fp_instr *inst2 = CALLOC_STRUCT(fp_instr);
   list_inithead(&inst2->link);
   inst2->mfu_sched.address = list_length(&fp->mfu_instructions);
   inst2->alu_sched.address = inst->alu_sched.address + 1;
   inst2->alu_sched.num_instructions = 1;

   struct fp_alu_instr_packet *p0 = CALLOC_STRUCT(fp_alu_instr_packet);
   struct fp_alu_instr_packet *p1 = CALLOC_STRUCT(fp_alu_instr_packet);
   list_inithead(&p0->link);
   list_inithead(&p1->link);

   struct fp_alu_src_operand src_a = fp_alu_src_fx10(s + 1, true);

   for (int i = 0; i < 4; i++) {
      int reg = i / 2;
      bool high = i & 1;
      bool sa = i == 3 ? a_sa : rgb_sa;
      struct fp_alu_src_operand dc = fp_alu_src_fx10(d + reg, high);
      struct fp_alu_src_operand sc = fp_alu_src_fx10(s + reg, high);
      struct fp_alu_src_operand neg_a = src_a;
      neg_a.negate = true;

      /* dst.c = dst.c - src.a * dst.c */
      p0->slots[i] = (struct fp_alu_instr) {
         .op = FP_ALU_OP_MAD,
         .dst = {
            .index = d + reg,
            .write_low_sub_reg = !high,
            .write_high_sub_reg = high,
         },
         .src = { neg_a, dc, dc, fp_alu_src_one() },
      };

      /* src.c = src.c * (1 or src.a) + dst.c */
      p1->slots[i] = (struct fp_alu_instr) {
         .op = FP_ALU_OP_MAD,
         .dst = {
            .index = s + reg,
            .write_low_sub_reg = !high,
            .write_high_sub_reg = high,
            .saturate = true,
         },
         .src = { sc, sa ? src_a : fp_alu_src_one(), dc, fp_alu_src_one() },
      };
   }

   list_addtail(&p0->link, &fp->alu_instructions);
   list_addtail(&p1->link, &fp->alu_instructions);

   inst2->dw = store->dw;
   store->dw.enable = false;

   list_addtail(&inst->link, &fp->fp_instructions);
   list_addtail(&inst2->link, &fp->fp_instructions);
}


static void
emit_tgsi_input(struct grate_fp_shader *fp, const struct tgsi_full_declaration *decl)
{
   assert(decl->Range.First == decl->Range.Last);

   /* export 0 is the position, varyings follow in declaration order */
   uint32_t src = LINK_SRC(1 + fp->info.num_inputs);
   /* TRAM row the input is read from (bits 9..14), see emit_vMOV/emit_TEX */
   src |= decl->Range.First << 9;
   uint32_t dst = 0;
   for (int i = 0; i < 4; ++i)
      dst |= LINK_DST(i, i, LINK_DST_FP20);

   fp->info.inputs[fp->info.num_inputs].src = src;
   fp->info.inputs[fp->info.num_inputs].reg = decl->Range.First;
   fp->info.inputs[fp->info.num_inputs].semantic =
      decl->Declaration.Semantic ? decl->Semantic.Name << 8 | decl->Semantic.Index : 0;
   fp->info.inputs[fp->info.num_inputs].dst = dst;

   if (decl->Declaration.Semantic == TGSI_SEMANTIC_COLOR)
      fp->info.color_input = decl->Range.First;

   fp->info.num_inputs++;
}

static void
emit_tgsi_declaration(struct grate_fp_shader *fp, const struct tgsi_full_declaration *decl)
{
   switch (decl->Declaration.File) {
   case TGSI_FILE_INPUT:
      emit_tgsi_input(fp, decl);
      break;
   }
}

void
grate_tgsi_to_fp(struct grate_fp_shader *fp, struct tgsi_parse_context *tgsi)
{
   list_inithead(&fp->fp_instructions);
   list_inithead(&fp->alu_instructions);
   list_inithead(&fp->mfu_instructions);

   fp->tex_temp = -1;
   fp->alu_buf_size = 1;
   fp->info.num_inputs = 0;
   fp->info.color_input = -1;
   fp->info.max_tram_row = 1;

   while (!tgsi_parse_end_of_tokens(tgsi)) {
      tgsi_parse_token(tgsi);
      switch (tgsi->FullToken.Token.Type) {
      case TGSI_TOKEN_TYPE_DECLARATION:
         emit_tgsi_declaration(fp, &tgsi->FullToken.FullDeclaration);
         break;

      case TGSI_TOKEN_TYPE_INSTRUCTION:
         if (tgsi->FullToken.FullInstruction.Instruction.Opcode != TGSI_OPCODE_END)
            emit_tgsi_instr(fp, &tgsi->FullToken.FullInstruction);
         break;
      }
   }

   /*
    * HACK: insert barycentric interpolation setup
    * This will overwrite instructions in some cases, need proper scheduler
    * to fix properly
    */
   /*
    * Every MFU instruction that interpolates needs the weights set up in
    * its own batch (they don't carry over to later instructions).
    */
   list_for_each_entry(struct fp_mfu_instr, first, &fp->mfu_instructions, link) {
    if (first != list_first_entry(&fp->mfu_instructions, struct fp_mfu_instr, link) &&
        first->var[0].op == FP_VAR_OP_NOP && first->var[1].op == FP_VAR_OP_NOP)
       continue;
    first->sfu.op = FP_SFU_OP_RCP;
    first->sfu.reg = 4;
    first->mul[0].dst = FP_MFU_MUL_DST_BARYCENTRIC_WEIGHT;
    first->mul[0].src[0] = FP_MFU_MUL_SRC_SFU_RESULT;
    first->mul[0].src[1] = FP_MFU_MUL_SRC_BARYCENTRIC_COEF_0;

    first->mul[1].dst = FP_MFU_MUL_DST_BARYCENTRIC_WEIGHT;
    first->mul[1].src[0] = FP_MFU_MUL_SRC_SFU_RESULT;
    first->mul[1].src[1] = FP_MFU_MUL_SRC_BARYCENTRIC_COEF_1;
   }
}
