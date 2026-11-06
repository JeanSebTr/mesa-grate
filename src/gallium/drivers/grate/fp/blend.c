/*
 * Blending in the fragment program: GR3D has no blend unit, so the program
 * reads the render target pixel (PSEQ) and does src-over itself.
 */

#include "util/list.h"
#include "util/u_memory.h"

#include "grate_compiler.h"
#include "fpir.h"

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


