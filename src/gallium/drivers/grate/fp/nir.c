/*
 * NIR -> GR3D fragment program.
 *
 * First version: correctness over speed. The shader is lowered to scalar
 * SSA in one block, then every operation gets its own EXEC batch (one ALU
 * packet with one live slot, or one MFU instruction for special functions),
 * so a row-register write is always visible to the next user. Batch 0 does
 * all the interpolation (later batches can't interpolate reliably); a TEX
 * batch reads its coordinates from a register pair and writes the texel as
 * four fx10 to a pair; the last batch writes the colour from R0-R1 or R2-R3.
 *
 * Values live in fp20 row registers, or as fx10 halves of one (texels,
 * colour varyings), and are allocated by live range over the batches.
 * Uniforms and immediates go to the 32 uniform registers (32..63) through
 * a table the draw code uploads (grate_fp_info.uniforms).
 */

#include <stdio.h>

#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_schedule.h"
#include "tgsi/tgsi_from_mesa.h"
#include "util/bitset.h"
#include "util/u_math.h"
#include "util/u_debug.h"
#include "util/u_memory.h"

#include "grate_compiler.h"
#include "fpir.h"

#define LINK_SRC(index) ((index) << 3)
#define LINK_DST(index, comp, type) (((comp) | (type) << 2) << ((index) * 4))
#define LINK_DST_FX10_LOW  1
#define LINK_DST_FX10_HIGH 2
#define LINK_DST_FP20      3

#define REG_LOWP_0_1 31
/*
 * Registers 24-27 (ALU results) are not used: they aren't per fragment, so
 * reading one gives a neighbouring pixel's value.
 */
#define REG_EMBEDDED 28
#define REG_UNIFORM 32
#define REG_POS_X 72
#define REG_POS_Y 73
#define MAX_ROW_REGS 16

/* where a scalar lives */
enum val_kind {
   VAL_NONE,
   VAL_ROW,      /* row register, fp20 or one fx10 half */
   VAL_UNIFORM,  /* uniform register 32 + index */
   VAL_FIXED,    /* a fixed source register (fragment position) */
   VAL_VARYING,  /* not interpolated yet: input index, component in .slot */
   VAL_SPILLED,  /* only in a spill slot (see .spill), reload before use */
};

struct val {
   enum val_kind kind;
   int index;   /* row register, uniform index or register number */
   bool fx10;
   bool high;
   int def;     /* batch that writes it */
   int last;    /* last batch that reads it */
   /* ALU packet slot that computed it, readable as an ALU result register
    * by the later slots of that packet */
   struct fp_alu_instr_packet *pkt;
   int slot;
   /* uniform table key of a VAL_UNIFORM (the index is per segment) */
   uint32_t ukey;
   /* spill slot holding a copy (slot + 1, 0: none) and its place there */
   int spill;
   int spill_pos;
};

/*
 * Spill slots are render targets 4 and up, one per slot. DW writes a row
 * register pair to the slot's surface at the fragment's position and a
 * later PSEQ reads it back into a pair. A fp16 slot (render target format
 * 0x11, two fp16) keeps the pair's two fp20 values at fp16 precision; an
 * RGBA8 slot keeps its four fx10 halves (texels, colours) at 8 bits.
 */
#define SPILL_RT_FIRST 4
#define MAX_SPILL_SLOTS GRATE_FP_MAX_SPILL_SLOTS

struct comp {
   struct grate_fp_shader *fp;
   nir_shader *nir;
   const char *error;

   /* per SSA def, NIR_MAX_VEC_COMPONENTS scalars */
   struct val *vals;
   unsigned num_defs;

   int num_batches;
   /* row register occupancy for the allocator: batch it's free from */
   int reg_busy_until[MAX_ROW_REGS];
   int max_reg;

   /* the ALU packet still taking slots, its batch, and the row registers
    * its slots write (visible only from the next batch) */
   struct fp_alu_instr_packet *pkt;
   int pkt_slots;
   uint32_t pkt_rows_written;
   uint32_t pkt_halves_written;
   struct fp_alu_src_operand pkt_slot_row[4];

   /* TRAM row of each input, and the components used in each row */
   int input_row[16];
   int input_reg[16][4];
   int input_loc[16];
   unsigned tram_used[16];
   bool store_packet;
   struct fp_instr *cur;         /* last batch */
   struct fp_mfu_instr *cur_ipl; /* its MFU, if it only interpolates */

   /* spilling: allowed at all, the def of each SSA index, the instruction
    * being emitted, values that must stay in registers, slot formats */
   bool allow_spill;
   nir_def **defs;
   unsigned cur_index;
   struct val *protect[12];
   int nprotect;
   int num_slots;
   bool slot_rgba8[MAX_SPILL_SLOTS];
   /* last batch that reads (PSEQ) or writes (DW) each slot */
   int slot_last[MAX_SPILL_SLOTS];
   /* ensure() interpolates dropped varyings (not for TEX coordinates,
    * which emit_tex() interpolates straight into the pair it reads) */
   bool ensure_varyings;
   /* row registers R0-R3 some stage of the last batch reads or writes */
   unsigned touched;
};

static struct val *
def_val(struct comp *c, nir_def *def, unsigned comp)
{
   return &c->vals[def->index * NIR_MAX_VEC_COMPONENTS + comp];
}

/* ops the MFU computes: their source is a row register, no modifiers */
static bool
is_sfu_op(nir_op op)
{
   return op == nir_op_frcp || op == nir_op_frsq || op == nir_op_fsqrt ||
          op == nir_op_flog2 || op == nir_op_fexp2 || op == nir_op_ffract;
}

static bool
is_copy(nir_instr *instr)
{
   if (instr->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(instr);
   if (nir_op_is_vec_or_mov(alu->op))
      return true;
   /* fneg/fabs whose users are all plain ALU ops become source modifiers */
   if (alu->op != nir_op_fneg && alu->op != nir_op_fabs)
      return false;
   nir_foreach_use_including_if(u, &alu->def) {
      if (nir_src_is_if(u))
         return false;
      nir_instr *i = nir_src_parent_instr(u);
      if (i->type != nir_instr_type_alu ||
          nir_op_is_vec(nir_instr_as_alu(i)->op) ||
          is_sfu_op(nir_instr_as_alu(i)->op))
         return false;
   }
   return true;
}

/* through vecN, mov and folded fneg/fabs, collecting the modifiers */
static nir_scalar
chase(nir_scalar s, bool *negate, bool *absolute)
{
   for (;;) {
      s = nir_scalar_chase_movs(s);
      nir_instr *i = nir_def_instr(s.def);
      if (i->type != nir_instr_type_alu || !is_copy(i))
         return s;
      nir_alu_instr *alu = nir_instr_as_alu(i);
      if (alu->op == nir_op_fabs) {
         if (absolute) *absolute = true;
         if (negate) *negate = false;
      } else if (alu->op == nir_op_fneg) {
         if (negate) *negate = !*negate;
      } else {
         return s;
      }
      s = nir_get_scalar(alu->src[0].src.ssa, alu->src[0].swizzle[s.comp]);
   }
}

static nir_scalar
chase_plain(nir_def *def, unsigned comp)
{
   return chase(nir_get_scalar(def, comp), NULL, NULL);
}

/* vecN and mov are free: look through them to the scalar that holds it */
static struct val *
src_val(struct comp *c, nir_src src, unsigned comp)
{
   nir_scalar s = chase_plain(src.ssa, comp);
   return def_val(c, s.def, s.comp);
}

/* last instruction reading def, through vecN/mov copies */
static unsigned
last_use(nir_def *def)
{
   unsigned last = 0;
   nir_foreach_use_including_if(u, def) {
      if (nir_src_is_if(u))
         return UINT_MAX;
      nir_instr *i = nir_src_parent_instr(u);
      if (is_copy(i))
         last = MAX2(last, last_use(&nir_instr_as_alu(i)->def));
      else
         last = MAX2(last, i->index);
   }
   return last;
}

/* last instruction reading component comp of def, through vecN/mov copies */
static unsigned
last_use_comp(nir_def *def, unsigned comp)
{
   unsigned last = 0;
   nir_foreach_use_including_if(u, def) {
      if (nir_src_is_if(u))
         return UINT_MAX;
      nir_instr *i = nir_src_parent_instr(u);
      if (i->type != nir_instr_type_alu) {
         last = MAX2(last, i->index);
         continue;
      }
      nir_alu_instr *alu = nir_instr_as_alu(i);
      unsigned s = container_of(u, nir_alu_src, src) - alu->src;
      if (is_copy(i)) {
         /* which components of the copy come from comp */
         unsigned n = nir_op_is_vec(alu->op) ? 1 : alu->def.num_components;
         for (unsigned k = 0; k < n; k++)
            if (alu->src[s].swizzle[k] == comp)
               last = MAX2(last, last_use_comp(&alu->def,
                                               nir_op_is_vec(alu->op) ? s : k));
      } else {
         for (unsigned k = 0; k < nir_ssa_alu_instr_src_components(alu, s); k++)
            if (alu->src[s].swizzle[k] == comp)
               last = MAX2(last, i->index);
      }
   }
   return last;
}

/* first instruction after `after` reading component comp of def */
static unsigned
next_use_comp(nir_def *def, unsigned comp, unsigned after)
{
   unsigned next = UINT_MAX;
   nir_foreach_use_including_if(u, def) {
      if (nir_src_is_if(u))
         continue;
      nir_instr *i = nir_src_parent_instr(u);
      if (i->type != nir_instr_type_alu) {
         if (i->index > after)
            next = MIN2(next, i->index);
         continue;
      }
      nir_alu_instr *alu = nir_instr_as_alu(i);
      unsigned s = container_of(u, nir_alu_src, src) - alu->src;
      if (is_copy(i)) {
         unsigned n = nir_op_is_vec(alu->op) ? 1 : alu->def.num_components;
         for (unsigned k = 0; k < n; k++)
            if (alu->src[s].swizzle[k] == comp)
               next = MIN2(next, next_use_comp(&alu->def,
                                               nir_op_is_vec(alu->op) ? s : k,
                                               after));
      } else if (i->index > after) {
         for (unsigned k = 0; k < nir_ssa_alu_instr_src_components(alu, s); k++)
            if (alu->src[s].swizzle[k] == comp)
               next = MIN2(next, i->index);
      }
   }
   return next;
}

static void
fail(struct comp *c, const char *msg)
{
   if (!c->error)
      c->error = msg;
}

/* ------------------------------------------------------------------ */
/* uniform table */

static int
uniform_slot(struct comp *c, uint32_t key)
{
   struct grate_fp_info *info = &c->fp->info;
   for (int i = 0; i < info->num_uniforms; i++)
      if (info->uniforms[i] == key)
         return i;
   if (info->num_uniforms >= (int)ARRAY_SIZE(info->uniforms)) {
      fail(c, "too many fragment uniforms and immediates");
      return 0;
   }
   info->uniforms[info->num_uniforms] = key;
   return info->num_uniforms++;
}

static struct val
uniform_val(struct comp *c, uint32_t key)
{
   return (struct val) {
      .kind = VAL_UNIFORM,
      .index = uniform_slot(c, key),
      .ukey = key,
      .def = -1,
   };
}

static struct val
imm_val(struct comp *c, float f)
{
   return uniform_val(c, GRATE_FP_UNIFORM_IMM(f));
}

/* ------------------------------------------------------------------ */
/* batches */

static struct fp_instr *
new_batch(struct comp *c)
{
   struct grate_fp_shader *fp = c->fp;
   struct fp_instr *inst = CALLOC_STRUCT(fp_instr);
   list_inithead(&inst->link);
   inst->mfu_sched.address = list_length(&fp->mfu_instructions);
   inst->alu_sched.address = list_length(&fp->alu_instructions);
   list_addtail(&inst->link, &fp->fp_instructions);
   c->num_batches++;
   c->pkt = NULL;
   c->cur = inst;
   c->cur_ipl = NULL;
   c->touched = 0;
   return inst;
}

/*
 * A batch runs its stages in order: MFU, TEX, ALU, DW, and each stage sees
 * what the earlier ones wrote. Reuse the current batch for a stage that
 * comes after everything it already does.
 */
static struct fp_instr *
batch_for(struct comp *c, int stage)
{
   struct fp_instr *b = c->cur;
   if (b) {
      int last = b->dw.enable ? 4 : b->alu_sched.num_instructions ? 3 :
                 b->tex.enable ? 2 : b->mfu_sched.num_instructions ? 1 : 0;
      /* MFU results reach TEX and ALU only for interpolation */
      bool sfu = b->mfu_sched.num_instructions && !c->cur_ipl;
      if (stage > last && !(sfu && stage < 4))
         return b;
   }
   return new_batch(c);
}

static struct fp_alu_instr
alu_nop(void)
{
   struct fp_alu_src_operand lowp = {
      .index = REG_LOWP_0_1,
      .datatype = FP_DATATYPE_FIXED10,
   };
   struct fp_alu_src_operand one = lowp;
   one.sub_reg_select_high = true;
   /* grate-tools' NOP: r31 = 1.0 * 0.0 + 0.0, both halves written */
   return (struct fp_alu_instr) {
      .op = FP_ALU_OP_MAD,
      .dst = { .index = REG_LOWP_0_1, .write_low_sub_reg = true,
               .write_high_sub_reg = true },
      .src = { one, lowp, lowp, lowp },
   };
}

static struct fp_alu_src_operand
operand(struct comp *c, const struct val *v)
{
   struct fp_alu_src_operand s = { 0 };
   switch (v->kind) {
   case VAL_ROW:
      s.index = v->index;
      s.datatype = v->fx10 ? FP_DATATYPE_FIXED10 : FP_DATATYPE_FP20;
      s.sub_reg_select_high = v->high;
      break;
   case VAL_UNIFORM:
      /* a uniform loaded in an earlier segment: this segment's table */
      s.index = REG_UNIFORM + uniform_slot(c, v->ukey);
      break;
   case VAL_FIXED:
      s.index = v->index;
      break;
   default:
      s.index = REG_LOWP_0_1;
      s.datatype = FP_DATATYPE_FIXED10;
      break;
   }
   return s;
}

static struct fp_alu_src_operand
lowp(bool one)
{
   return (struct fp_alu_src_operand) {
      .index = REG_LOWP_0_1,
      .datatype = FP_DATATYPE_FIXED10,
      .sub_reg_select_high = one,
   };
}


/* ------------------------------------------------------------------ */
/* register allocation: live ranges are [def + 1, last] in batches */

static void
note_use(struct comp *c, struct val *v)
{
   if (v->kind == VAL_ROW)
      v->last = MAX2(v->last, c->num_batches);
}

/* a row register free over [from, ...) given current knowledge; uses are
 * recorded as the program is built, so a value's range is final once all its
 * users are emitted. Ranges are tracked lazily: a register is reusable once
 * every value in it has seen its last use, which the caller guarantees by
 * calling release() when the SSA def is dead (all uses emitted). */
static bool spill_any(struct comp *c);

static int
alloc_reg(struct comp *c, int first, int last_allowed)
{
   /* without spilling, allocate past R3 and fail at the end (compile()) */
   if (c->allow_spill)
      last_allowed = MIN2(last_allowed, 3);
   do {
      for (int r = first; r <= last_allowed && r < MAX_ROW_REGS; r++) {
         if (c->reg_busy_until[r] <= c->num_batches) {
            c->reg_busy_until[r] = INT_MAX;
            c->max_reg = MAX2(c->max_reg, r);
            return r;
         }
      }
   } while (c->allow_spill && !c->error && spill_any(c));
   fail(c, "out of row registers (1)");
   return 0;
}

static void
release(struct comp *c, struct val *v)
{
   if (v->kind == VAL_ROW && !v->fx10)
      c->reg_busy_until[v->index] = v->last;
}

/* colour output component -> fx10 half of R2-R3 (B G R A = 2.l 2.h 3.l 3.h) */
static const int out_reg[4] = { 3, 2, 2, 3 };
static const bool out_high[4] = { false, true, false, true };

static bool
at_output(const struct val *v, unsigned cc)
{
   return v->kind == VAL_ROW && v->fx10 && v->index == out_reg[cc] &&
          v->high == out_high[cc];
}

static void emit_mad(struct comp *c, struct val *dst, enum fp_alu_op op,
                     struct fp_alu_src_operand a, struct fp_alu_src_operand b,
                     struct fp_alu_src_operand cc, bool sat);
static struct fp_alu_src_operand operand(struct comp *c, const struct val *v);
static struct fp_alu_src_operand lowp(bool one);

static int64_t reg_score(struct comp *c, int r, bool *fx10);
static bool val_varying(struct comp *c, struct val *v, unsigned *loc,
                        unsigned *comp);
static bool spill_regs(struct comp *c, int lo, unsigned mask);

/*
 * Move every live value out of row registers [lo, hi] (TEX and DW need
 * fixed registers), into fp20 registers elsewhere.
 */
static void
evict(struct comp *c, int lo, int hi, struct val *const *keep, int nkeep)
{
   for (unsigned i = 0; i < c->num_defs * NIR_MAX_VEC_COMPONENTS; i++) {
      struct val *v = &c->vals[i];
      if (v->kind != VAL_ROW || v->index < lo || v->index > hi)
         continue;
      bool kept = false;
      for (int k = 0; k < nkeep; k++)
         kept |= keep[k] == v;
      if (kept)
         continue;
      /* a varying is interpolated again when needed: just drop it (an op
       * of this batch may still read the register) */
      unsigned loc, comp;
      if (c->allow_spill && val_varying(c, v, &loc, &comp)) {
         *v = (struct val) { .kind = VAL_VARYING, .index = loc, .slot = comp };
         continue;
      }
      int r = -1;
      for (int k = 0; k < (c->allow_spill ? 4 : MAX_ROW_REGS); k++) {
         if ((k < lo || k > hi) && c->reg_busy_until[k] <= c->num_batches) {
            r = k;
            break;
         }
      }
      if (r < 0 && c->allow_spill) {
         /* nowhere to move it: spill its register instead */
         bool fx10;
         if (reg_score(c, v->index, &fx10) >= 0 &&
             spill_regs(c, v->index & ~1, 1u << (v->index & 1)))
            continue;
      }
      if (r < 0) {
         fail(c, "out of row registers (2)");
         return;
      }
      c->reg_busy_until[r] = INT_MAX;
      c->max_reg = MAX2(c->max_reg, r);
      struct val moved = { .kind = VAL_ROW, .index = r,
                           .def = c->num_batches };
      emit_mad(c, &moved, FP_ALU_OP_MAD, operand(c, v), lowp(true), lowp(false),
               false);
      moved.last = c->num_batches;
      *v = moved;
   }
   for (int k = lo; k <= hi; k++)
      c->reg_busy_until[k] = MIN2(c->reg_busy_until[k], c->num_batches);
}


/* ------------------------------------------------------------------ */
/* spilling */

static bool
is_protected(struct comp *c, const struct val *v)
{
   for (int i = 0; i < c->nprotect; i++)
      if (c->protect[i] == v)
         return true;
   return false;
}

static void
protect(struct comp *c, struct val *v)
{
   if (c->nprotect < (int)ARRAY_SIZE(c->protect))
      c->protect[c->nprotect++] = v;
}

/* next instruction reading the value, from the current one on */
static unsigned
val_next_use(struct comp *c, struct val *v)
{
   unsigned i = v - c->vals;
   nir_def *def = c->defs[i / NIR_MAX_VEC_COMPONENTS];
   if (!def)
      return UINT_MAX;
   return next_use_comp(def, i % NIR_MAX_VEC_COMPONENTS,
                        c->cur_index ? c->cur_index - 1 : 0);
}

/* a plain fp20 varying component: interpolated again rather than spilled */
static bool
val_varying(struct comp *c, struct val *v, unsigned *loc, unsigned *comp)
{
   unsigned i = v - c->vals;
   nir_def *def = c->defs[i / NIR_MAX_VEC_COMPONENTS];
   if (!def || v->fx10)
      return false;
   nir_instr *instr = nir_def_instr(def);
   if (instr->type != nir_instr_type_intrinsic)
      return false;
   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   if (intr->intrinsic != nir_intrinsic_load_input &&
       intr->intrinsic != nir_intrinsic_load_interpolated_input)
      return false;
   if (!nir_src_is_const(intr->src[intr->intrinsic ==
                         nir_intrinsic_load_input ? 0 : 1]))
      return false;
   *loc = nir_intrinsic_io_semantics(intr).location;
   *comp = nir_intrinsic_component(intr) + i % NIR_MAX_VEC_COMPONENTS;
   return true;
}

static bool
slot_in_use(struct comp *c, int slot)
{
   for (unsigned i = 0; i < c->num_defs * NIR_MAX_VEC_COMPONENTS; i++) {
      struct val *v = &c->vals[i];
      if ((v->kind == VAL_ROW || v->kind == VAL_SPILLED) && v->spill == slot + 1)
         return true;
   }
   return false;
}

#define SCORE_FREE   INT64_MAX           /* nothing live, not reserved */
#define SCORE_DROP   ((int64_t)UINT_MAX + 2) /* only varyings: just drop */
#define SCORE_COPIED ((int64_t)UINT_MAX + 1) /* everything has a slot copy */

/*
 * How good a victim register r is: -1 can't be spilled, otherwise the
 * nearest next use of what it holds (farther is better), or one of the
 * SCORE_* values. *fx10 tells the kind of what it holds.
 */
static int64_t
reg_score(struct comp *c, int r, bool *fx10)
{
   bool live = false, copied = true, store = false;
   unsigned next = UINT_MAX;
   *fx10 = false;
   for (unsigned i = 0; i < c->num_defs * NIR_MAX_VEC_COMPONENTS; i++) {
      struct val *v = &c->vals[i];
      if (v->kind != VAL_ROW || v->index != r)
         continue;
      if (is_protected(c, v))
         return -1;
      live = true;
      unsigned loc, comp;
      if (val_varying(c, v, &loc, &comp))
         continue;
      store = true;
      *fx10 = v->fx10;
      copied &= v->spill != 0;
      next = MIN2(next, val_next_use(c, v));
   }
   if (!live)
      /* a busy register with nothing live in it is reserved by the caller */
      return c->reg_busy_until[r] <= c->num_batches ? SCORE_FREE : -1;
   if (!store)
      return SCORE_DROP;
   if (copied)
      return SCORE_COPIED;
   return next;
}

/*
 * Spill registers lo and/or lo + 1 (mask bits 0, 1): varyings are dropped,
 * values with a slot copy too, the rest DW'd into a slot (one DW if both
 * registers hold the same kind). The registers are free from the next batch.
 */
static bool
spill_regs(struct comp *c, int lo, unsigned mask)
{
   bool fx10[2] = { false, false }, store[2] = { false, false };
   for (unsigned i = 0; i < c->num_defs * NIR_MAX_VEC_COMPONENTS; i++) {
      struct val *v = &c->vals[i];
      unsigned loc, comp;
      if (v->kind != VAL_ROW || v->index < lo || v->index > lo + 1 ||
          !(mask & (1u << (v->index - lo))))
         continue;
      if (val_varying(c, v, &loc, &comp) || v->spill)
         continue;
      store[v->index - lo] = true;
      fx10[v->index - lo] = v->fx10;
   }

   int slot_of[2] = { -1, -1 };
   for (int h = 0; h < 2; h++) {
      if (!store[h])
         continue;
      if (h == 1 && store[0] && fx10[0] == fx10[1]) {
         slot_of[1] = slot_of[0];
         continue;
      }
      int slot = -1;
      for (int k = 0; k < c->num_slots && slot < 0; k++)
         if (c->slot_rgba8[k] == fx10[h] && !slot_in_use(c, k) &&
             k != slot_of[0])
            slot = k;
      if (slot < 0) {
         if (c->num_slots == MAX_SPILL_SLOTS) {
            fail(c, "out of spill slots");
            return false;
         }
         slot = c->num_slots++;
         c->slot_rgba8[slot] = fx10[h];
      }
      slot_of[h] = slot;
      /*
       * The DW goes in the latest batch before this one that has none and
       * comes after the values were written (they've sat in the registers
       * since): the registers are then untouched here, and a reload can
       * share this batch. Failing that, this batch.
       */
      /* and not before the slot's earlier contents were read back */
      int first = c->slot_last[slot];
      for (unsigned i = 0; i < c->num_defs * NIR_MAX_VEC_COMPONENTS; i++) {
         struct val *v = &c->vals[i];
         if (v->kind == VAL_ROW && v->index >= lo && v->index <= lo + 1 &&
             (mask & (1u << (v->index - lo))))
            first = MAX2(first, v->def);
      }
      struct fp_instr *b = NULL;
      int bi = 0;
      list_for_each_entry(struct fp_instr, it, &c->fp->fp_instructions, link) {
         if (bi >= first && bi < c->num_batches - 1 && !it->dw.enable)
            b = it;
         bi++;
      }
      if (!b) {
         b = batch_for(c, 4);
         c->touched |= 3u << lo;
         /* later ALU ops write the freed registers: not in this batch */
         c->pkt = NULL;
      }
      b->dw.enable = true;
      b->dw.index = SPILL_RT_FIRST + slot;
      b->dw.src_regs = lo ? FP_DW_REGS_R2_R3 : FP_DW_REGS_R0_R1;
      {
         int idx = 0;
         list_for_each_entry(struct fp_instr, it, &c->fp->fp_instructions, link) {
            if (it == b)
               break;
            idx++;
         }
         c->slot_last[slot] = idx + 1; /* a later DW: a later batch */
      }
   }

   for (unsigned i = 0; i < c->num_defs * NIR_MAX_VEC_COMPONENTS; i++) {
      struct val *v = &c->vals[i];
      if (v->kind != VAL_ROW || v->index < lo || v->index > lo + 1 ||
          !(mask & (1u << (v->index - lo))))
         continue;
      unsigned loc, comp;
      if (val_varying(c, v, &loc, &comp)) {
         *v = (struct val) { .kind = VAL_VARYING, .index = loc, .slot = comp };
         continue;
      }
      if (!v->spill) {
         v->spill = slot_of[v->index - lo] + 1;
         v->spill_pos = v->fx10 ? (v->index - lo) * 2 + v->high : v->index - lo;
      }
      v->kind = VAL_SPILLED;
   }
   for (int h = 0; h < 2; h++)
      if (mask & (1u << h))
         c->reg_busy_until[lo + h] = MIN2(c->reg_busy_until[lo + h],
                                          c->num_batches);
   return true;
}

/* spill the best victim register; false if none can go */
static bool
spill_any(struct comp *c)
{
   int best = -1;
   int64_t best_score = -1;
   for (int r = 0; r < 4; r++) {
      bool fx10;
      int64_t sc = reg_score(c, r, &fx10);
      if (sc != SCORE_FREE && sc > best_score) {
         best = r;
         best_score = sc;
      }
   }
   if (best < 0)
      return false;
   /*
    * Take the other register of the pair along when it holds the same kind
    * of value and isn't needed right away: one DW, and the two come back
    * together (a blur's running sums, say).
    */
   unsigned mask = 1u << (best & 1);
   int other = best ^ 1;
   bool fb, fo;
   reg_score(c, best, &fb);
   int64_t so = reg_score(c, other, &fo);
   if (so >= 0 && so != SCORE_FREE && fo == fb &&
       (so >= SCORE_COPIED || so > (int64_t)c->cur_index + 1))
      mask |= 1u << (other & 1);
   return spill_regs(c, best & ~1, mask);
}

static bool
pair_free(struct comp *c, int lo)
{
   return c->reg_busy_until[lo] <= c->num_batches &&
          c->reg_busy_until[lo + 1] <= c->num_batches;
}

/*
 * Values that must stay (an op's sources) can sit in both pairs, so neither
 * can be spilled whole: move them into the other pair's free registers
 * (spilling what's there if need be) to empty one pair. Returns that pair,
 * its registers holding only spillable values now, or -1.
 */
static int
compact_pair(struct comp *c)
{
   for (int q = 0; q <= 2; q += 2) {
      int o = 2 - q;
      int need = 0, room = 0;
      bool ok = true;
      for (int h = 0; h < 2; h++) {
         bool f;
         int64_t sq = reg_score(c, q + h, &f), so = reg_score(c, o + h, &f);
         if (sq < 0) {
            /* protected (or reserved): only a protected fp20 value moves */
            bool movable = false;
            for (unsigned i = 0; i < c->num_defs * NIR_MAX_VEC_COMPONENTS; i++) {
               struct val *v = &c->vals[i];
               if (v->kind == VAL_ROW && v->index == q + h)
                  movable = !v->fx10;
            }
            ok &= movable;
            need++;
         }
         if (so >= 0)
            room++;
      }
      if (!ok || need > room)
         continue;
      for (int h = 0; h < 2; h++) {
         bool f;
         if (reg_score(c, q + h, &f) >= 0)
            continue;
         int r = -1;
         for (int k = 0; k < 2 && r < 0; k++) {
            int64_t so = reg_score(c, o + k, &f);
            if (so == SCORE_FREE)
               r = o + k;
         }
         for (int k = 0; k < 2 && r < 0; k++) {
            if (reg_score(c, o + k, &f) >= 0 &&
                spill_regs(c, o, 1u << k))
               r = o + k;
         }
         if (r < 0)
            return -1;
         for (unsigned i = 0; i < c->num_defs * NIR_MAX_VEC_COMPONENTS; i++) {
            struct val *v = &c->vals[i];
            if (v->kind != VAL_ROW || v->index != q + h)
               continue;
            c->reg_busy_until[r] = INT_MAX;
            struct val moved = *v;
            moved.index = r;
            moved.def = c->num_batches;
            moved.pkt = NULL;
            emit_mad(c, &moved, FP_ALU_OP_MAD, operand(c, v), lowp(true),
                     lowp(false), false);
            moved.last = c->num_batches;
            *v = moved;
         }
         c->reg_busy_until[q + h] = MIN2(c->reg_busy_until[q + h],
                                         c->num_batches);
      }
      /* the moved values show from the next batch on */
      c->pkt = NULL;
      return q;
   }
   return -1;
}

/* make a whole pair free, spilling the cheaper one; -1 if neither can be */
static int
free_pair(struct comp *c)
{
   int best = -1;
   int64_t best_score = -1;
   for (int lo = 0; lo <= 2; lo += 2) {
      bool f0, f1;
      int64_t s0 = reg_score(c, lo, &f0), s1 = reg_score(c, lo + 1, &f1);
      int64_t sc = MIN2(s0, s1);
      if (sc > best_score) {
         best = lo;
         best_score = sc;
      }
   }
   if (best < 0 && c->allow_spill)
      best = compact_pair(c);
   if (best < 0 || !c->allow_spill)
      return -1;
   if (pair_free(c, best))
      return best;
   unsigned mask = 0;
   for (int h = 0; h < 2; h++)
      if (c->reg_busy_until[best + h] > c->num_batches)
         mask |= 1u << h;
   if (!spill_regs(c, best, mask))
      return -1;
   return best;
}

/* PSEQ a slot back into a free pair, with everything still live in it */
static void
reload(struct comp *c, struct val *v)
{
   int slot = v->spill - 1;
   int q = free_pair(c);
   if (q < 0) {
      fail(c, "out of row registers (3)");
      return;
   }

   /*
    * PSEQ is the first stage: it can join the last batch if that has none
    * and no stage of it touches the pair (PSEQ overwrites it before they
    * run); otherwise a fresh batch.
    */
   struct fp_instr *b = c->cur;
   if (!b || b->pseq || (c->touched & (3u << q)))
      b = new_batch(c);
   c->touched |= 3u << q;
   b->pseq = 0x00800008 | (q ? 0x2 : 0x0) | (SPILL_RT_FIRST + slot) << 16;
   /* a DW in this batch runs after the PSEQ: fine */
   c->slot_last[slot] = MAX2(c->slot_last[slot], c->num_batches - 1);
   c->max_reg = MAX2(c->max_reg, q + 1);

   bool rgba8 = c->slot_rgba8[slot];
   for (unsigned i = 0; i < c->num_defs * NIR_MAX_VEC_COMPONENTS; i++) {
      struct val *w = &c->vals[i];
      if (w->kind != VAL_SPILLED || w->spill != slot + 1)
         continue;
      w->kind = VAL_ROW;
      w->index = q + (rgba8 ? w->spill_pos / 2 : w->spill_pos);
      w->fx10 = rgba8;
      w->high = rgba8 && (w->spill_pos & 1);
      w->def = c->num_batches - 1;
      w->last = c->num_batches;
      w->pkt = NULL;
      c->reg_busy_until[w->index] = INT_MAX;
   }
}

/* at least n of R0-R3 free, spilling if need be */
static void
make_free(struct comp *c, int n)
{
   for (;;) {
      int nfree = 0;
      for (int r = 0; r < 4; r++)
         nfree += c->reg_busy_until[r] <= c->num_batches;
      if (nfree >= n || !c->allow_spill || !spill_any(c))
         return;
   }
}

/* a value about to be read: back into a register if it was spilled */
static int place(struct comp *c, unsigned location, int n, const int *j,
                 const int *want, int *reg);
static void emit_interpolation(struct comp *c, int e, int n, const int *reg);

static void
ensure(struct comp *c, struct val *v)
{
   if (v->kind == VAL_SPILLED)
      reload(c, v);
   if (v->kind == VAL_VARYING && c->ensure_varyings) {
      int j = v->slot, want = -1, reg;
      make_free(c, 1);
      int e = place(c, v->index, 1, &j, &want, &reg);
      if (e < 0)
         return;
      emit_interpolation(c, e, 1, &reg);
      *v = (struct val) { .kind = VAL_ROW, .index = reg,
                          .def = c->num_batches - 1, .last = c->num_batches };
   }
}

/* every scalar source of instr in registers, and kept there */
static bool
ensure_src(nir_src *src, void *data)
{
   struct comp *c = data;
   /* the components an ALU op reads, or all of them */
   unsigned comps[NIR_MAX_VEC_COMPONENTS], n = 0;
   nir_instr *user = nir_src_parent_instr(src);
   if (user->type == nir_instr_type_alu) {
      nir_alu_instr *alu = nir_instr_as_alu(user);
      unsigned s = container_of(src, nir_alu_src, src) - alu->src;
      for (unsigned k = 0; k < nir_ssa_alu_instr_src_components(alu, s); k++)
         comps[n++] = alu->src[s].swizzle[k];
   } else {
      for (unsigned k = 0; k < src->ssa->num_components; k++)
         comps[n++] = k;
   }
   for (unsigned k = 0; k < n; k++)
      protect(c, src_val(c, *src, comps[k]));
   for (unsigned k = 0; k < n; k++)
      ensure(c, src_val(c, *src, comps[k]));
   return true;
}

/* ------------------------------------------------------------------ */
/* inputs: interpolated in batch 0, var i writes row register i */

struct input_scalar {
   nir_def *def;
   unsigned comp;      /* NIR component */
   unsigned location;  /* varying slot */
   unsigned loc_comp;  /* component within the varying */
   bool fx10;
};

static void
setup_barycentrics(struct fp_mfu_instr *mfu)
{
   mfu->sfu.op = FP_SFU_OP_RCP;
   mfu->sfu.reg = 4;
   mfu->mul[0].dst = FP_MFU_MUL_DST_BARYCENTRIC_WEIGHT;
   mfu->mul[0].src[0] = FP_MFU_MUL_SRC_SFU_RESULT;
   mfu->mul[0].src[1] = FP_MFU_MUL_SRC_BARYCENTRIC_COEF_0;
   mfu->mul[1].dst = FP_MFU_MUL_DST_BARYCENTRIC_WEIGHT;
   mfu->mul[1].src[0] = FP_MFU_MUL_SRC_SFU_RESULT;
   mfu->mul[1].src[1] = FP_MFU_MUL_SRC_BARYCENTRIC_COEF_1;
}

/* as nir_to_tgsi names the vertex shader outputs (no TEXCOORD semantic) */
static void
varying_semantic(unsigned location, unsigned *name, unsigned *index)
{
   if (location >= VARYING_SLOT_VAR0 && location < VARYING_SLOT_PATCH0) {
      *name = TGSI_SEMANTIC_GENERIC;
      *index = location - VARYING_SLOT_VAR0;
      return;
   }
   tgsi_get_gl_varying_semantic(location, true, name, index);
}


/*
 * Varyings. The MFU writes TRAM component k of a row into R<k>, and
 * interpolation works in any batch (with ALU buffer size 1), so a varying is
 * interpolated at each use and holds a register only while needed. Which
 * register a component lands in is fixed by where the linker puts it in
 * TRAM; when that register is busy, the varying gets linked again into a
 * new TRAM row (another linker entry for the same export).
 */

/* a new linker entry, with its own TRAM row, for a varying */
static int
new_entry(struct comp *c, unsigned location)
{
   struct grate_fp_info *info = &c->fp->info;
   unsigned name, index;
   varying_semantic(location, &name, &index);

   int row = 0;
   while (row < 16 && c->tram_used[row])
      row++;
   if (row == 16 || info->num_inputs == 16) {
      fail(c, "out of TRAM");
      return -1;
   }
   c->tram_used[row] = 0xf;

   int e = info->num_inputs++;
   info->inputs[e].semantic = name << 8 | index;
   /* export 0 is the position; draw-time linking fixes up the rank */
   info->inputs[e].src = LINK_SRC(1 + e) | row << 9;
   info->inputs[e].dst = 0;
   info->inputs[e].reg = location;
   if (name == TGSI_SEMANTIC_COLOR && info->color_input < 0)
      info->color_input = e;
   info->max_tram_row = MAX2(info->max_tram_row, row + 1);
   c->input_row[e] = row;
   c->input_loc[e] = location;
   for (int j = 0; j < 4; j++)
      c->input_reg[e][j] = -1;
   return e;
}

/* can component j of entry e land in register r (free, or want) now? */
static int
entry_reg(struct comp *c, int e, int j, int want)
{
   int r = c->input_reg[e][j];
   if (r >= 0)
      return (want < 0 || want == r) &&
             c->reg_busy_until[r] <= c->num_batches ? r : -1;
   for (int t = 0; t < 4; t++) {
      if (want >= 0 && t != want)
         continue;
      bool taken = false;
      for (int m = 0; m < 4; m++)
         taken |= c->input_reg[e][m] == t;
      if (!taken && c->reg_busy_until[t] <= c->num_batches)
         return t;
   }
   return -1;
}

/*
 * Find (or make) a linker entry where components j[0..n-1] of the varying
 * can land in registers want[] (-1: any free one). Fills reg[], returns the
 * entry, or -1.
 */
static int
place(struct comp *c, unsigned location, int n, const int *j, const int *want,
      int *reg)
{
   struct grate_fp_info *info = &c->fp->info;
   for (int e = 0; e <= info->num_inputs; e++) {
      bool fresh = e == info->num_inputs;
      if (fresh) {
         e = new_entry(c, location);
         if (e < 0)
            return -1;
      } else if (c->input_loc[e] != (int)location) {
         continue;
      }
      bool ok = true;
      for (int k = 0; k < n && ok; k++) {
         reg[k] = entry_reg(c, e, j[k], want[k]);
         for (int m = 0; m < k; m++)
            ok &= reg[m] != reg[k];
         ok &= reg[k] >= 0;
      }
      if (!ok) {
         if (fresh)
            break; /* a fresh entry didn't fit: no register */
         continue;
      }
      for (int k = 0; k < n; k++) {
         if (c->input_reg[e][j[k]] < 0) {
            c->input_reg[e][j[k]] = reg[k];
            info->inputs[e].dst |= LINK_DST(j[k], reg[k], LINK_DST_FP20);
         }
      }
      return e;
   }
   if (getenv("GRATE_FP_DUMP_FAILED")) {
      fprintf(stderr, "place: loc %u n %d j0 %d want0 %d batch %d busy %d %d %d %d\n",
              location, n, j[0], want[0], c->num_batches,
              c->reg_busy_until[0], c->reg_busy_until[1],
              c->reg_busy_until[2], c->reg_busy_until[3]);
      for (int e = 0; e < info->num_inputs; e++)
         fprintf(stderr, "  entry %d loc %d row %d regs %d %d %d %d\n", e,
                 c->input_loc[e], c->input_row[e], c->input_reg[e][0],
                 c->input_reg[e][1], c->input_reg[e][2], c->input_reg[e][3]);
   }
   fail(c, "no free register to interpolate into");
   return -1;
}

/* one MFU batch interpolating n fp20 components of entry e into reg[] */
static void
emit_interpolation(struct comp *c, int e, int n, const int *reg)
{
   /*
    * MFU runs before TEX, ALU and DW: interpolation can join the last
    * batch if no stage of it touches the registers it writes.
    */
   unsigned regs = 0;
   for (int k = 0; k < n; k++)
      regs |= 1u << reg[k];
   bool fits = c->cur && !(c->touched & regs);

   /* join the interpolation the current batch already does */
   struct fp_mfu_instr *prev = fits ? c->cur_ipl : NULL;
   for (int k = 0; k < n && prev; k++)
      if (prev->var[reg[k]].op != FP_VAR_OP_NOP)
         prev = NULL;
   if (prev) {
      for (int k = 0; k < n; k++) {
         prev->var[reg[k]].op = FP_VAR_OP_FP20;
         prev->var[reg[k]].tram_row = c->input_row[e];
         c->reg_busy_until[reg[k]] = INT_MAX;
         c->max_reg = MAX2(c->max_reg, reg[k]);
      }
      c->touched |= regs;
      return;
   }
   struct fp_instr *inst = fits && !c->cur->mfu_sched.num_instructions ?
                           c->cur : batch_for(c, 1);
   struct fp_mfu_instr *mfu = CALLOC_STRUCT(fp_mfu_instr);
   list_inithead(&mfu->link);
   setup_barycentrics(mfu);
   c->cur_ipl = mfu;
   for (int k = 0; k < n; k++) {
      mfu->var[reg[k]].op = FP_VAR_OP_FP20;
      mfu->var[reg[k]].tram_row = c->input_row[e];
      c->reg_busy_until[reg[k]] = INT_MAX;
      c->max_reg = MAX2(c->max_reg, reg[k]);
   }
   c->touched |= regs;
   list_addtail(&mfu->link, &c->fp->mfu_instructions);
   inst->mfu_sched.num_instructions = 1;
}

/* every reader is a texture's coordinates (through vecN/mov) */
static bool
only_tex_coords(nir_def *def)
{
   nir_foreach_use_including_if(u, def) {
      if (nir_src_is_if(u))
         return false;
      nir_instr *i = nir_src_parent_instr(u);
      if (i->type == nir_instr_type_alu &&
          nir_op_is_vec_or_mov(nir_instr_as_alu(i)->op)) {
         if (!only_tex_coords(&nir_instr_as_alu(i)->def))
            return false;
      } else if (i->type != nir_instr_type_tex) {
         return false;
      }
   }
   return true;
}

static void
emit_load_input(struct comp *c, nir_intrinsic_instr *intr)
{
   struct grate_fp_info *info = &c->fp->info;

   if (!nir_src_is_const(intr->src[intr->intrinsic ==
                         nir_intrinsic_load_input ? 0 : 1])) {
      fail(c, "indirect varying");
      return;
   }

   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   unsigned first = nir_intrinsic_component(intr);
   unsigned name, index;
   varying_semantic(sem.location, &name, &index);

   if (name == TGSI_SEMANTIC_COLOR) {
      /*
       * The whole colour as fx10: G B in one register (high, low), R A in
       * another. One linker entry, reloaded into the same registers.
       */
      int e = -1;
      for (int i = 0; i < info->num_inputs; i++)
         if (c->input_loc[i] == (int)sem.location)
            e = i;
      if (e < 0) {
         e = new_entry(c, sem.location);
         if (e < 0)
            return;
         make_free(c, 2);
         int regs[2], n = 0;
         for (int t = 0; t < 4 && n < 2; t++)
            if (c->reg_busy_until[t] <= c->num_batches)
               regs[n++] = t;
         if (n < 2) {
            fail(c, "no free register to interpolate into");
            return;
         }
         c->input_reg[e][0] = regs[0];
         c->input_reg[e][1] = regs[1];
         info->inputs[e].dst = LINK_DST(0, regs[1], LINK_DST_FX10_LOW) |
                               LINK_DST(1, regs[0], LINK_DST_FX10_HIGH) |
                               LINK_DST(2, regs[0], LINK_DST_FX10_LOW) |
                               LINK_DST(3, regs[1], LINK_DST_FX10_HIGH);
      }
      for (int k = 0; k < 2; k++)
         if (c->reg_busy_until[c->input_reg[e][k]] > c->num_batches)
            evict(c, c->input_reg[e][k], c->input_reg[e][k], NULL, 0);
      struct fp_instr *inst = new_batch(c);
      struct fp_mfu_instr *mfu = CALLOC_STRUCT(fp_mfu_instr);
      list_inithead(&mfu->link);
      setup_barycentrics(mfu);
      for (int k = 0; k < 2; k++) {
         int r = c->input_reg[e][k];
         mfu->var[r].op = FP_VAR_OP_FX10;
         mfu->var[r].tram_row = c->input_row[e];
         c->reg_busy_until[r] = INT_MAX;
         c->max_reg = MAX2(c->max_reg, r);
      }
      list_addtail(&mfu->link, &c->fp->mfu_instructions);
      inst->mfu_sched.num_instructions = 1;

      static const bool high_of[4] = { false, true, false, true };
      for (unsigned k = 0; k < intr->def.num_components; k++) {
         unsigned cc = first + k;
         *def_val(c, &intr->def, k) = (struct val) {
            .kind = VAL_ROW, .index = c->input_reg[e][(cc == 1 || cc == 2) ? 0 : 1],
            .fx10 = true, .high = high_of[cc], .def = c->num_batches - 1,
            .last = c->num_batches,
         };
      }
      return;
   }

   /* texture coordinates: interpolated by the TEX itself, see emit_tex() */
   if (only_tex_coords(&intr->def)) {
      for (unsigned k = 0; k < intr->def.num_components; k++)
         *def_val(c, &intr->def, k) = (struct val) {
            .kind = VAL_VARYING, .index = sem.location, .slot = first + k,
         };
      return;
   }

   int n = intr->def.num_components, j[4] = { 0 };
   int want[4] = { -1, -1, -1, -1 }, reg[4];
   for (int k = 0; k < n; k++)
      j[k] = first + k;
   make_free(c, n);
   int e = place(c, sem.location, n, j, want, reg);
   if (e < 0)
      return;
   emit_interpolation(c, e, n, reg);
   for (int k = 0; k < n; k++)
      *def_val(c, &intr->def, k) = (struct val) {
         .kind = VAL_ROW, .index = reg[k], .def = c->num_batches - 1,
         .last = c->num_batches,
      };
}

/* ------------------------------------------------------------------ */
/* ALU */

static struct val
new_row_val(struct comp *c)
{
   struct val v = {
      .kind = VAL_ROW,
      .index = alloc_reg(c, 0, MAX_ROW_REGS - 1),
      .def = c->num_batches,
      .last = c->num_batches + 1,
   };
   return v;
}

static struct fp_alu_dst_operand
dst_of(const struct val *v, bool sat)
{
   return (struct fp_alu_dst_operand) {
      .index = v->index,
      /* both halves = the fp20 register (neither writes nothing) */
      .write_low_sub_reg = !v->fx10 || !v->high,
      .write_high_sub_reg = !v->fx10 || v->high,
      .saturate = sat,
   };
}

/* dst = a * b + c, one batch */
static void
emit_mad(struct comp *c, struct val *dst, enum fp_alu_op op,
         struct fp_alu_src_operand a, struct fp_alu_src_operand b,
         struct fp_alu_src_operand cc, bool sat)
{
   struct fp_alu_src_operand *srcs[3] = { &a, &b, &cc };

   /*
    * Join the open packet unless it's full or this reads a row register
    * one of its slots writes (the new value only shows next batch).
    */
   bool join = c->pkt && c->pkt_slots < 4;
   /* two slots writing the same half of a register: keep them apart */
   uint32_t halves = 0;
   if (dst->index < 16)
      halves = ((!dst->fx10 || !dst->high) ? 1u : 0) << (2 * dst->index) |
               ((!dst->fx10 || dst->high) ? 2u : 0) << (2 * dst->index);
   if (c->pkt_halves_written & halves)
      join = false;
   for (int i = 0; i < 3 && join; i++)
      if (!c->store_packet && srcs[i]->index < 16 &&
          (c->pkt_rows_written & (1u << srcs[i]->index)))
         join = false;

   /* a full packet: another one in the same batch (up to three) */
   bool conflict = c->pkt_halves_written & halves;
   for (int i = 0; i < 3; i++)
      conflict |= !c->store_packet && srcs[i]->index < 16 &&
                  (c->pkt_rows_written & (1u << srcs[i]->index));
   if (!join && c->pkt && !conflict && c->pkt_slots == 4 &&
       c->cur->alu_sched.num_instructions < 3) {
      struct fp_alu_instr_packet *p = CALLOC_STRUCT(fp_alu_instr_packet);
      list_inithead(&p->link);
      for (int i = 0; i < 4; i++)
         p->slots[i] = alu_nop();
      list_addtail(&p->link, &c->fp->alu_instructions);
      c->cur->alu_sched.num_instructions++;
      c->pkt = p;
      c->pkt_slots = 0;
      join = true;
   }

   if (!join) {
      struct grate_fp_shader *fp = c->fp;
      struct fp_instr *inst = batch_for(c, 3);
      struct fp_alu_instr_packet *p = CALLOC_STRUCT(fp_alu_instr_packet);
      list_inithead(&p->link);
      for (int i = 0; i < 4; i++)
         p->slots[i] = alu_nop();
      list_addtail(&p->link, &fp->alu_instructions);
      inst->alu_sched.num_instructions = 1;
      c->pkt = p;
      c->pkt_slots = 0;
      c->pkt_rows_written = 0;
      c->pkt_halves_written = 0;
   }

   int k = c->pkt_slots++;
   struct fp_alu_instr *slot = &c->pkt->slots[k];
   if (dst->index < 4)
      c->touched |= 1u << dst->index;
   for (int i = 0; i < 3; i++)
      if (srcs[i]->index < 4)
         c->touched |= 1u << srcs[i]->index;
   slot->op = op;
   slot->dst = dst_of(dst, sat);
   slot->src[0] = a;
   slot->src[1] = b;
   slot->src[2] = cc;
   slot->src[3] = lowp(true);

   if (dst->index < 16) {
      c->pkt_rows_written |= 1u << dst->index;
      c->pkt_halves_written |= halves;
      dst->pkt = c->pkt;
      dst->slot = k;
      dst->def = c->num_batches - 1;
      c->pkt_slot_row[k] = (struct fp_alu_src_operand) {
         .index = dst->index,
         .datatype = dst->fx10 ? FP_DATATYPE_FIXED10 : FP_DATATYPE_FP20,
         .sub_reg_select_high = dst->high,
      };
   }
}

static struct fp_alu_src_operand
use(struct comp *c, nir_alu_instr *alu, int s, unsigned chan)
{
   bool negate = false, absolute = false;
   nir_scalar sc = chase(nir_get_scalar(alu->src[s].src.ssa,
                                        alu->src[s].swizzle[chan]),
                         &negate, &absolute);
   struct val *v = def_val(c, sc.def, sc.comp);
   note_use(c, v);
   struct fp_alu_src_operand op = operand(c, v);
   if (absolute) {
      op.absolute_value = true;
      op.negate = false;
   }
   if (negate)
      op.negate = !op.negate;
   return op;
}

static struct fp_alu_src_operand
neg(struct fp_alu_src_operand s)
{
   s.negate = !s.negate;
   return s;
}

static struct fp_alu_src_operand
abs_(struct fp_alu_src_operand s)
{
   s.absolute_value = true;
   s.negate = false;
   return s;
}

/* SFU in its own batch: MFU computes op(reg) into an R0-R3 register */
static void
emit_sfu(struct comp *c, struct val *dst, enum fp_sfu_op op, struct val *src)
{
   if (src->kind != VAL_ROW || src->fx10) {
      /* the SFU reads a row register: copy into one first */
      struct val tmp = new_row_val(c);
      note_use(c, src);
      emit_mad(c, &tmp, FP_ALU_OP_MAD, operand(c, src), lowp(true), lowp(false),
               false);
      emit_sfu(c, dst, op, &tmp);
      tmp.last = c->num_batches;
      release(c, &tmp);
      return;
   }

   note_use(c, src);
   struct fp_instr *inst = new_batch(c);
   struct fp_mfu_instr *mfu = CALLOC_STRUCT(fp_mfu_instr);
   list_inithead(&mfu->link);
   mfu->sfu.op = op;
   mfu->sfu.reg = src->index;
   c->touched |= (1u << src->index) | (1u << dst->index);
   mfu->mul[0].dst = FP_MFU_MUL_DST_ROW_REG_0 + dst->index;
   mfu->mul[0].src[0] = FP_MFU_MUL_SRC_SFU_RESULT;
   mfu->mul[0].src[1] = FP_MFU_MUL_SRC_CONST_1;
   list_addtail(&mfu->link, &c->fp->mfu_instructions);
   inst->mfu_sched.num_instructions = 1;
}

/* colour component a scalar def feeds, if the store is its only reader */
static int
store_component(nir_def *def)
{
   if (!list_is_singular(&def->uses))
      return -1;
   nir_src *u = list_first_entry(&def->uses, nir_src, use_link);
   if (nir_src_is_if(u))
      return -1;
   nir_instr *user = nir_src_parent_instr(u);
   int k = 0;
   if (is_copy(user)) {
      nir_alu_instr *vec = nir_instr_as_alu(user);
      k = -1;
      for (unsigned i = 0; i < nir_op_infos[vec->op].num_inputs; i++)
         if (vec->src[i].src.ssa == def)
            k = k < 0 ? (int)i : -2;
      if (k < 0 || vec->op == nir_op_mov || !list_is_singular(&vec->def.uses))
         return -1;
      u = list_first_entry(&vec->def.uses, nir_src, use_link);
      if (nir_src_is_if(u))
         return -1;
      user = nir_src_parent_instr(u);
   }
   if (user->type != nir_instr_type_intrinsic)
      return -1;
   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(user);
   if (intr->intrinsic != nir_intrinsic_store_output)
      return -1;
   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   if (sem.location != FRAG_RESULT_COLOR && sem.location != FRAG_RESULT_DATA0)
      return -1;
   return nir_intrinsic_component(intr) + k;
}

/* nothing live in that half of the register, except this op's dying sources */
static bool
half_free(struct comp *c, nir_alu_instr *alu, int reg, bool high)
{
   for (unsigned i = 0; i < c->num_defs * NIR_MAX_VEC_COMPONENTS; i++) {
      struct val *v = &c->vals[i];
      if (v->kind != VAL_ROW || v->index != reg || (v->fx10 && v->high != high))
         continue;
      bool dying_src = false;
      for (unsigned s = 0; s < nir_op_infos[alu->op].num_inputs; s++) {
         for (unsigned ch = 0; ch < alu->def.num_components; ch++) {
            nir_scalar sc = chase_plain(alu->src[s].src.ssa, alu->src[s].swizzle[ch]);
            if (def_val(c, sc.def, sc.comp) == v &&
                last_use_comp(sc.def, sc.comp) <= alu->instr.index)
               dying_src = true;
         }
      }
      if (!dying_src)
         return false;
   }
   return true;
}

static void
emit_alu(struct comp *c, nir_alu_instr *alu)
{
   /* vecN and mov are looked through, see src_val() */
   if (is_copy(&alu->instr))
      return;

   unsigned n = alu->def.num_components;
   assert(n == 1 || alu->op == nir_op_mov || nir_op_is_vec(alu->op));

   for (unsigned ch = 0; ch < n; ch++) {
      struct val *dst = def_val(c, &alu->def, ch);
      bool sfu = alu->op == nir_op_frcp || alu->op == nir_op_frsq ||
                 alu->op == nir_op_fsqrt || alu->op == nir_op_flog2 ||
                 alu->op == nir_op_fexp2 || alu->op == nir_op_ffract;

      /* SFU results come out of the MFU muls, which only reach R0-R3 */
      int cc = sfu ? -1 : store_component(&alu->def);
      if (sfu) {
         *dst = (struct val) {
            .kind = VAL_ROW, .index = alloc_reg(c, 0, 3),
            .def = c->num_batches, .last = c->num_batches + 1,
         };
      } else if (cc >= 0 && half_free(c, alu, out_reg[cc], out_high[cc])) {
         /* only the colour output reads it: compute it in place */
         *dst = (struct val) {
            .kind = VAL_ROW, .index = out_reg[cc], .fx10 = true,
            .high = out_high[cc],
            .def = c->num_batches, .last = c->num_batches + 1,
         };
         c->reg_busy_until[out_reg[cc]] = INT_MAX;
      } else {
         *dst = new_row_val(c);
      }
      protect(c, dst);

      struct fp_alu_src_operand zero = lowp(false), one = lowp(true);

      switch (alu->op) {
      case nir_op_mov:
         emit_mad(c, dst, FP_ALU_OP_MAD, use(c, alu, 0, ch), one, zero, false);
         break;
      case nir_op_vec2:
      case nir_op_vec3:
      case nir_op_vec4: {
         struct val *s = src_val(c, alu->src[ch].src, alu->src[ch].swizzle[0]);
         note_use(c, s);
         emit_mad(c, dst, FP_ALU_OP_MAD, operand(c, s), one, zero, false);
         break;
      }
      case nir_op_fneg:
         emit_mad(c, dst, FP_ALU_OP_MAD, neg(use(c, alu, 0, ch)), one, zero,
                  false);
         break;
      case nir_op_fabs:
         emit_mad(c, dst, FP_ALU_OP_MAD, abs_(use(c, alu, 0, ch)), one, zero,
                  false);
         break;
      case nir_op_fsat:
         emit_mad(c, dst, FP_ALU_OP_MAD, use(c, alu, 0, ch), one, zero, true);
         break;
      case nir_op_fadd:
         emit_mad(c, dst, FP_ALU_OP_MAD, use(c, alu, 0, ch), one,
                  use(c, alu, 1, ch), false);
         break;
      case nir_op_fmul:
         emit_mad(c, dst, FP_ALU_OP_MAD, use(c, alu, 0, ch),
                  use(c, alu, 1, ch), zero, false);
         break;
      case nir_op_ffma:
         emit_mad(c, dst, FP_ALU_OP_MAD, use(c, alu, 0, ch),
                  use(c, alu, 1, ch), use(c, alu, 2, ch), false);
         break;
      case nir_op_fmin:
         emit_mad(c, dst, FP_ALU_OP_MIN, use(c, alu, 0, ch), one,
                  use(c, alu, 1, ch), false);
         break;
      case nir_op_fmax:
         emit_mad(c, dst, FP_ALU_OP_MAX, use(c, alu, 0, ch), one,
                  use(c, alu, 1, ch), false);
         break;
      case nir_op_fcsel:
         /* (-cond < 0) ? a : b, cond is 1.0 or 0.0 */
         emit_mad(c, dst, FP_ALU_OP_CSEL, neg(use(c, alu, 0, ch)),
                  use(c, alu, 1, ch), use(c, alu, 2, ch), false);
         break;
      case nir_op_slt:
      case nir_op_sge:
      case nir_op_seq:
      case nir_op_sne: {
         /* t = a - b, then select on its sign */
         struct val t = new_row_val(c);
         emit_mad(c, &t, FP_ALU_OP_MAD, use(c, alu, 0, ch), one,
                  neg(use(c, alu, 1, ch)), false);
         note_use(c, &t);
         struct fp_alu_src_operand ts = operand(c, &t);
         switch (alu->op) {
         case nir_op_slt: /* t < 0 ? 1 : 0 */
            emit_mad(c, dst, FP_ALU_OP_CSEL, ts, one, zero, false);
            break;
         case nir_op_sge: /* t < 0 ? 0 : 1 */
            emit_mad(c, dst, FP_ALU_OP_CSEL, ts, zero, one, false);
            break;
         case nir_op_seq: /* -|t| < 0 ? 0 : 1 */
            emit_mad(c, dst, FP_ALU_OP_CSEL, neg(abs_(ts)), zero, one, false);
            break;
         default: /* sne */
            emit_mad(c, dst, FP_ALU_OP_CSEL, neg(abs_(ts)), one, zero, false);
            break;
         }
         t.last = c->num_batches;
         release(c, &t);
         break;
      }
      case nir_op_frcp:
      case nir_op_frsq:
      case nir_op_fsqrt:
      case nir_op_flog2:
      case nir_op_ffract: {
         static const enum fp_sfu_op ops[] = {
            [nir_op_frcp] = FP_SFU_OP_RCP,
            [nir_op_frsq] = FP_SFU_OP_RSQ,
            [nir_op_fsqrt] = FP_SFU_OP_SQRT,
            [nir_op_flog2] = FP_SFU_OP_LG2,
            [nir_op_ffract] = FP_SFU_OP_FRC,
         };
         struct val *s = src_val(c, alu->src[0].src, alu->src[0].swizzle[ch]);
         emit_sfu(c, dst, ops[alu->op], s);
         break;
      }
      case nir_op_fexp2: {
         struct val *s = src_val(c, alu->src[0].src, alu->src[0].swizzle[ch]);
         emit_sfu(c, dst, FP_SFU_OP_PREEX2, s);
         dst->last = c->num_batches + 1;
         emit_sfu(c, dst, FP_SFU_OP_EX2, dst);
         break;
      }
      default:
         fail(c, nir_op_infos[alu->op].name);
         return;
      }
   }
}

/* ------------------------------------------------------------------ */
/* intrinsics, textures, output */

static void
emit_intrinsic(struct comp *c, nir_intrinsic_instr *intr)
{
   switch (intr->intrinsic) {
   case nir_intrinsic_load_interpolated_input:
   case nir_intrinsic_load_input:
      emit_load_input(c, intr);
      break;
   case nir_intrinsic_load_barycentric_pixel:
   case nir_intrinsic_load_barycentric_centroid:
   case nir_intrinsic_load_barycentric_sample:
      break; /* batch 0 */

   case nir_intrinsic_load_ubo:
   case nir_intrinsic_load_ubo_vec4: {
      if (!nir_src_is_const(intr->src[0]) || nir_src_as_uint(intr->src[0]) ||
          !nir_src_is_const(intr->src[1])) {
         fail(c, "indirect uniform");
         return;
      }
      unsigned dword = intr->intrinsic == nir_intrinsic_load_ubo_vec4 ?
         (nir_intrinsic_base(intr) + nir_src_as_uint(intr->src[1])) * 4 +
            nir_intrinsic_component(intr) :
         nir_src_as_uint(intr->src[1]) / 4;
      for (unsigned k = 0; k < intr->def.num_components; k++)
         *def_val(c, &intr->def, k) =
            uniform_val(c, GRATE_FP_UNIFORM_CB(dword + k));
      break;
   }

   case nir_intrinsic_load_frag_coord: {
      /*
       * pos.x = 8192 + x, pos.y = 8192 + (h - 1) - y: upper-left origin,
       * half-integer centres, h from the uniform table at draw time.
       */
      nir_component_mask_t read = nir_def_components_read(&intr->def);
      for (unsigned k = 0; k < intr->def.num_components; k++) {
         struct val *dst = def_val(c, &intr->def, k);
         if (k >= 2) {
            *dst = imm_val(c, 1.0f); /* z, w: 2D only for now */
            continue;
         }
         if (!(read & (1u << k)))
            continue;
         /*
          * fp20 has 13 explicit significand bits: around 8192 only whole
          * numbers exist, so take the integer pixel first, then add 0.5.
          * x = (pos.x - 8192) + 0.5, y = (pos.y * ys + yo) + 0.5, where the
          * draw code picks ys/yo by the viewport's y direction.
          */
         *dst = new_row_val(c);
         struct val pos = { .kind = VAL_FIXED, .index = REG_POS_X + k };
         struct val off = k == 0 ? imm_val(c, -8192.0f) :
                          uniform_val(c, GRATE_FP_UNIFORM_FB_YOFF);
         struct val scale = k == 0 ? imm_val(c, 1.0f) :
                            uniform_val(c, GRATE_FP_UNIFORM_FB_YSCALE);
         emit_mad(c, dst, FP_ALU_OP_MAD, operand(c, &pos), operand(c, &scale),
                  operand(c, &off), false);
         struct val half = imm_val(c, 0.5f);
         struct fp_alu_src_operand d = operand(c, dst);
         emit_mad(c, dst, FP_ALU_OP_MAD, d, lowp(true), operand(c, &half), false);
         dst->last = c->num_batches + 1;
      }
      break;
   }

   case nir_intrinsic_store_output: {
      nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
      if (sem.location != FRAG_RESULT_COLOR &&
          sem.location != FRAG_RESULT_DATA0) {
         fail(c, "unsupported output");
         return;
      }
      /* colour as fx10 B G R A in R2-R3, then DW from them */
      unsigned first = nir_intrinsic_component(intr);
      unsigned n = intr->src[0].ssa->num_components;
      /*
       * All the copies go in one fresh packet: its slots read R2-R3 before
       * any of them writes, so sources there need no moving.
       */
      c->pkt = NULL;
      c->store_packet = true;
      for (unsigned k = 0; k < n; k++) {
         unsigned cc = first + k;
         struct val *s = src_val(c, intr->src[0], k);
         note_use(c, s);
         if (at_output(s, cc))
            continue;
         struct val out = {
            .kind = VAL_ROW, .index = out_reg[cc], .fx10 = true,
            .high = out_high[cc],
         };
         /* R2/R3 must be free: everything else is dead by now except
          * values still to be copied, which may sit in R2/R3 themselves;
          * the allocator keeps outputs last, see compile(). */
         emit_mad(c, &out, FP_ALU_OP_MAD, operand(c, s), lowp(true), lowp(false),
                  false);
      }
      c->store_packet = false;
      struct fp_instr *dw = batch_for(c, 4);
      dw->dw.enable = 1;
      c->touched |= 0xc;
      dw->dw.index = 1;
      dw->dw.src_regs = FP_DW_REGS_R2_R3;
      break;
   }

   default:
      fail(c, nir_intrinsic_infos[intr->intrinsic].name);
      break;
   }
}

/* live values in row registers [lo, lo + 1] that are still read after instr */
static int
pair_live_after(struct comp *c, int lo, nir_tex_instr *tex, int coord)
{
   int n = 0;
   for (unsigned i = 0; i < c->num_defs * NIR_MAX_VEC_COMPONENTS; i++) {
      struct val *v = &c->vals[i];
      if (v->kind != VAL_ROW || v->index < lo || v->index > lo + 1)
         continue;
      /* the coordinates themselves don't count if TEX is their last use */
      bool coord_dying = false;
      for (int k = 0; k < 2; k++) {
         nir_scalar s = chase_plain(tex->src[coord].src.ssa, k);
         if (def_val(c, s.def, s.comp) == v &&
             last_use_comp(s.def, s.comp) <= tex->instr.index)
            coord_dying = true;
      }
      n += !coord_dying;
   }
   return n;
}

static bool
coords_at(struct comp *c, nir_tex_instr *tex, int coord, int lo)
{
   for (int k = 0; k < 2; k++) {
      struct val *s = src_val(c, tex->src[coord].src, k);
      if (s->kind != VAL_ROW || s->fx10 || s->index != lo + k)
         return false;
   }
   return true;
}

/*
 * A deferred coordinate whose partner is already in a register: interpolate
 * it into any free register, then emit_tex() moves the pair as usual.
 */
static bool
materialize_coords(struct comp *c, nir_tex_instr *tex, int coord)
{
   for (int k = 0; k < 2; k++) {
      struct val *v = src_val(c, tex->src[coord].src, k);
      if (v->kind != VAL_VARYING)
         continue;
      int j = v->slot, want = -1, reg;
      make_free(c, 1);
      int e = place(c, v->index, 1, &j, &want, &reg);
      if (e < 0)
         return false;
      emit_interpolation(c, e, 1, &reg);
      *v = (struct val) { .kind = VAL_ROW, .index = reg,
                          .def = c->num_batches - 1, .last = c->num_batches };
   }
   return true;
}

static void
emit_tex(struct comp *c, nir_tex_instr *tex)
{
   /*
    * External images (imported dma-bufs) sample like 2D textures here, and
    * an explicit LOD or bias of zero is what a plain lookup does anyway.
    */
   int extra = nir_tex_instr_src_index(tex, tex->op == nir_texop_txl ?
                                       nir_tex_src_lod : nir_tex_src_bias);
   bool zero_extra = extra >= 0 && nir_src_is_const(tex->src[extra].src) &&
                     nir_src_as_float(tex->src[extra].src) == 0.0f;
   bool op_ok = tex->op == nir_texop_tex ||
                ((tex->op == nir_texop_txl || tex->op == nir_texop_txb) &&
                 zero_extra);
   if (!op_ok || tex->is_array || tex->is_shadow ||
       (tex->sampler_dim != GLSL_SAMPLER_DIM_2D &&
        tex->sampler_dim != GLSL_SAMPLER_DIM_EXTERNAL)) {
      fail(c, "unsupported texture op");
      return;
   }

   int coord = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   if (coord < 0 || tex->num_srcs != 1 + (tex->op != nir_texop_tex)) {
      fail(c, "unsupported texture sources");
      return;
   }

   /*
    * TEX reads S, T as fp20 from R0-R1 or R2-R3 and writes the texel as
    * four fx10 to R0-R1 or R2-R3: pick the pairs that move the least.
    */
   int q;
   bool deferred = false;
   struct val *v0 = src_val(c, tex->src[coord].src, 0);
   struct val *v1 = src_val(c, tex->src[coord].src, 1);
   if (v0->kind == VAL_VARYING && v1->kind == VAL_VARYING &&
       v0->index == v1->index) {
      /* interpolate the coordinates straight into the pair TEX reads */
      q = pair_live_after(c, 2, tex, coord) < pair_live_after(c, 0, tex, coord) ? 2 : 0;
      evict(c, q, q + 1, NULL, 0);
      int j[2] = { v0->slot, v1->slot }, want[2] = { q, q + 1 }, reg[2];
      int e = place(c, v0->index, 2, j, want, reg);
      if (e < 0)
         return;
      emit_interpolation(c, e, 2, reg);
      deferred = true;
   } else if ((v0->kind == VAL_VARYING || v1->kind == VAL_VARYING) &&
              !materialize_coords(c, tex, coord)) {
      return;
   } else if (coords_at(c, tex, coord, 0))
      q = 0;
   else if (coords_at(c, tex, coord, 2))
      q = 2;
   else {
      q = pair_live_after(c, 2, tex, coord) < pair_live_after(c, 0, tex, coord) ? 2 : 0;
      /* coordinates already in the pair stay: one packet reads both
       * before it writes either */
      struct val *cv[2] = { src_val(c, tex->src[coord].src, 0),
                            src_val(c, tex->src[coord].src, 1) };
      evict(c, q, q + 1, cv, 2);
      c->pkt = NULL;
      c->store_packet = true;
      for (int k = 0; k < 2; k++) {
         struct val *s = cv[k];
         note_use(c, s);
         struct val d = { .kind = VAL_ROW, .index = q + k };
         emit_mad(c, &d, FP_ALU_OP_MAD, operand(c, s), lowp(true), lowp(false),
                  false);
      }
      c->store_packet = false;
      c->reg_busy_until[q] = c->reg_busy_until[q + 1] = INT_MAX;
   }
   for (int k = 0; k < 2; k++)
      note_use(c, src_val(c, tex->src[coord].src, k));

   /* ties go to R2-R3: the texel is then laid out as the colour output */
   int p = pair_live_after(c, 2, tex, coord) <= pair_live_after(c, 0, tex, coord) ? 2 : 0;
   if (pair_live_after(c, p, tex, coord)) {
      /* keep the coordinates readable by the TEX batch: they're in q; in
       * the same pair, only those TEX reads last need no moving */
      struct val *keep[2];
      int nkeep = 0;
      for (int k = 0; k < 2; k++) {
         nir_scalar sc = chase_plain(tex->src[coord].src.ssa, k);
         if (p != q || last_use_comp(sc.def, sc.comp) <= tex->instr.index)
            keep[nkeep++] = src_val(c, tex->src[coord].src, k);
      }
      evict(c, p, p + 1, keep, nkeep);
   }

   struct fp_instr *inst = batch_for(c, 2);
   inst->tex.enable = true;
   c->touched |= (3u << q) | (3u << p);
   inst->tex.sampler = tex->sampler_index;
   inst->tex.coords_from_r2 = q == 2;
   inst->tex.dst_r2 = p == 2;
   if (deferred && p != q)
      c->reg_busy_until[q] = c->reg_busy_until[q + 1] = c->num_batches;
   c->reg_busy_until[p] = c->reg_busy_until[p + 1] = INT_MAX;

   /* texel: B G R A as P.l P.h (P+1).l (P+1).h */
   static const int reg_of[4] = { 1, 0, 0, 1 };
   static const bool high_of[4] = { false, true, false, true };
   for (unsigned k = 0; k < tex->def.num_components; k++)
      *def_val(c, &tex->def, k) = (struct val) {
         .kind = VAL_ROW, .index = p + reg_of[k], .fx10 = true,
         .high = high_of[k],
         .def = c->num_batches - 1, .last = c->num_batches,
      };
}

static void
emit_load_const(struct comp *c, nir_load_const_instr *lc)
{
   for (unsigned k = 0; k < lc->def.num_components; k++)
      *def_val(c, &lc->def, k) = imm_val(c, nir_const_value_as_float(
                                         lc->value[k], lc->def.bit_size));
}

/*
 * Free a def's registers once its last user has been emitted. A texel's
 * two registers are freed together when all four components are dead.
 */
static bool
free_dead(nir_src *src, void *data)
{
   struct comp *c = data;
   nir_instr *user = nir_src_parent_instr(src);
   /* a copy's sources die through the copy's own users */
   if (is_copy(user))
      return true;
   for (unsigned k = 0; k < src->ssa->num_components; k++) {
      nir_scalar s = chase_plain(src->ssa, k);
      if (last_use_comp(s.def, s.comp) > user->index)
         continue;
      struct val *v = def_val(c, s.def, s.comp);
      if (v->kind == VAL_SPILLED) {
         v->kind = VAL_NONE;
         continue;
      }
      if (v->kind != VAL_ROW)
         continue;
      v->kind = VAL_NONE;
      /* the other fx10 half may still be live */
      bool shared = false;
      for (unsigned i = 0; i < c->num_defs * NIR_MAX_VEC_COMPONENTS; i++)
         shared |= c->vals[i].kind == VAL_ROW && c->vals[i].index == v->index;
      if (!shared)
         c->reg_busy_until[v->index] = MIN2(c->reg_busy_until[v->index],
                                            MAX2(v->last, c->num_batches));
   }
   return true;
}

static bool
prefree_dying(nir_src *src, void *data)
{
   struct comp *c = data;
   nir_instr *user = nir_src_parent_instr(src);
   for (unsigned k = 0; k < src->ssa->num_components; k++) {
      nir_scalar s = chase_plain(src->ssa, k);
      struct val *v = def_val(c, s.def, s.comp);
      if (v->kind != VAL_ROW || v->fx10 || last_use_comp(s.def, s.comp) > user->index)
         continue;
      /* every component of the def must die here (it's per register) */
      c->reg_busy_until[v->index] = MIN2(c->reg_busy_until[v->index],
                                         c->num_batches);
   }
   return true;
}


static uint32_t
fp20(float f)
{
   uint32_t u = fui(f);
   if (f == 0.0f)
      return 0;
   uint32_t sign = u >> 31, exponent = (u >> 23) & 0xff;
   int e = (int)exponent - 127 + 31;
   if (exponent == 0xff || e > 0x3f)
      e = 0x3f;
   if (e <= 0)
      return sign << 19;
   return sign << 19 | (uint32_t)e << 13 | (u & 0x7fffff) >> 10;
}

static bool
slot_is_nop(const struct fp_alu_instr *s)
{
   return s->dst.index == REG_LOWP_0_1 && s->src[0].index == REG_LOWP_0_1;
}
/*
 * Past 32 uniform registers: uniforms that only ever scale an fx10 value (a
 * texel or a colour, as a blur's kernel weights do) only need fx10 too, so
 * pack them two per register (low and high halves).
 */
static void
pack_weights(struct comp *c)
{
   struct grate_fp_info *info = &c->fp->info;
   int n = info->num_uniforms;
   bool ok[96];
   for (int k = 0; k < n; k++)
      ok[k] = GRATE_FP_UNIFORM_KIND(info->uniforms[k]) == GRATE_FP_UNIFORM_KIND_CB &&
              GRATE_FP_UNIFORM_PAYLOAD(info->uniforms[k]) < (1 << 14);

   list_for_each_entry(struct fp_alu_instr_packet, p, &c->fp->alu_instructions, link)
      for (int s = 0; s < (p->has_imm ? 3 : 4); s++)
         for (int o = 0; o < 4; o++) {
            struct fp_alu_instr *a = &p->slots[s];
            int k = a->src[o].index - REG_UNIFORM;
            if (k < 0 || k >= n)
               continue;
            /* a multiplicand (A or B) whose partner is fx10 */
            if (o > 1 || a->op != FP_ALU_OP_MAD ||
                a->src[1 - o].datatype != FP_DATATYPE_FIXED10)
               ok[k] = false;
         }

   /* pair candidates until it fits */
   int remap[96], half[96], out = 0;
   int pending = -1, need = n - GRATE_FP_MAX_UNIFORMS;
   uint32_t table[96];
   for (int k = 0; k < n; k++) {
      if (ok[k] && need > 0) {
         if (pending < 0) {
            pending = k;
            continue;
         }
         table[out] = GRATE_FP_UNIFORM_CB_PAIR(
            GRATE_FP_UNIFORM_PAYLOAD(info->uniforms[pending]),
            GRATE_FP_UNIFORM_PAYLOAD(info->uniforms[k]));
         remap[pending] = remap[k] = out++;
         half[pending] = 0;
         half[k] = 1;
         pending = -1;
         need--;
         continue;
      }
      remap[k] = out;
      half[k] = -1;
      table[out++] = info->uniforms[k];
   }
   if (pending >= 0) {
      remap[pending] = out;
      half[pending] = -1;
      table[out++] = info->uniforms[pending];
   }
   if (need > 0)
      return;

   list_for_each_entry(struct fp_alu_instr_packet, p, &c->fp->alu_instructions, link)
      for (int s = 0; s < (p->has_imm ? 3 : 4); s++)
         for (int o = 0; o < 4; o++) {
            struct fp_alu_src_operand *op = &p->slots[s].src[o];
            int k = op->index - REG_UNIFORM;
            if (k < 0 || k >= n)
               continue;
            op->index = REG_UNIFORM + remap[k];
            if (half[k] >= 0) {
               op->datatype = FP_DATATYPE_FIXED10;
               op->sub_reg_select_high = half[k];
            }
         }
   memcpy(info->uniforms, table, out * sizeof(uint32_t));
   info->num_uniforms = out;
}


/*
 * Move immediates out of the 32 uniform registers into the packets: a
 * packet whose fourth slot is free carries three fp20 constants instead
 * (registers 28-30). Then drop the table entries nothing reads any more.
 */
static void
embed_immediates(struct comp *c)
{
   struct grate_fp_info *info = &c->fp->info;

   list_for_each_entry(struct fp_alu_instr_packet, p, &c->fp->alu_instructions, link) {
      if (!slot_is_nop(&p->slots[3]))
         continue;
      int n = 0;
      int slot_of[96];
      memset(slot_of, -1, sizeof(slot_of));
      for (int s = 0; s < 3; s++) {
         for (int o = 0; o < 3; o++) {
            struct fp_alu_src_operand *op = &p->slots[s].src[o];
            int k = op->index - REG_UNIFORM;
            if (k < 0 || k >= info->num_uniforms || op->datatype != FP_DATATYPE_FP20 ||
                GRATE_FP_UNIFORM_KIND(info->uniforms[k]) != GRATE_FP_UNIFORM_KIND_IMM)
               continue;
            if (slot_of[k] < 0) {
               if (n == 3)
                  continue;
               slot_of[k] = n;
               p->imm[n++] =
                  fp20(uif(GRATE_FP_UNIFORM_PAYLOAD(info->uniforms[k]) << 4));
            }
            op->index = REG_EMBEDDED + slot_of[k];
         }
      }
      p->has_imm = n > 0;
   }

   /* compact the table */
   int remap[96];
   bool used[96] = { false };
   list_for_each_entry(struct fp_alu_instr_packet, p, &c->fp->alu_instructions, link)
      for (int s = 0; s < (p->has_imm ? 3 : 4); s++)
         for (int o = 0; o < 4; o++) {
            int k = p->slots[s].src[o].index - REG_UNIFORM;
            if (k >= 0 && k < info->num_uniforms)
               used[k] = true;
         }
   int n = 0;
   for (int k = 0; k < info->num_uniforms; k++) {
      remap[k] = used[k] ? n : -1;
      if (used[k])
         info->uniforms[n++] = info->uniforms[k];
   }
   list_for_each_entry(struct fp_alu_instr_packet, p, &c->fp->alu_instructions, link)
      for (int s = 0; s < (p->has_imm ? 3 : 4); s++)
         for (int o = 0; o < 4; o++) {
            struct fp_alu_src_operand *op = &p->slots[s].src[o];
            int k = op->index - REG_UNIFORM;
            if (k >= 0 && k < info->num_uniforms)
               op->index = REG_UNIFORM + remap[k];
         }
   info->num_uniforms = n;
   if (n > GRATE_FP_MAX_UNIFORMS)
      pack_weights(c);
   if (info->num_uniforms > GRATE_FP_MAX_UNIFORMS)
      fail(c, "too many fragment uniforms");
}

static void
segment_begin(struct comp *c, struct grate_fp_shader *fp)
{
   c->fp = fp;
   list_inithead(&fp->fp_instructions);
   list_inithead(&fp->alu_instructions);
   list_inithead(&fp->mfu_instructions);
   fp->alu_buf_size = 1;
   memset(&fp->info, 0, sizeof(fp->info));
   fp->info.color_input = -1;
   fp->info.max_tram_row = 1;

   c->num_batches = 0;
   memset(c->reg_busy_until, 0, sizeof(c->reg_busy_until));
   c->max_reg = 0;
   c->pkt = NULL;
   c->pkt_slots = 0;
   c->pkt_rows_written = 0;
   c->pkt_halves_written = 0;
   c->cur = NULL;
   c->cur_ipl = NULL;
   c->touched = 0;
   for (int i = 0; i < 16; i++) {
      c->input_row[i] = c->input_loc[i] = -1;
      c->tram_used[i] = 0;
      for (int j = 0; j < 4; j++)
         c->input_reg[i][j] = -1;
   }
   /* the slots were written by an earlier pass, if at all */
   memset(c->slot_last, 0, sizeof(c->slot_last));
}

/* everything still in a register goes to a spill slot, for the next pass */
static void
spill_all(struct comp *c)
{
   c->nprotect = 0;
   for (int lo = 0; lo <= 2 && !c->error; lo += 2) {
      unsigned mask = 0;
      for (unsigned i = 0; i < c->num_defs * NIR_MAX_VEC_COMPONENTS; i++) {
         struct val *v = &c->vals[i];
         if (v->kind == VAL_ROW && v->index >= lo && v->index <= lo + 1)
            mask |= 1u << (v->index - lo);
      }
      if (mask)
         spill_regs(c, lo, mask);
   }
}

static void
segment_end(struct comp *c)
{
   struct grate_fp_shader *fp = c->fp;

   /* a program needs at least one batch (a shader with no output has none) */
   if (!c->num_batches)
      new_batch(c);

   if (!c->error)
      embed_immediates(c);
   if (!c->error && c->num_batches > 63)
      fail(c, "program too long");

   /* the linker needs at least one instruction */
   if (!fp->info.num_inputs) {
      fp->info.inputs[0].src = LINK_SRC(0);
      fp->info.num_inputs = 1;
   }

   fp->alu_buf_size = DIV_ROUND_UP(c->max_reg + 1, 4);
   /* more ALU buffer rows (R4 and up) hang GR3D */
   if (!c->error && fp->alu_buf_size > 1)
      fail(c, "needs more than four row registers");
}

static void
fp_clear(struct grate_fp_shader *fp)
{
   list_for_each_entry_safe(struct fp_instr, i, &fp->fp_instructions, link)
      FREE(i);
   list_for_each_entry_safe(struct fp_alu_instr_packet, p, &fp->alu_instructions, link)
      FREE(p);
   list_for_each_entry_safe(struct fp_mfu_instr, m, &fp->mfu_instructions, link)
      FREE(m);
   list_inithead(&fp->fp_instructions);
   list_inithead(&fp->alu_instructions);
   list_inithead(&fp->mfu_instructions);
}

/*
 * Compile into up to max_segments programs (passes). A pass ends once it
 * holds `cut` batches or `ucut` uniforms: what's live then goes to spill
 * slots and the next pass reads it back. *num_segments is set on success.
 */
static bool
compile(struct grate_fp_shader *fps, int max_segments, int cut, int ucut,
        int *num_segments, nir_shader *s, bool allow_spill, const char **error)
{
   struct comp c = { .nir = s };
   struct grate_fp_shader *fp = &fps[0];
   int seg = 0;

   segment_begin(&c, fp);

   nir_function_impl *impl = nir_shader_get_entrypoint(s);
   if (!nir_cf_list_is_empty_block(&impl->body) &&
       !exec_list_is_singular(&impl->body)) {
      *error = "control flow";
      return false;
   }

   nir_index_ssa_defs(impl);
   nir_index_instrs(impl);
   c.num_defs = impl->ssa_alloc;
   c.vals = CALLOC(c.num_defs * NIR_MAX_VEC_COMPONENTS, sizeof(struct val));

   c.allow_spill = allow_spill;
   c.defs = CALLOC(c.num_defs, sizeof(nir_def *));
   nir_foreach_block(block, impl)
      nir_foreach_instr(instr, block) {
         nir_def *d = nir_instr_def(instr);
         if (d)
            c.defs[d->index] = d;
      }

   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (c.error)
            break;
         c.cur_index = instr->index;
         c.nprotect = 0;

         /* a full pass: hand over to the next one */
         if (seg + 1 < max_segments && !is_copy(instr) &&
             instr->type != nir_instr_type_load_const &&
             (c.num_batches >= cut || c.fp->info.num_uniforms >= ucut)) {
            spill_all(&c);
            segment_end(&c);
            if (c.error)
               break;
            fp = &fps[++seg];
            segment_begin(&c, fp);
            c.cur_index = instr->index;
            c.nprotect = 0;
         }

         /* spilled sources back into registers first */
         if (c.allow_spill && !is_copy(instr)) {
            c.ensure_varyings = instr->type != nir_instr_type_tex;
            nir_foreach_src(instr, ensure_src, &c);
         }
         /*
          * An ALU op reads its sources in its first batch, so a source
          * that dies here can hand its register to the result.
          */
         if (instr->type == nir_instr_type_alu &&
             !is_copy(instr))
            nir_foreach_src(instr, prefree_dying, &c);
         switch (instr->type) {
         case nir_instr_type_alu:
            emit_alu(&c, nir_instr_as_alu(instr));
            break;
         case nir_instr_type_intrinsic:
            emit_intrinsic(&c, nir_instr_as_intrinsic(instr));
            break;
         case nir_instr_type_tex:
            emit_tex(&c, nir_instr_as_tex(instr));
            break;
         case nir_instr_type_load_const:
            emit_load_const(&c, nir_instr_as_load_const(instr));
            break;
         case nir_instr_type_undef:
            for (unsigned k = 0; k < nir_instr_as_undef(instr)->def.num_components; k++)
               *def_val(&c, &nir_instr_as_undef(instr)->def, k) = imm_val(&c, 0.0f);
            break;
         default:
            fail(&c, "unsupported instruction type");
            break;
         }
         nir_foreach_src(instr, free_dead, &c);
      }
   }

   if (!c.error)
      segment_end(&c);

   for (int k = 0; k <= seg; k++) {
      fps[k].info.num_spill_slots = c.num_slots;
      for (int i = 0; i < c.num_slots; i++)
         if (c.slot_rgba8[i])
            fps[k].info.spill_rgba8 |= 1u << i;
   }

   FREE(c.vals);
   FREE(c.defs);
   *error = c.error;

   if (c.error && getenv("GRATE_FP_DUMP_FAILED")) {
      /* print, never run, what was built before the failure */
      fprintf(stderr, "grate: failed program (%s), %d batches:\n", c.error,
              c.num_batches);
      list_for_each_entry(struct fp_instr, i, &fp->fp_instructions, link) {
         fprintf(stderr, "EXEC mfu %d tex %d dw %d alu@%d x%d\n",
                 i->mfu_sched.num_instructions, i->tex.enable, i->dw.enable,
                 i->alu_sched.address, i->alu_sched.num_instructions);
      }
      int pk = 0;
      list_for_each_entry(struct fp_alu_instr_packet, p, &fp->alu_instructions, link) {
         for (int s = 0; s < 4; s++) {
            struct fp_alu_instr *a = &p->slots[s];
            if (slot_is_nop(a))
               continue;
            fprintf(stderr, "  alu%d.%d op%d r%d%s%s <- %d %d %d\n", pk, s, a->op,
                    a->dst.index, a->dst.write_low_sub_reg ? "l" : "",
                    a->dst.write_high_sub_reg ? "h" : "", a->src[0].index,
                    a->src[1].index, a->src[2].index);
         }
         pk++;
      }
   }
   if (c.error) {
      /* Gallium can't fail a shader: draw nothing instead */
      for (int k = 0; k <= seg; k++)
         fp_clear(&fps[k]);
      segment_begin(&c, &fps[0]);
      fps[0].info.inputs[0].src = LINK_SRC(0);
      fps[0].info.num_inputs = 1;
      new_batch(&c);
      seg = 0;
   }
   *num_segments = seg + 1;
   return !c.error;
}

/*
 * Reorder the (single-block) shader as a depth-first walk from the output:
 * each value is computed just before its first user, bigger operand trees
 * first (Sethi-Ullman), so few values are live at once.
 */
struct dfs_state {
   nir_instr **order;
   unsigned n;
   BITSET_WORD *done;
   unsigned *size;
};

static bool
dfs_size_src(nir_src *src, void *data)
{
   unsigned *sum = data;
   nir_instr *i = nir_def_instr(src->ssa);
   *sum += i->pass_flags;
   return true;
}

static void
dfs_visit(struct dfs_state *st, nir_instr *instr);

struct dfs_src_list {
   nir_instr *instr[16];
   unsigned n;
};

static bool
dfs_collect_src(nir_src *src, void *data)
{
   struct dfs_src_list *l = data;
   if (l->n < ARRAY_SIZE(l->instr))
      l->instr[l->n++] = nir_def_instr(src->ssa);
   return true;
}

static void
dfs_visit(struct dfs_state *st, nir_instr *instr)
{
   if (BITSET_TEST(st->done, instr->index))
      return;
   BITSET_SET(st->done, instr->index);

   struct dfs_src_list l = { .n = 0 };
   nir_foreach_src(instr, dfs_collect_src, &l);
   /* biggest subtree first */
   for (unsigned a = 0; a < l.n; a++)
      for (unsigned b = a + 1; b < l.n; b++)
         if (l.instr[b]->pass_flags > l.instr[a]->pass_flags) {
            nir_instr *t = l.instr[a];
            l.instr[a] = l.instr[b];
            l.instr[b] = t;
         }
   for (unsigned k = 0; k < l.n; k++)
      dfs_visit(st, l.instr[k]);
   st->order[st->n++] = instr;
}

static bool
reorder_depth_first(nir_shader *s)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(s);
   if (!exec_list_is_singular(&impl->body))
      return false;
   nir_block *block = nir_start_block(impl);

   nir_index_instrs(impl);
   unsigned count = 0;
   nir_foreach_instr(instr, block)
      count++;

   /* subtree sizes, in program order (sources come first) */
   nir_foreach_instr(instr, block) {
      unsigned sum = 1;
      nir_foreach_src(instr, dfs_size_src, &sum);
      instr->pass_flags = MIN2(sum, 255);
   }

   struct dfs_state st = {
      .order = calloc(count, sizeof(nir_instr *)),
      .done = calloc(BITSET_WORDS(count + 1), sizeof(BITSET_WORD)),
   };

   /* sinks: instructions without a def (stores), in their original order */
   nir_foreach_instr(instr, block) {
      nir_def *def = nir_instr_def(instr);
      if (!def)
         dfs_visit(&st, instr);
   }
   /* anything unreachable from a sink keeps its place at the end */
   nir_foreach_instr(instr, block)
      dfs_visit(&st, instr);

   for (unsigned i = 0; i < st.n; i++) {
      nir_instr_remove(st.order[i]);
      nir_instr_insert(nir_after_block(block), st.order[i]);
   }
   free(st.order);
   free(st.done);
   nir_progress(true, impl, nir_metadata_control_flow);
   return true;
}


/*
 * Reorder top-down, greedily: of the instructions whose sources are ready,
 * take the one that frees the most register space (its sources' last
 * reader) for the least it takes. Space is counted in fx10 halves (an fp20
 * value takes two) per component, so a sum of texture taps finishes each
 * tap before fetching the next, instead of one channel over every tap.
 */
static unsigned
pressure_unit(nir_instr *instr)
{
   switch (instr->type) {
   case nir_instr_type_alu:
      return is_copy(instr) ? 0 : 2;
   case nir_instr_type_tex:
      return 1;
   case nir_instr_type_intrinsic:
      return nir_instr_as_intrinsic(instr)->intrinsic ==
             nir_intrinsic_load_frag_coord ? 2 : 0;
   default:
      return 0;
   }
}

static unsigned
pressure_cost(nir_instr *instr)
{
   nir_def *def = nir_instr_def(instr);
   return def ? pressure_unit(instr) * def->num_components : 0;
}

/* the scalars (underlying def instruction index * 4 + component) instr
 * reads, through copies, without repeats */
struct pressure_reads {
   unsigned key[64];
   unsigned n;
};

static bool
pressure_collect(nir_src *src, void *data)
{
   struct pressure_reads *r = data;
   nir_instr *user = nir_src_parent_instr(src);
   unsigned comps[NIR_MAX_VEC_COMPONENTS], n = 0;
   if (user->type == nir_instr_type_alu) {
      nir_alu_instr *alu = nir_instr_as_alu(user);
      unsigned si = container_of(src, nir_alu_src, src) - alu->src;
      for (unsigned k = 0; k < nir_ssa_alu_instr_src_components(alu, si); k++)
         comps[n++] = alu->src[si].swizzle[k];
   } else {
      for (unsigned k = 0; k < src->ssa->num_components; k++)
         comps[n++] = k;
   }
   for (unsigned k = 0; k < n; k++) {
      nir_scalar sc = chase_plain(src->ssa, comps[k]);
      unsigned key = nir_def_instr(sc.def)->index * 4 + sc.comp;
      bool seen = false;
      for (unsigned i = 0; i < r->n; i++)
         seen |= r->key[i] == key;
      if (!seen && r->n < ARRAY_SIZE(r->key))
         r->key[r->n++] = key;
   }
   return true;
}

static void
pressure_reads_of(nir_instr *instr, struct pressure_reads *r)
{
   r->n = 0;
   if (!is_copy(instr))
      nir_foreach_src(instr, pressure_collect, r);
}

static bool
pressure_unready_src(nir_src *src, void *data)
{
   unsigned *n = data;
   (*n)++;
   return true;
}

static bool
reorder_pressure(nir_shader *s)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(s);
   if (!exec_list_is_singular(&impl->body))
      return false;
   nir_block *block = nir_start_block(impl);
   nir_index_instrs(impl);

   unsigned count = 0;
   nir_foreach_instr(instr, block)
      count = instr->index + 1;
   nir_instr **by_index = calloc(count, sizeof(nir_instr *));
   unsigned *remaining = calloc(count * 4, sizeof(unsigned));
   unsigned *unready = calloc(count, sizeof(unsigned));
   bool *done = calloc(count, sizeof(bool));
   nir_instr **order = calloc(count, sizeof(nir_instr *));
   unsigned n = 0;
   struct pressure_reads r;

   nir_foreach_instr(instr, block) {
      by_index[instr->index] = instr;
      nir_foreach_src(instr, pressure_unready_src, &unready[instr->index]);
      pressure_reads_of(instr, &r);
      for (unsigned i = 0; i < r.n; i++)
         remaining[r.key[i]]++;
   }

   while (n < count) {
      nir_instr *best = NULL;
      int best_gain = INT_MIN;
      for (unsigned i = 0; i < count; i++) {
         nir_instr *instr = by_index[i];
         if (!instr || done[i] || unready[i])
            continue;
         int gain;
         if (!pressure_cost(instr)) {
            /* copies, constants and loads take nothing: right away, but
             * the colour store last */
            gain = INT_MAX;
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic ==
                nir_intrinsic_store_output)
               gain = INT_MIN + 1;
         } else {
            pressure_reads_of(instr, &r);
            gain = -(int)pressure_cost(instr);
            for (unsigned k = 0; k < r.n; k++)
               if (remaining[r.key[k]] == 1)
                  gain += pressure_unit(by_index[r.key[k] / 4]);
         }
         if (gain > best_gain) {
            best = instr;
            best_gain = gain;
         }
      }
      if (!best)
         break;
      done[best->index] = true;
      order[n++] = best;
      pressure_reads_of(best, &r);
      for (unsigned k = 0; k < r.n; k++)
         if (remaining[r.key[k]])
            remaining[r.key[k]]--;
      nir_def *def = nir_instr_def(best);
      if (def)
         nir_foreach_use(u, def)
            unready[nir_src_parent_instr(u)->index]--;
   }

   bool ok = n == count;
   if (ok) {
      for (unsigned i = 0; i < n; i++) {
         nir_instr_remove(order[i]);
         nir_instr_insert(nir_after_block(block), order[i]);
      }
   }
   free(by_index);
   free(remaining);
   free(unready);
   free(done);
   free(order);
   nir_progress(ok, impl, nir_metadata_control_flow);
   return ok;
}

/*
 * Varyings and gl_FragCoord cost one MFU or ALU op to read again, while a
 * held value costs one of only four registers: give every user its own load.
 */
static bool
rematerialize_inputs(nir_shader *s)
{
   bool progress = false;
   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_load_input &&
                intr->intrinsic != nir_intrinsic_load_interpolated_input &&
                intr->intrinsic != nir_intrinsic_load_frag_coord)
               continue;
            bool first = true;
            nir_foreach_use_including_if_safe(u, &intr->def) {
               if (nir_src_is_if(u))
                  continue;
               if (first) {
                  first = false;
                  continue;
               }
               nir_instr *user = nir_src_parent_instr(u);
               nir_instr *copy = nir_instr_clone(s, instr);
               nir_instr_insert_before(user, copy);
               nir_src_rewrite(u, &nir_instr_as_intrinsic(copy)->def);
               progress = true;
            }
         }
      }
      nir_progress(progress, impl, nir_metadata_control_flow);
   }
   return progress;
}

static int
type_size_vec4(const struct glsl_type *type, bool bindless)
{
   return glsl_count_attribute_slots(type, false);
}

/* scalar, boolean-free, branch-free SSA for grate_nir_to_fp() */
void
grate_nir_fs_lower(nir_shader *s)
{
   const nir_opt_peephole_select_options flatten = {
      .limit = ~0u,
      .expensive_alu_ok = true,
   };
   bool progress;

   NIR_PASS(_, s, nir_lower_io, nir_var_shader_in | nir_var_shader_out,
            type_size_vec4, (nir_lower_io_options)0);
   NIR_PASS(_, s, nir_lower_io_to_scalar, nir_var_shader_in, NULL, NULL);
   NIR_PASS(_, s, nir_lower_alu_to_scalar, NULL, NULL);
   NIR_PASS(_, s, nir_lower_bool_to_float, false);
   do {
      progress = false;
      NIR_PASS(progress, s, nir_opt_loop_unroll);
      NIR_PASS(progress, s, nir_opt_peephole_select, &flatten);
      NIR_PASS(progress, s, nir_lower_alu_to_scalar, NULL, NULL);
      NIR_PASS(progress, s, nir_opt_copy_prop);
      NIR_PASS(progress, s, nir_opt_dce);
      NIR_PASS(progress, s, nir_opt_algebraic);
      NIR_PASS(progress, s, nir_opt_constant_folding);
      NIR_PASS(progress, s, nir_opt_dead_cf);
   } while (progress);
   /* a * b + c is one ALU slot (MAD) */
   NIR_PASS(_, s, nir_opt_algebraic_late);
   NIR_PASS(_, s, nir_opt_copy_prop);
   NIR_PASS(_, s, nir_opt_dce);
   /* sample textures and compute values late, next to their users */
   NIR_PASS(_, s, nir_opt_move, nir_move_tex_sample | nir_move_alu |
            nir_move_copies | nir_move_comparisons | nir_move_load_input);
   nir_sweep(s);
}

/*
 * Four row registers: no single instruction order fits every shader, so try
 * a few, cheapest code first:
 *  0. NIR's order;
 *  1. inputs re-loaded per user, then a depth-first order from the output;
 *  2. NIR's order through nir_schedule's register-pressure fallback;
 *  3. 1 then 2;
 *  4. a greedy order that frees registers first (reorder_pressure()).
 * Then the same orders again, spilling to render targets when registers
 * run out.
 */
static bool
try_compile(struct grate_fp_shader *fps, int *num_segments, nir_shader *s,
            int order, bool spill, int max_segments, int cut,
            const char **error)
{
   const nir_schedule_options sched = { .threshold = 4, .fallback = true };
   nir_shader *v = nir_shader_clone(NULL, s);
   if (order == 4) {
      NIR_PASS(_, v, reorder_pressure);
   } else {
      if (order & 1) {
         NIR_PASS(_, v, rematerialize_inputs);
         NIR_PASS(_, v, reorder_depth_first);
      }
      if (order & 2)
         NIR_PASS(_, v, nir_schedule, &sched);
   }
   bool ok = compile(fps, max_segments, cut, 28, num_segments, v, spill, error);
   ralloc_free(v);
   return ok;
}

bool
grate_nir_to_fp(struct grate_fp_shader *fps, int *num_segments, nir_shader *s,
                const char **error)
{
   const char *err = NULL;

   /* every order in one pass without spilling first, then with */
   for (int spill = 0; spill < 2; spill++)
      for (int order = 0; order < 5; order++)
         if (try_compile(fps, num_segments, s, order, spill, 1, INT_MAX,
                         spill || order ? &err : error))
            return true;

   /* then several passes, shorter ones until it fits */
   static const int cuts[] = { 56, 48, 40, 32 };
   for (unsigned i = 0; i < ARRAY_SIZE(cuts); i++)
      for (int order = 4; order >= 0; order--)
         if (try_compile(fps, num_segments, s, order, true,
                         GRATE_FP_MAX_SEGMENTS, cuts[i], &err))
            return true;
   return false;
}
