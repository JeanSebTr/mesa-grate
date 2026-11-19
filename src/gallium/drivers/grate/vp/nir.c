/*
 * NIR to vertex processor (VP) programs.
 *
 * The VP is a vec4 machine: each instruction pairs a vector op and a
 * scalar op, reads temporaries, attributes and uniforms with a swizzle and
 * negate/absolute modifiers, and writes temporaries or outputs through a
 * write mask. Values stay vectors here; each SSA value gets a temporary
 * for its live range. Inputs, uniforms and constants are read in place.
 */

#include "grate_compiler.h"
#include "vpir.h"

#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "util/u_memory.h"

/* temporaries 28..31 are for the fetch moves of legalize_fetches() */
#define VP_NUM_TEMPS 28

enum val_kind {
   VAL_NONE,
   VAL_TEMP,     /* a temporary */
   VAL_ATTRIB,   /* an attribute, read in place */
   VAL_UNIFORM,  /* a uniform (or an immediate's slot), read in place */
   VAL_CONST,    /* a constant, given an immediate slot when read */
};

struct val {
   enum val_kind kind;
   int index;
   /* component c of the value is component comp[c] of the register */
   uint8_t comp[4];
   uint32_t value[4];  /* VAL_CONST */
   unsigned num_values;
   unsigned last_use;  /* index of the last instruction reading it */
   /*
    * written straight to an output instead of a temporary: component c of
    * the value goes to component out_comp[c] of output out (-1: unused)
    */
   int out;
   int8_t out_comp[4];
   /* a vec only read by a store_output: its sources go there directly */
   bool folded;
};

struct vp_comp {
   struct grate_vp_shader *vp;
   struct val *vals;  /* by SSA index */
   unsigned temp_free_after[VP_NUM_TEMPS];  /* busy until this instr index */
   unsigned num_imm_comps[32];
   int num_outputs;
   int output_of_location[VARYING_SLOT_MAX];
   const char *error;
};

static void
fail(struct vp_comp *c, const char *error)
{
   if (!c->error)
      c->error = error;
}

static struct vp_instr *
new_instr(struct vp_comp *c)
{
   struct vp_instr *instr = CALLOC_STRUCT(vp_instr);
   list_inithead(&instr->link);
   instr->vec.op = VP_VEC_OP_NOP;
   instr->vec.dst.file = VP_DST_FILE_UNDEF;
   instr->scalar.op = VP_SCALAR_OP_NOP;
   instr->scalar.dst.file = VP_DST_FILE_UNDEF;
   for (int i = 0; i < 3; i++) {
      instr->vec.src[i].file = VP_SRC_FILE_UNDEF;
      for (int k = 0; k < 4; k++)
         instr->vec.src[i].swizzle[k] = k;
   }
   instr->scalar.src.file = VP_SRC_FILE_UNDEF;
   for (int k = 0; k < 4; k++)
      instr->scalar.src.swizzle[k] = k;
   list_addtail(&instr->link, &c->vp->instructions);
   return instr;
}

/* an immediate slot holding all of the constant's components */
static struct val
imm_val(struct vp_comp *c, const uint32_t *v, unsigned n)
{
   struct val r = { .kind = VAL_UNIFORM };
   struct grate_vp_shader *vp = c->vp;
   for (int s = 0; s <= vp->num_immediates && s < 32; s++) {
      if (s == vp->num_immediates) {
         vp->num_immediates++;
         c->num_imm_comps[s] = 0;
      }
      uint32_t comps[4];
      unsigned used = c->num_imm_comps[s];
      memcpy(comps, vp->immediates[s], sizeof(comps));
      bool ok = true;
      for (unsigned i = 0; i < n && ok; i++) {
         unsigned k = 0;
         while (k < used && comps[k] != v[i])
            k++;
         if (k == used) {
            if (used == 4)
               ok = false;
            else
               comps[used++] = v[i];
         }
         r.comp[i] = k;
      }
      if (!ok)
         continue;
      memcpy(vp->immediates[s], comps, sizeof(comps));
      c->num_imm_comps[s] = used;
      r.index = 255 - s;
      for (unsigned i = n; i < 4; i++)
         r.comp[i] = r.comp[n - 1];
      return r;
   }
   fail(c, "too many immediates");
   return r;
}

static struct val *
def_val(struct vp_comp *c, nir_def *def)
{
   return &c->vals[def->index];
}

/* where an immediate is read from (the rest of the val stays) */
static void
set_imm(struct vp_comp *c, nir_def *def, struct val imm)
{
   struct val *v = def_val(c, def);
   v->kind = imm.kind;
   v->index = imm.index;
   memcpy(v->comp, imm.comp, sizeof(v->comp));
}

static int
alloc_temp(struct vp_comp *c, unsigned now, unsigned until)
{
   for (int t = 0; t < VP_NUM_TEMPS; t++) {
      if (c->temp_free_after[t] <= now) {
         c->temp_free_after[t] = until;
         return t;
      }
   }
   fail(c, "out of temporaries");
   return 0;
}

/* a temporary for def, written by the instruction with index now */
static struct val *
temp_val(struct vp_comp *c, nir_def *def, unsigned now)
{
   struct val *v = def_val(c, def);
   v->kind = VAL_TEMP;
   /*
    * free from the instruction after the last use: an instruction reads
    * its sources before it writes, but keep it simple
    */
   v->index = alloc_temp(c, now, v->last_use + 1);
   for (int k = 0; k < 4; k++)
      v->comp[k] = k;
   return v;
}

static struct vp_src_operand
operand(struct vp_comp *c, nir_def *def, const uint8_t *swizzle,
        unsigned num_comps)
{
   struct vp_src_operand op = { 0 };
   struct val *v = def_val(c, def);
   if (v->kind == VAL_CONST)
      set_imm(c, def, imm_val(c, v->value, v->num_values));
   switch (v->kind) {
   case VAL_TEMP:
      op.file = VP_SRC_FILE_TEMP;
      break;
   case VAL_ATTRIB:
      op.file = VP_SRC_FILE_ATTRIB;
      break;
   case VAL_UNIFORM:
      op.file = VP_SRC_FILE_UNIFORM;
      break;
   default:
      fail(c, "source without a value");
      op.file = VP_SRC_FILE_UNDEF;
      break;
   }
   op.index = v->index;
   for (unsigned k = 0; k < 4; k++)
      op.swizzle[k] = v->comp[swizzle[MIN2(k, num_comps - 1)]];
   return op;
}

static struct vp_src_operand
alu_operand(struct vp_comp *c, nir_alu_instr *alu, unsigned s)
{
   struct val *v = def_val(c, &alu->def);
   if (v->out < 0)
      return operand(c, alu->src[s].src.ssa, alu->src[s].swizzle,
                     alu->def.num_components);
   /* output component j computes the value's component k */
   uint8_t swz[4];
   unsigned k0 = 0;
   for (unsigned k = 0; k < alu->def.num_components; k++)
      if (v->out_comp[k] >= 0)
         k0 = k;
   for (unsigned j = 0; j < 4; j++)
      swz[j] = alu->src[s].swizzle[k0];
   for (unsigned k = 0; k < alu->def.num_components; k++)
      if (v->out_comp[k] >= 0)
         swz[v->out_comp[k]] = alu->src[s].swizzle[k];
   return operand(c, alu->src[s].src.ssa, swz, 4);
}

static struct vp_src_operand
neg(struct vp_src_operand op)
{
   op.negate = !op.negate;
   return op;
}

static struct vp_dst_operand
dst_of(struct vp_comp *c, nir_alu_instr *alu, unsigned now)
{
   struct val *o = def_val(c, &alu->def);
   if (o->out >= 0) {
      struct vp_dst_operand dst = { .file = VP_DST_FILE_OUTPUT, .index = o->out };
      for (unsigned k = 0; k < alu->def.num_components; k++)
         if (o->out_comp[k] >= 0)
            dst.write_mask |= 1u << o->out_comp[k];
      c->vp->output_mask |= 1u << o->out;
      return dst;
   }
   struct val *v = temp_val(c, &alu->def, now);
   struct vp_dst_operand dst = {
      .file = VP_DST_FILE_TEMP,
      .index = v->index,
      .write_mask = BITFIELD_MASK(alu->def.num_components),
   };
   return dst;
}

static void
emit_vec(struct vp_comp *c, enum vp_vec_op op, struct vp_dst_operand dst,
         struct vp_src_operand a, struct vp_src_operand b,
         struct vp_src_operand cc)
{
   struct vp_instr *instr = new_instr(c);
   instr->vec.op = op;
   instr->vec.dst = dst;
   instr->vec.src[0] = a;
   instr->vec.src[1] = b;
   instr->vec.src[2] = cc;
}

static struct vp_src_operand
undef_src(void)
{
   struct vp_src_operand op = { .file = VP_SRC_FILE_UNDEF };
   for (int k = 0; k < 4; k++)
      op.swizzle[k] = k;
   return op;
}

static void
emit_vec1(struct vp_comp *c, enum vp_vec_op op, struct vp_dst_operand dst,
          struct vp_src_operand a)
{
   emit_vec(c, op, dst, a, undef_src(), undef_src());
}

static void
emit_vec2(struct vp_comp *c, enum vp_vec_op op, struct vp_dst_operand dst,
          struct vp_src_operand a, struct vp_src_operand b)
{
   /* ADD reads its operands from A and C */
   if (op == VP_VEC_OP_ADD)
      emit_vec(c, op, dst, a, undef_src(), b);
   else
      emit_vec(c, op, dst, a, b, undef_src());
}

static void
emit_scalar(struct vp_comp *c, enum vp_scalar_op op, struct vp_dst_operand dst,
            struct vp_src_operand a)
{
   struct vp_instr *instr = new_instr(c);
   instr->scalar.op = op;
   instr->scalar.dst = dst;
   instr->scalar.src = a;
}

/*
 * Moves of n components into dst: component j of dst from component
 * comp[j] of def[j] (skipped where def[j] is NULL), one move per register
 * read.
 */
static void
emit_moves(struct vp_comp *c, struct vp_dst_operand dst, nir_def **def,
           const uint8_t *comp)
{
   unsigned done = 0;
   for (unsigned j = 0; j < 4; j++) {
      if (!def[j] || (done & (1u << j)))
         continue;
      uint8_t swz[4] = { comp[j], comp[j], comp[j], comp[j] };
      struct vp_src_operand first = operand(c, def[j], swz, 4);
      struct vp_dst_operand d = dst;
      d.write_mask = 0;
      for (unsigned i = j; i < 4; i++) {
         if (!def[i] || (done & (1u << i)))
            continue;
         uint8_t si[4] = { comp[i], comp[i], comp[i], comp[i] };
         struct vp_src_operand op = operand(c, def[i], si, 4);
         if (op.file != first.file || op.index != first.index)
            continue;
         first.swizzle[i] = op.swizzle[0];
         d.write_mask |= 1u << i;
         done |= 1u << i;
      }
      if (dst.file == VP_DST_FILE_OUTPUT)
         c->vp->output_mask |= 1u << dst.index;
      emit_vec1(c, VP_VEC_OP_MOV, d, first);
   }
}

static void
emit_alu(struct vp_comp *c, nir_alu_instr *alu, unsigned now)
{
   unsigned n = alu->def.num_components;

   switch (alu->op) {
   case nir_op_vec2:
   case nir_op_vec3:
   case nir_op_vec4: {
      /* one move per source; sources are read before the temporary is
       * allocated, and a source may share it only if this is its last
       * use, which the moves don't care about */
      if (def_val(c, &alu->def)->folded)
         return; /* see the store_output */
      nir_def *defs[4] = { NULL };
      uint8_t comps[4] = { 0 };
      for (unsigned i = 0; i < n; i++) {
         defs[i] = alu->src[i].src.ssa;
         comps[i] = alu->src[i].swizzle[0];
      }
      emit_moves(c, dst_of(c, alu, now), defs, comps);
      return;
   }
   default:
      break;
   }

   struct vp_src_operand s[3];
   for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs && i < 3; i++)
      s[i] = alu_operand(c, alu, i);

   switch (alu->op) {
   case nir_op_mov:
      emit_vec1(c, VP_VEC_OP_MOV, dst_of(c, alu, now), s[0]);
      break;
   case nir_op_fneg:
      emit_vec1(c, VP_VEC_OP_MOV, dst_of(c, alu, now), neg(s[0]));
      break;
   case nir_op_fabs:
      s[0].absolute = true;
      s[0].negate = false;
      emit_vec1(c, VP_VEC_OP_MOV, dst_of(c, alu, now), s[0]);
      break;
   case nir_op_fsat: {
      struct vp_dst_operand dst = dst_of(c, alu, now);
      dst.saturate = true;
      emit_vec1(c, VP_VEC_OP_MOV, dst, s[0]);
      break;
   }
   case nir_op_fadd:
      emit_vec2(c, VP_VEC_OP_ADD, dst_of(c, alu, now), s[0], s[1]);
      break;
   case nir_op_fsub:
      emit_vec2(c, VP_VEC_OP_ADD, dst_of(c, alu, now), s[0], neg(s[1]));
      break;
   case nir_op_fmul:
      emit_vec2(c, VP_VEC_OP_MUL, dst_of(c, alu, now), s[0], s[1]);
      break;
   case nir_op_ffma:
      emit_vec(c, VP_VEC_OP_MAD, dst_of(c, alu, now), s[0], s[1], s[2]);
      break;
   case nir_op_fmin:
      emit_vec2(c, VP_VEC_OP_MIN, dst_of(c, alu, now), s[0], s[1]);
      break;
   case nir_op_fmax:
      emit_vec2(c, VP_VEC_OP_MAX, dst_of(c, alu, now), s[0], s[1]);
      break;
   case nir_op_slt:
      emit_vec2(c, VP_VEC_OP_SLT, dst_of(c, alu, now), s[0], s[1]);
      break;
   case nir_op_sge:
      emit_vec2(c, VP_VEC_OP_SGE, dst_of(c, alu, now), s[0], s[1]);
      break;
   case nir_op_seq:
      emit_vec2(c, VP_VEC_OP_SEQ, dst_of(c, alu, now), s[0], s[1]);
      break;
   case nir_op_sne:
      emit_vec2(c, VP_VEC_OP_SNE, dst_of(c, alu, now), s[0], s[1]);
      break;
   case nir_op_ffloor:
      emit_vec1(c, VP_VEC_OP_FLR, dst_of(c, alu, now), s[0]);
      break;
   case nir_op_ffract:
      emit_vec1(c, VP_VEC_OP_FRC, dst_of(c, alu, now), s[0]);
      break;
   case nir_op_fsign:
      emit_vec1(c, VP_VEC_OP_SSG, dst_of(c, alu, now), s[0]);
      break;
   case nir_op_fdot3:
   case nir_op_fdot4: {
      /* the operands are vec3/vec4: swizzle them by their own width */
      unsigned w = alu->op == nir_op_fdot3 ? 3 : 4;
      struct vp_src_operand a = operand(c, alu->src[0].src.ssa, alu->src[0].swizzle, w);
      struct vp_src_operand b = operand(c, alu->src[1].src.ssa, alu->src[1].swizzle, w);
      emit_vec2(c, w == 3 ? VP_VEC_OP_DP3 : VP_VEC_OP_DP4,
                dst_of(c, alu, now), a, b);
      break;
   }
   case nir_op_fcsel: {
      /* c ? a : b with c 0.0 or 1.0 (booleans are floats): b + c * (a - b) */
      struct vp_src_operand cnd = s[0], a = s[1], b = s[2];
      struct vp_dst_operand dst = dst_of(c, alu, now);
      emit_vec2(c, VP_VEC_OP_ADD, dst, a, neg(b));
      struct vp_src_operand d = { .file = VP_SRC_FILE_TEMP, .index = dst.index };
      for (int k = 0; k < 4; k++)
         d.swizzle[k] = k;
      emit_vec(c, VP_VEC_OP_MAD, dst, cnd, d, b);
      break;
   }
   case nir_op_frsq:
   case nir_op_frcp:
   case nir_op_fexp2:
   case nir_op_flog2: {
      enum vp_scalar_op op = alu->op == nir_op_frsq ? VP_SCALAR_OP_RSQ :
                             alu->op == nir_op_frcp ? VP_SCALAR_OP_RCP :
                             alu->op == nir_op_fexp2 ? VP_SCALAR_OP_EX2 :
                             VP_SCALAR_OP_LG2;
      emit_scalar(c, op, dst_of(c, alu, now), s[0]);
      break;
   }
   default:
      fail(c, "unsupported ALU op");
      break;
   }
}

static int
output_index(struct vp_comp *c, unsigned location)
{
   if (c->output_of_location[location] < 0) {
      if (c->num_outputs >= 16) {
         fail(c, "too many outputs");
         return 0;
      }
      c->output_of_location[location] = c->num_outputs++;
   }
   return c->output_of_location[location];
}

static void
emit_intrinsic(struct vp_comp *c, nir_intrinsic_instr *intr, unsigned now)
{
   switch (intr->intrinsic) {
   case nir_intrinsic_load_input: {
      if (!nir_src_is_const(intr->src[0])) {
         fail(c, "indirect input");
         return;
      }
      struct val *v = def_val(c, &intr->def);
      v->kind = VAL_ATTRIB;
      v->index = nir_intrinsic_base(intr) + nir_src_as_uint(intr->src[0]);
      for (unsigned k = 0; k < 4; k++)
         v->comp[k] = MIN2(nir_intrinsic_component(intr) + k, 3);
      return;
   }
   case nir_intrinsic_load_ubo: {
      if (!nir_src_is_const(intr->src[0]) || nir_src_as_uint(intr->src[0]) ||
          !nir_src_is_const(intr->src[1])) {
         fail(c, "indirect uniform");
         return;
      }
      unsigned offset = nir_src_as_uint(intr->src[1]);
      struct val *v = def_val(c, &intr->def);
      v->kind = VAL_UNIFORM;
      v->index = offset / 16;
      if (offset % 4 || (offset % 16) / 4 + intr->def.num_components > 4 ||
          v->index >= 255 - 32) {
         fail(c, "unsupported uniform load");
         return;
      }
      for (unsigned k = 0; k < 4; k++)
         v->comp[k] = MIN2((offset % 16) / 4 + k, 3);
      return;
   }
   case nir_intrinsic_load_ubo_vec4: {
      if (!nir_src_is_const(intr->src[0]) || nir_src_as_uint(intr->src[0]) ||
          !nir_src_is_const(intr->src[1])) {
         fail(c, "indirect uniform");
         return;
      }
      struct val *v = def_val(c, &intr->def);
      unsigned first = nir_intrinsic_component(intr);
      v->kind = VAL_UNIFORM;
      v->index = nir_intrinsic_base(intr) + nir_src_as_uint(intr->src[1]);
      if (first + intr->def.num_components > 4 || v->index >= 255 - 32) {
         fail(c, "unsupported uniform load");
         return;
      }
      for (unsigned k = 0; k < 4; k++)
         v->comp[k] = MIN2(first + k, 3);
      return;
   }
   case nir_intrinsic_store_output: {
      nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
      if (!nir_src_is_const(intr->src[1])) {
         fail(c, "indirect output");
         return;
      }
      int o = output_index(c, sem.location + nir_src_as_uint(intr->src[1]));
      unsigned first = nir_intrinsic_component(intr);
      unsigned mask = nir_intrinsic_write_mask(intr);
      nir_def *src = intr->src[0].ssa;
      struct val *sv = def_val(c, src);
      /* output component first + i from component i of the value, or of
       * the folded vec's source i; values already written there skipped */
      nir_def *defs[4] = { NULL };
      uint8_t comps[4] = { 0 };
      nir_alu_instr *vec = sv->folded ? nir_def_as_alu(src) : NULL;
      for (unsigned i = 0; i < src->num_components && first + i < 4; i++) {
         if (!(mask & (1u << i)))
            continue;
         nir_def *d = vec ? vec->src[i].src.ssa : src;
         unsigned k = vec ? vec->src[i].swizzle[0] : i;
         if (def_val(c, d)->out == o)
            continue;
         defs[first + i] = d;
         comps[first + i] = k;
      }
      struct vp_dst_operand dst = { .file = VP_DST_FILE_OUTPUT, .index = o };
      c->vp->output_mask |= 1u << o;
      emit_moves(c, dst, defs, comps);
      return;
   }
   default:
      fail(c, "unsupported intrinsic");
      return;
   }
}

/*
 * An instruction fetches one uniform and one attribute (a single fetch
 * index each, see vp/pack.c): further distinct uniforms or attributes are
 * moved to temporaries 31, 30... first. Immediates are uniforms too.
 */
static void
legalize_fetches(struct vp_comp *c, struct vp_instr *instr)
{
   struct vp_src_operand *srcs[4] = {
      &instr->vec.src[0], &instr->vec.src[1], &instr->vec.src[2],
      &instr->scalar.src,
   };
   int uniform = -1, attr = -1, scratch = 31;
   for (int i = 0; i < 4; i++) {
      struct vp_src_operand *src = srcs[i];
      int *fetch;
      if (src->file == VP_SRC_FILE_UNIFORM)
         fetch = &uniform;
      else if (src->file == VP_SRC_FILE_ATTRIB)
         fetch = &attr;
      else
         continue;
      if (*fetch < 0 || *fetch == src->index) {
         *fetch = src->index;
         continue;
      }
      /* the whole register into a temporary, then read it from there */
      struct vp_instr *mov = CALLOC_STRUCT(vp_instr);
      mov->vec.op = VP_VEC_OP_MOV;
      mov->vec.dst = (struct vp_dst_operand) {
         .file = VP_DST_FILE_TEMP, .index = scratch, .write_mask = 0xf,
      };
      mov->vec.src[0] = (struct vp_src_operand) {
         .file = src->file, .index = src->index,
         .swizzle = { VP_SWZ_X, VP_SWZ_Y, VP_SWZ_Z, VP_SWZ_W },
      };
      mov->vec.src[1] = mov->vec.src[2] = undef_src();
      mov->scalar.op = VP_SCALAR_OP_NOP;
      mov->scalar.dst.file = VP_DST_FILE_UNDEF;
      mov->scalar.src = undef_src();
      list_addtail(&mov->link, &instr->link); /* before instr */
      src->file = VP_SRC_FILE_TEMP;
      src->index = scratch--;
   }
}

static bool
note_last_use(nir_src *src, void *data)
{
   struct vp_comp *c = data;
   nir_instr *user = nir_src_parent_instr(src);
   struct val *v = &c->vals[src->ssa->index];
   v->last_use = MAX2(v->last_use, user->index);
   return true;
}

/* the ops the scalar unit does, one component at a time */
static bool
scalar_op_filter(const nir_instr *instr, const void *data)
{
   if (instr->type != nir_instr_type_alu)
      return false;
   switch (nir_instr_as_alu(instr)->op) {
   case nir_op_frsq:
   case nir_op_frcp:
   case nir_op_fexp2:
   case nir_op_flog2:
   case nir_op_fdot2:
      return true;
   default:
      return false;
   }
}

/* ALU ops whose result can be written straight to an output */
static bool
writes_any_dst(nir_def *def)
{
   if (!nir_def_is_alu(def))
      return false;
   switch (nir_def_as_alu(def)->op) {
   case nir_op_vec2:
   case nir_op_vec3:
   case nir_op_vec4:
   case nir_op_fcsel: /* reads back its destination */
      return false;
   default:
      return true;
   }
}

/* every use of def is a source of user */
static bool
only_used_by(nir_def *def, nir_instr *user)
{
   nir_foreach_use_including_if(u, def)
      if (nir_src_is_if(u) || nir_src_parent_instr(u) != user)
         return false;
   return true;
}

/*
 * Before emitting: let what a store_output stores be computed straight
 * into the output where nothing else reads it, and a vec only stored there
 * be put together in the output itself.
 */
static void
plan_output(struct vp_comp *c, nir_intrinsic_instr *intr)
{
   if (intr->intrinsic != nir_intrinsic_store_output ||
       !nir_src_is_const(intr->src[1]))
      return;
   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   int o = output_index(c, sem.location + nir_src_as_uint(intr->src[1]));
   unsigned first = nir_intrinsic_component(intr);
   unsigned mask = nir_intrinsic_write_mask(intr);
   nir_def *src = intr->src[0].ssa;
   if (first + src->num_components > 4)
      return;

   if (writes_any_dst(src) && only_used_by(src, &intr->instr) &&
       mask == BITFIELD_MASK(src->num_components)) {
      struct val *v = def_val(c, src);
      v->out = o;
      for (unsigned i = 0; i < src->num_components; i++)
         v->out_comp[i] = first + i;
      return;
   }

   if (!nir_def_is_alu(src) || !only_used_by(src, &intr->instr))
      return;
   nir_alu_instr *vec = nir_def_as_alu(src);
   if (vec->op != nir_op_vec2 && vec->op != nir_op_vec3 && vec->op != nir_op_vec4)
      return;
   def_val(c, src)->folded = true;
   for (unsigned i = 0; i < src->num_components; i++) {
      nir_def *d = vec->src[i].src.ssa;
      /* the vec's sources are read at the store now */
      struct val *dv = def_val(c, d);
      dv->last_use = MAX2(dv->last_use, intr->instr.index);
   }
   for (unsigned i = 0; i < src->num_components; i++) {
      nir_def *d = vec->src[i].src.ssa;
      struct val *dv = def_val(c, d);
      if (!(mask & (1u << i)) || !writes_any_dst(d) || dv->out >= 0 ||
          !only_used_by(d, &vec->instr))
         continue;
      /* each component of d at most once, and all of them in the vec */
      int8_t oc[4];
      memset(oc, -1, sizeof(oc));
      bool ok = true;
      for (unsigned j = 0; j < src->num_components && ok; j++) {
         if (vec->src[j].src.ssa != d)
            continue;
         unsigned k = vec->src[j].swizzle[0];
         ok = (mask & (1u << j)) && oc[k] < 0;
         oc[k] = first + j;
      }
      for (unsigned k = 0; k < d->num_components && ok; k++)
         ok = oc[k] >= 0;
      if (!ok)
         continue;
      dv->out = o;
      memcpy(dv->out_comp, oc, sizeof(oc));
   }
}

static int
type_size_vec4(const struct glsl_type *type, bool bindless)
{
   return glsl_count_attribute_slots(type, false);
}

static uint8_t
vectorize_filter(const nir_instr *instr, const void *data)
{
   return scalar_op_filter(instr, data) ? 1 : 4;
}

/* branch-free SSA with float booleans for grate_nir_to_vp() */
void
grate_nir_vs_lower(nir_shader *s)
{
   const nir_opt_peephole_select_options flatten = {
      .limit = ~0u,
      .expensive_alu_ok = true,
   };
   bool progress;

   if (!s->info.io_lowered)
      NIR_PASS(_, s, nir_lower_io, nir_var_shader_in | nir_var_shader_out,
               type_size_vec4, (nir_lower_io_options)0);
   do {
      progress = false;
      NIR_PASS(progress, s, nir_opt_loop_unroll);
      NIR_PASS(progress, s, nir_opt_peephole_select, &flatten);
      NIR_PASS(progress, s, nir_opt_copy_prop);
      NIR_PASS(progress, s, nir_opt_dce);
      NIR_PASS(progress, s, nir_opt_algebraic);
      NIR_PASS(progress, s, nir_opt_constant_folding);
      NIR_PASS(progress, s, nir_opt_dead_cf);
      NIR_PASS(progress, s, nir_opt_cse);
      /* components nothing reads aren't computed */
      NIR_PASS(progress, s, nir_opt_shrink_vectors, true);
   } while (progress);
   NIR_PASS(_, s, nir_lower_bool_to_float, true);
   NIR_PASS(_, s, nir_lower_alu_to_scalar, scalar_op_filter, NULL);
   NIR_PASS(_, s, nir_opt_algebraic);
   NIR_PASS(_, s, nir_opt_algebraic_late);
   /* scalar ops of the same kind side by side as one vector op */
   NIR_PASS(_, s, nir_opt_vectorize, vectorize_filter, NULL);
   NIR_PASS(_, s, nir_opt_constant_folding);
   NIR_PASS(_, s, nir_opt_copy_prop);
   NIR_PASS(_, s, nir_opt_cse);
   NIR_PASS(_, s, nir_opt_dce);
   nir_sweep(s);
}

bool
grate_nir_to_vp(struct grate_vp_shader *vp, nir_shader *s, const char **error)
{
   struct vp_comp c = { .vp = vp };
   list_inithead(&vp->instructions);
   vp->output_mask = 0;
   memset(vp->output_semantic, 0, sizeof(vp->output_semantic));
   vp->num_immediates = 0;
   memset(c.output_of_location, -1, sizeof(c.output_of_location));
   /* the position is export 0 */
   c.output_of_location[VARYING_SLOT_POS] = c.num_outputs++;

   nir_function_impl *impl = nir_shader_get_entrypoint(s);
   if (exec_list_length(&impl->body) != 1) {
      *error = "control flow";
      return false;
   }
   nir_index_ssa_defs(impl);
   nir_index_instrs(impl);
   c.vals = rzalloc_array(NULL, struct val, impl->ssa_alloc);

   for (unsigned i = 0; i < impl->ssa_alloc; i++) {
      c.vals[i].out = -1;
      memset(c.vals[i].out_comp, -1, sizeof(c.vals[i].out_comp));
   }
   nir_foreach_block(block, impl)
      nir_foreach_instr(instr, block)
         nir_foreach_src(instr, note_last_use, &c);
   nir_foreach_block(block, impl)
      nir_foreach_instr(instr, block)
         if (instr->type == nir_instr_type_intrinsic)
            plan_output(&c, nir_instr_as_intrinsic(instr));

   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         unsigned now = instr->index;
         switch (instr->type) {
         case nir_instr_type_alu:
            emit_alu(&c, nir_instr_as_alu(instr), now);
            break;
         case nir_instr_type_intrinsic:
            emit_intrinsic(&c, nir_instr_as_intrinsic(instr), now);
            break;
         case nir_instr_type_load_const: {
            nir_load_const_instr *lc = nir_instr_as_load_const(instr);
            struct val *v = def_val(&c, &lc->def);
            v->kind = VAL_CONST;
            v->num_values = lc->def.num_components;
            for (unsigned k = 0; k < lc->def.num_components; k++)
               v->value[k] = lc->value[k].u32;
            break;
         }
         case nir_instr_type_undef: {
            nir_undef_instr *u = nir_instr_as_undef(instr);
            struct val *v = def_val(&c, &u->def);
            v->kind = VAL_CONST;
            v->num_values = u->def.num_components;
            break;
         }
         default:
            fail(&c, "unsupported instruction");
            break;
         }
      }
   }

   for (unsigned l = 0; l < VARYING_SLOT_MAX; l++) {
      int o = c.output_of_location[l];
      if (o < 0 || !(vp->output_mask & (1u << o)))
         continue;
      unsigned name, index;
      grate_varying_semantic(l, &name, &index);
      vp->output_semantic[o] = name << 8 | index;
   }

   /* one uniform and one attribute fetch per instruction */
   list_for_each_entry_safe(struct vp_instr, instr, &vp->instructions, link)
      legalize_fetches(&c, instr);

   if (list_is_empty(&vp->instructions))
      new_instr(&c);

   ralloc_free(c.vals);
   if (c.error) {
      *error = c.error;
      list_for_each_entry_safe(struct vp_instr, instr, &vp->instructions, link)
         FREE(instr);
      list_inithead(&vp->instructions);
      return false;
   }
   return true;
}
