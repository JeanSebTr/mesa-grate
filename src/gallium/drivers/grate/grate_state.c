#include <stdio.h>
#include <math.h>

#include "util/format/u_format.h"
#include "util/u_bitcast.h"
#include "util/u_helpers.h"
#include "util/u_framebuffer.h"
#include "util/u_inlines.h"
#include "util/u_math.h"
#include "util/u_memory.h"

#include "grate_common.h"
#include "grate_context.h"
#include "grate_program.h"
#include "grate_resource.h"
#include "grate_state.h"

#include "tgr_3d.xml.h"
#include "host1x01_hardware.h"

static void
grate_set_sample_mask(struct pipe_context *pcontext,
                      unsigned int sample_mask)
{
   unimplemented();
}

static void
grate_set_constant_buffer(struct pipe_context *pcontext, mesa_shader_stage shader,
                          uint index,
                          const struct pipe_constant_buffer *buffer)
{
   struct grate_context *context = grate_context(pcontext);

   assert(index == 0);
   assert(!buffer || buffer->user_buffer);

   util_copy_constant_buffer(&context->constant_buffer[shader], buffer);
}

static void
grate_set_framebuffer_state(struct pipe_context *pcontext,
                            const struct pipe_framebuffer_state *framebuffer)
{
   struct grate_context *context = grate_context(pcontext);
   unsigned int i;
   uint32_t mask = 0;

   if (framebuffer->zsbuf.texture &&
       !grate_resource(framebuffer->zsbuf.texture)->no_hw_zs) {
      struct grate_resource *res = grate_resource(framebuffer->zsbuf.texture);
      uint32_t rt_params;

      rt_params  = TGR3D_VAL(RT_PARAMS, FORMAT, res->format);
      rt_params |= TGR3D_VAL(RT_PARAMS, PITCH, res->pitch);
      rt_params |= TGR3D_BOOL(RT_PARAMS, TILED, res->tiled);

      context->framebuffer.rt_params[0] = rt_params;
      context->framebuffer.bos[0] = res->bo;
      mask |= 1;
   } else {
      context->framebuffer.rt_params[0] = 0;
      context->framebuffer.bos[0] = NULL;
   }

   for (i = 0; i < framebuffer->nr_cbufs; i++) {
      struct grate_resource *res = grate_resource(framebuffer->cbufs[i].texture);
      uint32_t rt_params;

      if (!res) {
         context->framebuffer.rt_params[1 + i] = 0;
         context->framebuffer.bos[1 + i] = NULL;
         continue;
      }

      rt_params  = TGR3D_VAL(RT_PARAMS, FORMAT, res->format);
      rt_params |= TGR3D_VAL(RT_PARAMS, PITCH, res->pitch);
      rt_params |= TGR3D_BOOL(RT_PARAMS, TILED, res->tiled);

      context->framebuffer.rt_params[1 + i] = rt_params;
      context->framebuffer.bos[1 + i] = res->bo;
      mask |= 1 << (1 + i);
   }

   util_copy_framebuffer_state(&context->framebuffer.base, framebuffer);

   context->framebuffer.num_rts = 1 + i;
   context->framebuffer.mask = mask;

   /* prepare the scissor-registers for the non-scissor case */
   context->no_scissor[0]  = host1x_opcode_incr(TGR3D_SCISSOR_HORIZ, 2);
   context->no_scissor[1]  = TGR3D_VAL(SCISSOR_HORIZ, MIN, 0);
   context->no_scissor[1] |= TGR3D_VAL(SCISSOR_HORIZ, MAX, framebuffer->width);
   context->no_scissor[2]  = TGR3D_VAL(SCISSOR_VERT, MIN, 0);
   context->no_scissor[2] |= TGR3D_VAL(SCISSOR_VERT, MAX, framebuffer->height);
}

static void
grate_set_polygon_stipple(struct pipe_context *pcontext,
                          const struct pipe_poly_stipple *stipple)
{
   unimplemented();
}

static void
grate_set_scissor_states(struct pipe_context *pcontext,
                         unsigned start_slot,
                         unsigned num_scissors,
                         const struct pipe_scissor_state * scissors)
{
   assert(num_scissors == 1);
   grate_context(pcontext)->scissor = scissors[0];
}

static void
grate_set_viewport_states(struct pipe_context *pcontext,
                          unsigned start_slot,
                          unsigned num_viewports,
                          const struct pipe_viewport_state *viewports)
{
   struct grate_context *context = grate_context(pcontext);
   const float zeps = powf(2.0f, -21.0f);
   unsigned int hw_scale;

   if (context->tegra114)
      hw_scale = 0xFFFFFF;
   else
      hw_scale = 0xFFFFF;

   assert(num_viewports == 1);
   assert(start_slot == 0);

   context->viewport[0] = host1x_opcode_incr(TGR3D_VIEWPORT_X_BIAS, 6);
   context->viewport[1] = u_bitcast_f2u(viewports[0].translate[0] * 16.0f);
   context->viewport[2] = u_bitcast_f2u(viewports[0].translate[1] * 16.0f);
   context->viewport[3] = u_bitcast_f2u(viewports[0].translate[2] - zeps);
   context->viewport[4] = u_bitcast_f2u(viewports[0].scale[0] * 16.0f);
   context->viewport[5] = u_bitcast_f2u(viewports[0].scale[1] * 16.0f);
   context->viewport[6] = u_bitcast_f2u(viewports[0].scale[2] - zeps);

   uint32_t depth_near = (viewports[0].translate[2] - viewports[0].scale[2]) * hw_scale;
   uint32_t depth_far = (viewports[0].translate[2] + viewports[0].scale[2]) * hw_scale;
   context->viewport[7] = host1x_opcode_incr(TGR3D_DEPTH_RANGE_NEAR, 2);
   context->viewport[8] = depth_near;
   context->viewport[9] = depth_far;

   assert(viewports[0].scale[0] >= 0.0f);
   float max_x = fabs(viewports[0].translate[0]);
   float max_y = fabs(viewports[0].translate[1]);
   float scale_x = viewports[0].scale[0];
   float scale_y = fabs(viewports[0].scale[1]);
   context->guardband[0] = host1x_opcode_incr(TGR3D_GUARDBAND_WIDTH, 3);
   context->guardband[1] = u_bitcast_f2u((3967 - max_x) / scale_x);
   context->guardband[2] = u_bitcast_f2u((3967 - max_y) / scale_y);
   context->guardband[3] = u_bitcast_f2u(6.99);

   context->y_invert = viewports[0].scale[1] < 0.0f;
}

static void
grate_set_vertex_buffers(struct pipe_context *pcontext, unsigned int count,
                         const struct pipe_vertex_buffer *buffer)
{
   struct grate_context *context = grate_context(pcontext);
   struct grate_vertexbuf_state *vbs = &context->vbs;

   util_set_vertex_buffers_mask(vbs->vb, &vbs->enabled, buffer, count);
   vbs->count = util_last_bit(vbs->enabled);
}


static void
grate_set_sampler_views(struct pipe_context *pctx, mesa_shader_stage shader,
                        unsigned start_slot, unsigned num_views,
                        unsigned unbind_num_trailing_slots,
                        struct pipe_sampler_view **views)
{
   struct grate_context *context = grate_context(pctx);
   unsigned i;

   if (shader != MESA_SHADER_FRAGMENT)
      return;

   for (i = 0; i < num_views; i++) {
      struct pipe_sampler_view *view = views ? views[i] : NULL;
      pipe_sampler_view_reference(&context->fragment_views[start_slot + i], view);
   }
   for (; i < num_views + unbind_num_trailing_slots; i++)
      pipe_sampler_view_reference(&context->fragment_views[start_slot + i], NULL);

   context->num_fragment_views = 0;
   for (i = 0; i < PIPE_MAX_SHADER_SAMPLER_VIEWS; i++) {
      if (context->fragment_views[i])
         context->num_fragment_views = i + 1;
   }
}

static struct pipe_sampler_view *
grate_create_sampler_view(struct pipe_context *pctx, struct pipe_resource *texture,
                          const struct pipe_sampler_view *template)
{
   struct pipe_sampler_view *view = CALLOC_STRUCT(pipe_sampler_view);
   if (!view)
      return NULL;

   *view = *template;
   view->reference.count = 1;
   view->texture = NULL;
   pipe_resource_reference(&view->texture, texture);
   view->context = pctx;
   return view;
}

static void
grate_sampler_view_destroy(struct pipe_context *pctx, struct pipe_sampler_view *view)
{
   pipe_resource_reference(&view->texture, NULL);
   FREE(view);
}

static void
grate_set_blend_color(struct pipe_context *pctx,
                      const struct pipe_blend_color *blend_color)
{
}

static void
grate_set_stencil_ref(struct pipe_context *pctx,
                      const struct pipe_stencil_ref stencil_ref)
{
}

static void
grate_fp_state_bind(struct pipe_context *pctx, void *hwcso)
{
}

static void
grate_vp_state_bind(struct pipe_context *pctx, void *hwcso)
{
}

void
grate_context_state_init(struct pipe_context *pcontext)
{
   pcontext->set_blend_color = grate_set_blend_color;
   pcontext->set_stencil_ref = grate_set_stencil_ref;
   pcontext->bind_fs_state = grate_fp_state_bind;
   pcontext->bind_vs_state = grate_vp_state_bind;
   pcontext->set_sample_mask = grate_set_sample_mask;
   pcontext->set_constant_buffer = grate_set_constant_buffer;
   pcontext->set_framebuffer_state = grate_set_framebuffer_state;
   pcontext->set_polygon_stipple = grate_set_polygon_stipple;
   pcontext->set_scissor_states = grate_set_scissor_states;
   pcontext->set_viewport_states = grate_set_viewport_states;
   pcontext->set_sampler_views = grate_set_sampler_views;
   pcontext->create_sampler_view = grate_create_sampler_view;
   pcontext->sampler_view_destroy = grate_sampler_view_destroy;
   pcontext->sampler_view_release = u_default_sampler_view_release;
   pcontext->set_vertex_buffers = grate_set_vertex_buffers;
}

static void *
grate_create_blend_state(struct pipe_context *pcontext,
                         const struct pipe_blend_state *template)
{
   struct pipe_blend_state *so = CALLOC_STRUCT(pipe_blend_state);
   if (!so)
      return NULL;

   *so = *template;

   return so;
}

static void
grate_bind_blend_state(struct pipe_context *pcontext, void *so)
{
   /* applied in the fragment program, see grate_fs_variant() */
   grate_context(pcontext)->blend = so;
}

static void
grate_delete_blend_state(struct pipe_context *pcontext, void *so)
{
   FREE(so);
}

void
grate_context_blend_init(struct pipe_context *pcontext)
{
   pcontext->create_blend_state = grate_create_blend_state;
   pcontext->bind_blend_state = grate_bind_blend_state;
   pcontext->delete_blend_state = grate_delete_blend_state;
}

static void *
grate_create_sampler_state(struct pipe_context *pcontext,
            const struct pipe_sampler_state *template)
{
   struct pipe_sampler_state *so = CALLOC_STRUCT(pipe_sampler_state);
   if (!so)
      return NULL;

   *so = *template;

   return so;
}

static void
grate_bind_sampler_states(struct pipe_context *pcontext,
                          mesa_shader_stage shader, unsigned start_slot,
                          unsigned num_samplers, void **samplers)
{
   struct grate_context *context = grate_context(pcontext);

   if (shader != MESA_SHADER_FRAGMENT)
      return;

   for (unsigned i = 0; i < num_samplers; i++)
      context->fragment_samplers[start_slot + i] = samplers ? samplers[i] : NULL;

   context->num_fragment_samplers = 0;
   for (unsigned i = 0; i < PIPE_MAX_SAMPLERS; i++) {
      if (context->fragment_samplers[i])
         context->num_fragment_samplers = i + 1;
   }
}

static void
grate_delete_sampler_state(struct pipe_context *pcontext, void *so)
{
   FREE(so);
}

void
grate_context_sampler_init(struct pipe_context *pcontext)
{
   pcontext->create_sampler_state = grate_create_sampler_state;
   pcontext->bind_sampler_states = grate_bind_sampler_states;
   pcontext->delete_sampler_state = grate_delete_sampler_state;
}

static int
grate_cull_face(int cull_face, bool front_ccw)
{
   switch (cull_face) {
   case PIPE_FACE_NONE:
      return TGR3D_CULL_FACE_NONE;

   case PIPE_FACE_FRONT:
      return front_ccw ? TGR3D_CULL_FACE_CCW : TGR3D_CULL_FACE_CW;

   case PIPE_FACE_BACK:
      return front_ccw ? TGR3D_CULL_FACE_CW : TGR3D_CULL_FACE_CCW;

   case PIPE_FACE_FRONT_AND_BACK:
      return TGR3D_CULL_FACE_BOTH;

   default:
      UNREACHABLE("unknown cull_face");
   }
}

static void *
grate_create_rasterizer_state(struct pipe_context *pcontext,
                              const struct pipe_rasterizer_state *template)
{
   struct grate_rasterizer_state *so = CALLOC_STRUCT(grate_rasterizer_state);
   if (!so)
      return NULL;

   so->base = *template;

   so->draw_params = TGR3D_VAL(DRAW_PARAMS, PROVOKING_VERTEX, !template->flatshade_first);

   /* normal */
   so->cull_face[0] = TGR3D_BOOL(CULL_FACE_LINKER_SETUP, FRONT_CW,
                                 !template->front_ccw);
   so->cull_face[0] |= TGR3D_VAL(CULL_FACE_LINKER_SETUP, CULL_FACE,
                                 grate_cull_face(template->cull_face,
                                                 template->front_ccw));

   /* y-inverted */
   so->cull_face[1] = TGR3D_BOOL(CULL_FACE_LINKER_SETUP, FRONT_CW,
                                 template->front_ccw);
   so->cull_face[1] |= TGR3D_VAL(CULL_FACE_LINKER_SETUP, CULL_FACE,
                                 grate_cull_face(template->cull_face,
                                                 !template->front_ccw));

   return so;
}

static void
grate_bind_rasterizer_state(struct pipe_context *pcontext, void *so)
{
   grate_context(pcontext)->rast = so;
}

static void
grate_delete_rasterizer_state(struct pipe_context *pcontext, void *so)
{
   FREE(so);
}

void
grate_context_rasterizer_init(struct pipe_context *pcontext)
{
   pcontext->create_rasterizer_state = grate_create_rasterizer_state;
   pcontext->bind_rasterizer_state = grate_bind_rasterizer_state;
   pcontext->delete_rasterizer_state = grate_delete_rasterizer_state;
}

static int
grate_compare_func(enum pipe_compare_func func)
{
   switch (func) {
   case PIPE_FUNC_NEVER: return TGR3D_COMPARE_FUNC_NEVER;
   case PIPE_FUNC_LESS: return TGR3D_COMPARE_FUNC_LESS;
   case PIPE_FUNC_EQUAL: return TGR3D_COMPARE_FUNC_EQUAL;
   case PIPE_FUNC_LEQUAL: return TGR3D_COMPARE_FUNC_LEQUAL;
   case PIPE_FUNC_GREATER: return TGR3D_COMPARE_FUNC_GREATER;
   case PIPE_FUNC_NOTEQUAL: return TGR3D_COMPARE_FUNC_NOTEQUAL;
   case PIPE_FUNC_ALWAYS: return TGR3D_COMPARE_FUNC_ALWAYS;
   default: UNREACHABLE("unknown pipe_compare_func");
   }
}

static void *
grate_create_zsa_state(struct pipe_context *pcontext,
                       const struct pipe_depth_stencil_alpha_state *template)
{
   struct grate_context *context = grate_context(pcontext);
   struct grate_zsa_state *so = CALLOC_STRUCT(grate_zsa_state);
   if (!so)
      return NULL;

   so->base = *template;

   uint32_t depth_test = 0;
   depth_test |= TGR3D_VAL(DEPTH_TEST_PARAMS, FUNC,
                           grate_compare_func(template->depth_func));
   depth_test |= TGR3D_BOOL(DEPTH_TEST_PARAMS, DEPTH_TEST,
                            template->depth_enabled);
   depth_test |= TGR3D_BOOL(DEPTH_TEST_PARAMS, DEPTH_WRITE,
                            template->depth_writemask);
   depth_test |= 0x200;

   so->commands[0] = host1x_opcode_incr(TGR3D_DEPTH_TEST_PARAMS, 1);
   so->commands[1] = depth_test;
   so->num_commands = 2;

   if (context->tegra114) {
      so->commands[2] = host1x_opcode_incr(0xe45, 1);
      so->commands[3] = depth_test;
      so->num_commands = 4;
   }

   return so;
}

static void
grate_bind_zsa_state(struct pipe_context *pcontext, void *so)
{
   grate_context(pcontext)->zsa = so;
}

static void
grate_delete_zsa_state(struct pipe_context *pcontext, void *so)
{
   FREE(so);
}

void
grate_context_zsa_init(struct pipe_context *pcontext)
{
   pcontext->create_depth_stencil_alpha_state = grate_create_zsa_state;
   pcontext->bind_depth_stencil_alpha_state = grate_bind_zsa_state;
   pcontext->delete_depth_stencil_alpha_state = grate_delete_zsa_state;
}

/*
 * Note: this does not include the stride, which needs to be mixed in later
 **/
static uint32_t
attrib_mode(const struct pipe_vertex_element *e)
{
   const struct util_format_description *desc = util_format_description(e->src_format);
   const int c = util_format_get_first_non_void_channel(e->src_format);
   uint32_t type, format;

   assert(!desc->is_mixed);
   assert(c >= 0);

   switch (desc->channel[c].type) {
   case UTIL_FORMAT_TYPE_UNSIGNED:
   case UTIL_FORMAT_TYPE_SIGNED:
      switch (desc->channel[c].size) {
      case 8:
         type = TGR3D_ATTRIB_TYPE_UBYTE;
         break;

      case 16:
         type = TGR3D_ATTRIB_TYPE_USHORT;
         break;

      case 32:
         type = TGR3D_ATTRIB_TYPE_UINT;
         break;

      default:
         UNREACHABLE("invalid channel-size");
      }

      if (desc->channel[c].type == UTIL_FORMAT_TYPE_SIGNED)
         type += 2;

      if (desc->channel[c].normalized)
         type += 1;

      break;

   case UTIL_FORMAT_TYPE_FIXED:
      assert(desc->channel[c].size == 32);
      type = TGR3D_ATTRIB_TYPE_FIXED16;
      break;

   case UTIL_FORMAT_TYPE_FLOAT:
      assert(desc->channel[c].size == 32); /* TODO: float16 ? */
      type = TGR3D_ATTRIB_TYPE_FLOAT32;
      break;

   default:
      UNREACHABLE("invalid channel-type");
   }

   format  = TGR3D_VAL(ATTRIB_MODE, TYPE, type);
   format |= TGR3D_VAL(ATTRIB_MODE, SIZE, desc->nr_channels);
   return format;
}

static void *
grate_create_vertex_state(struct pipe_context *pcontext, unsigned int count,
                          const struct pipe_vertex_element *elements)
{
   unsigned int i;
   uint16_t mask = 0;
   struct grate_vertex_state *vtx = CALLOC_STRUCT(grate_vertex_state);
   if (!vtx)
      return NULL;

   for (i = 0; i < count; ++i) {
      const struct pipe_vertex_element *src = elements + i;
      struct grate_vertex_element *dst = vtx->elements + i;
      dst->attrib = attrib_mode(src);
      dst->buffer_index = src->vertex_buffer_index;
      dst->offset = src->src_offset;
      dst->stride = src->src_stride;
      mask |= 1 << i;
   }

   vtx->num_elements = count;
   vtx->mask = mask;

   return vtx;
}

static void
grate_bind_vertex_state(struct pipe_context *pcontext, void *so)
{
   grate_context(pcontext)->vs = so;
}

static void
grate_delete_vertex_state(struct pipe_context *pcontext, void *so)
{
   FREE(so);
}

static void
emit_attribs(struct grate_context *context)
{
   unsigned int i;
   struct grate_stream *stream = &context->gr3d->stream;

   assert(context->vs);

   for (i = 0; i < context->vs->num_elements; ++i) {
      const struct pipe_vertex_buffer *vb;
      const struct grate_vertex_element *e = context->vs->elements + i;
      const struct grate_resource *r;

      assert(e->buffer_index < context->vbs.count);
      vb = context->vbs.vb + e->buffer_index;
      assert(!vb->is_user_buffer);
      r = grate_resource(vb->buffer.resource);

      uint32_t attrib = e->attrib;
      assert(e->stride < 1 << 24);
      attrib |= TGR3D_VAL(ATTRIB_MODE, STRIDE, e->stride);

      grate_stream_push(stream, host1x_opcode_incr(TGR3D_ATTRIB_PTR(i), 2));
      grate_stream_push_reloc(stream, r->bo, vb->buffer_offset + e->offset);
      grate_stream_push(stream, attrib);
   }
}

static void
emit_render_targets(struct grate_context *context)
{
   unsigned int i;
   struct grate_stream *stream = &context->gr3d->stream;
   const struct grate_framebuffer_state *fb = &context->framebuffer;

   grate_stream_push(stream, host1x_opcode_incr(TGR3D_RT_PARAMS(0), fb->num_rts));
   for (i = 0; i < fb->num_rts; ++i) {
      uint32_t rt_params = fb->rt_params[i];
      /* TODO: setup dither */
      /* rt_params |= TGR3D_BOOL(RT_PARAMS, DITHER_ENABLE, enable_dither); */
      grate_stream_push(stream, rt_params);
   }

   /*
    * RT 0 is the depth buffer, absent without a zsbuf. The host1x firewall
    * only accepts relocated values in address registers, so skip it.
    */
   for (i = 0; i < fb->num_rts; ++i) {
      if (!fb->bos[i])
         continue;
      grate_stream_push(stream, host1x_opcode_incr(TGR3D_RT_PTR(i), 1));
      grate_stream_push_reloc(stream, fb->bos[i], 0);
   }

   grate_stream_push(stream, host1x_opcode_incr(TGR3D_RT_ENABLE, 1));
   grate_stream_push(stream, fb->mask);
}

static void
emit_scissor(struct grate_context *context)
{
   struct grate_stream *stream = &context->gr3d->stream;

   if (context->rast && context->rast->base.scissor) {
      const struct pipe_scissor_state *s = &context->scissor;
      unsigned w = context->framebuffer.base.width;
      unsigned h = context->framebuffer.base.height;
      uint32_t words[3] = {
         host1x_opcode_incr(TGR3D_SCISSOR_HORIZ, 2),
         TGR3D_VAL(SCISSOR_HORIZ, MIN, MIN2(s->minx, w)) |
         TGR3D_VAL(SCISSOR_HORIZ, MAX, MIN2(s->maxx, w)),
         TGR3D_VAL(SCISSOR_VERT, MIN, MIN2(s->miny, h)) |
         TGR3D_VAL(SCISSOR_VERT, MAX, MIN2(s->maxy, h)),
      };
      grate_stream_push_words(stream, words, 3, 0);
      return;
   }

   grate_stream_push_words(stream, context->no_scissor, 3, 0);
}

static void
emit_viewport(struct grate_context *context)
{
   struct grate_stream *stream = &context->gr3d->stream;
   grate_stream_push_words(stream, context->viewport, 10, 0);
}

static void
emit_guardband(struct grate_context *context)
{
   struct grate_stream *stream = &context->gr3d->stream;
   grate_stream_push_words(stream, context->guardband, 4, 0);
}

static void
emit_zsa_state(struct grate_context *context)
{
   struct grate_stream *stream = &context->gr3d->stream;
   grate_stream_push_words(stream, context->zsa->commands,
                           context->zsa->num_commands, 0);
}

static void
emit_vs_uniforms(struct grate_context *context)
{
   struct grate_stream *stream = &context->gr3d->stream;
   struct pipe_constant_buffer *constbuf = &context->constant_buffer[MESA_SHADER_VERTEX];
   struct grate_vertex_shader_state *vs = context->vshader;
   uint32_t consts[256 * 4] = { 0 };
   int len = 0;

   if (constbuf->user_buffer != NULL) {
      assert(constbuf->buffer_size % sizeof(uint32_t) == 0);

      len = constbuf->buffer_size / 4;
      assert(len < 256 * 4);
      memcpy(consts, constbuf->user_buffer, len * 4);
   }

   /* immediates live at the top of the uniform space, see vp/tgsi.c */
   for (int i = 0; i < vs->num_immediates; i++)
      memcpy(&consts[(255 - i) * 4], vs->immediates[i], 4 * sizeof(uint32_t));

   if (vs->num_immediates)
      len = 256 * 4;

   if (len) {
      grate_stream_push(stream, host1x_opcode_imm(TGR3D_VP_UPLOAD_CONST_ID, 0));
      grate_stream_push(stream, host1x_opcode_nonincr(TGR3D_VP_UPLOAD_CONST, len));
      grate_stream_push_words(stream, consts, len, 0);
   }
}

static uint32_t
float_to_fp20(float f)
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

/* the uniform registers a NIR-compiled fragment program reads, see fp/nir.c */
static void
emit_fs_uniforms(struct grate_context *context, const struct grate_fp_info *info)
{
   struct grate_stream *stream = &context->gr3d->stream;
   const struct pipe_constant_buffer *cb =
      &context->constant_buffer[MESA_SHADER_FRAGMENT];
   const float *data = cb->user_buffer;
   uint32_t regs[GRATE_FP_MAX_UNIFORMS];

   if (!info->num_uniforms)
      return;

   for (int i = 0; i < info->num_uniforms; i++) {
      uint32_t u = info->uniforms[i];
      uint32_t payload = GRATE_FP_UNIFORM_PAYLOAD(u);
      float f = 0.0f;
      switch (GRATE_FP_UNIFORM_KIND(u)) {
      case GRATE_FP_UNIFORM_KIND_CB:
         if (data && payload * 4 < cb->buffer_size)
            f = data[payload];
         break;
      case GRATE_FP_UNIFORM_KIND_IMM:
         f = uif(payload << 4);
         break;
      case GRATE_FP_UNIFORM_KIND_FB_Y:
         /*
          * pos.y counts render target rows when the viewport isn't flipped
          * (FBOs), and from the bottom when it is.
          */
         if (context->y_invert)
            f = payload ? 8191.0f + context->framebuffer.base.height : -1.0f;
         else
            f = payload ? -8192.0f : 1.0f;
         break;
      case GRATE_FP_UNIFORM_KIND_CB_PAIR: {
         /* two fx10 (1/256 steps, [-4, 4)) in the low and high halves */
         uint32_t lohi = 0;
         for (int h = 0; h < 2; h++) {
            uint32_t dw = (payload >> (14 * h)) & 0x3fff;
            float v = data && dw * 4 < cb->buffer_size ? data[dw] : 0.0f;
            int x = (int)lrintf(CLAMP(v, -4.0f, 1023.0f / 256.0f) * 256.0f);
            lohi |= (uint32_t)(x & 0x3ff) << (10 * h);
         }
         regs[i] = lohi;
         continue;
      }
      }
      regs[i] = float_to_fp20(f);
   }

   grate_stream_push(stream, host1x_opcode_incr(TGR3D_FP_CONST(0),
                                                info->num_uniforms));
   grate_stream_push_words(stream, regs, info->num_uniforms, 0);
}

static void
emit_shader(struct grate_stream *stream, const struct grate_shader_blob *blob)
{
   grate_stream_push_words(stream, blob->commands, blob->num_commands, 0);
}

static void
emit_program(struct grate_context *context)
{
   struct grate_stream *stream = &context->gr3d->stream;

   emit_shader(stream, &context->vshader->blob);
   emit_shader(stream, grate_fs_variant(context));
   emit_fs_uniforms(context, &context->fshader->info);

   uint32_t cull_face_linker_setup = TGR3D_VAL(CULL_FACE_LINKER_SETUP,
                                               UNK_18_31, 0x2e38);

   /* depends on cull-face */
   cull_face_linker_setup |= context->rast->cull_face[context->y_invert];

   /* depends on linking */
   struct grate_fp_info *info = &context->fshader->info;
   assert(info->num_inputs > 0);
   cull_face_linker_setup |= TGR3D_VAL(CULL_FACE_LINKER_SETUP,
                                       LINKER_INST_COUNT,
                                       info->num_inputs - 1);

   uint32_t linker_insts[3 + info->num_inputs * 2];
   linker_insts[0] = host1x_opcode_incr(TGR3D_CULL_FACE_LINKER_SETUP, 1);
   linker_insts[1] = cull_face_linker_setup;
   linker_insts[2] = host1x_opcode_incr(TGR3D_LINKER_INSTRUCTION(0), info->num_inputs * 2);

   for (int i = 0; i < info->num_inputs; ++i) {
      /*
       * Link from the vertex output with the same semantic. Exports are
       * numbered by rank among the enabled outputs (export 0 = position).
       */
      uint32_t src = info->inputs[i].src;
      struct grate_vertex_shader_state *vs = context->vshader;
      for (int o = 0; o < 16; o++) {
         if ((vs->output_mask & (1 << o)) && info->inputs[i].semantic &&
             vs->output_semantic[o] == info->inputs[i].semantic) {
            unsigned rank = util_bitcount(vs->output_mask & ((1u << o) - 1));
            src = (src & ~(0x1f << 3)) | rank << 3;
            break;
         }
      }
      linker_insts[3 + i * 2] = src;
      linker_insts[3 + i * 2 + 1] = info->inputs[i].dst;
   }

   if (context->rast->base.flatshade && info->color_input >= 0)
      linker_insts[3 + info->color_input * 2 + 1] |= 0xf << 16;

   grate_stream_push_words(stream, linker_insts, ARRAY_SIZE(linker_insts), 0);
}

static bool
grate_wrap_clamps(unsigned wrap)
{
   return wrap == PIPE_TEX_WRAP_CLAMP_TO_EDGE || wrap == PIPE_TEX_WRAP_CLAMP ||
          wrap == PIPE_TEX_WRAP_CLAMP_TO_BORDER;
}

static void
emit_textures(struct grate_context *context)
{
   struct grate_stream *stream = &context->gr3d->stream;

   for (unsigned i = 0; i < context->num_fragment_views; i++) {
      struct pipe_sampler_view *view = context->fragment_views[i];
      struct pipe_sampler_state *sampler = context->fragment_samplers[i];
      if (!view)
         continue;

      struct grate_resource *res = grate_resource(view->texture);
      unsigned width = view->texture->width0;
      unsigned height = view->texture->height0;
      bool linear_mag = sampler && sampler->mag_img_filter == PIPE_TEX_FILTER_LINEAR;
      bool linear_min = sampler && sampler->min_img_filter == PIPE_TEX_FILTER_LINEAR;
      bool clamp_s = !sampler || grate_wrap_clamps(sampler->wrap_s);
      bool clamp_t = !sampler || grate_wrap_clamps(sampler->wrap_t);

      uint32_t desc1 = TGR3D_VAL(TEXTURE_DESC1, FORMAT, res->format);
      desc1 |= TGR3D_BOOL(TEXTURE_DESC1, MAGFILTER_LINEAR, linear_mag);
      desc1 |= TGR3D_BOOL(TEXTURE_DESC1, MINFILTER_LINEAR_WITHIN, linear_min);
      desc1 |= TGR3D_BOOL(TEXTURE_DESC1, WRAP_S_CLAMP_TO_EDGE, clamp_s);
      desc1 |= TGR3D_BOOL(TEXTURE_DESC1, WRAP_T_CLAMP_TO_EDGE, clamp_t);
      if (sampler) {
         desc1 |= TGR3D_BOOL(TEXTURE_DESC1, WRAP_S_MIRRORED_REPEAT, sampler->wrap_s == PIPE_TEX_WRAP_MIRROR_REPEAT);
         desc1 |= TGR3D_BOOL(TEXTURE_DESC1, WRAP_T_MIRRORED_REPEAT, sampler->wrap_t == PIPE_TEX_WRAP_MIRROR_REPEAT);
      }

      uint32_t desc2 = TGR3D_BOOL(TEXTURE_DESC2, MIPMAP_DISABLE, 1);
      if (util_is_power_of_two_nonzero(width) && util_is_power_of_two_nonzero(height)) {
         desc2 |= TGR3D_VAL(TEXTURE_DESC2, WIDTH_LOG2, util_logbase2(width));
         desc2 |= TGR3D_VAL(TEXTURE_DESC2, HEIGHT_LOG2, util_logbase2(height));
      } else {
         desc2 |= TGR3D_BOOL(TEXTURE_DESC2, NOT_POW2_DIMENSIONS, 1);
         desc2 |= TGR3D_VAL(TEXTURE_DESC2, WIDTH, width);
         desc2 |= TGR3D_VAL(TEXTURE_DESC2, HEIGHT, height);
      }

      grate_stream_push(stream, host1x_opcode_incr(TGR3D_TEXTURE_DESC1(i), 2));
      grate_stream_push(stream, desc1);
      grate_stream_push(stream, desc2);
      grate_stream_push(stream, host1x_opcode_incr(TGR3D_TEXTURE_POINTER(i), 1));
      grate_stream_push_reloc(stream, res->bo, 0);
   }
}

void
grate_emit_state(struct grate_context *context)
{
   emit_render_targets(context);
   emit_viewport(context);
   emit_guardband(context);
   emit_scissor(context);
   emit_zsa_state(context);
   emit_attribs(context);
   emit_vs_uniforms(context);
   emit_program(context);
   emit_textures(context);
}

void
grate_context_vbo_init(struct pipe_context *pcontext)
{
   pcontext->create_vertex_elements_state = grate_create_vertex_state;
   pcontext->bind_vertex_elements_state = grate_bind_vertex_state;
   pcontext->delete_vertex_elements_state = grate_delete_vertex_state;
}
