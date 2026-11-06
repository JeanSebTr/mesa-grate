#include <stdio.h>

#include "drm-uapi/drm_fourcc.h"

#include <inttypes.h>
#include <xf86drm.h>

#include "compiler/nir/nir.h"
#include "util/os_file.h"
#include "util/u_memory.h"
#include "util/u_screen.h"

#include "grate_common.h"
#include "grate_context.h"
#include "grate_resource.h"
#include "grate_screen.h"

#include "opentegra_lib.h"

static const struct debug_named_value debug_options[] = {
   { "unimplemented", GRATE_DEBUG_UNIMPLEMENTED,
     "Print unimplemented functions" },
   { "tgsi", GRATE_DEBUG_TGSI,
     "Dump TGSI during program compile" },
   { "fs", GRATE_DEBUG_FS,
     "Print fragment shaders as NIR (all of them, or with the failure for unsupported ones)" },
   { NULL }
};

DEBUG_GET_ONCE_FLAGS_OPTION(grate_debug, "GRATE_DEBUG", debug_options, 0)
uint32_t grate_debug;

static void
grate_screen_destroy(struct pipe_screen *pscreen)
{
   struct grate_screen *screen = grate_screen(pscreen);

   slab_destroy_parent(&screen->transfer_pool);

   drm_tegra_close(screen->drm);
   FREE(screen);
}

static const char *
grate_screen_get_name(struct pipe_screen *pscreen)
{
   return "Tegra";
}

static const char *
grate_screen_get_vendor(struct pipe_screen *pscreen)
{
   return "Grate";
}

static const char *
grate_screen_get_device_vendor(struct pipe_screen *pscreen)
{
   return "NVIDIA";
}

/* Total CMA, which is where every GR3D buffer lives. */
static bool
grate_get_total_cma(uint64_t *size)
{
   char *meminfo = os_read_file("/proc/meminfo", NULL);
   if (!meminfo)
      return false;

   uint64_t kb = 0;
   char *str = strstr(meminfo, "CmaTotal:");
   bool ok = str && sscanf(str, "CmaTotal: %" SCNu64, &kb) == 1 && kb;
   free(meminfo);

   *size = kb << 10;
   return ok;
}

static void
grate_init_screen_caps(struct pipe_screen *pscreen)
{
   struct pipe_caps *caps = (struct pipe_caps *)&pscreen->caps;

   u_init_pipe_screen_caps(pscreen, 1);

   caps->clear_scissored = true;
   caps->npot_textures = true; /* not really, but mesa requires it for now! */
   caps->max_render_targets = 8; /* ??? */
   caps->max_texture_2d_size = 2048;
   caps->max_texture_3d_levels = 0;
   caps->max_texture_cube_levels = 16; /* ??? */
   caps->supported_prim_modes = BITFIELD_BIT(MESA_PRIM_TRIANGLES);
   caps->supported_prim_modes_with_restart = 0;
   caps->blend_equation_separate = true;
   /* well, not quite. but perhaps close enough? */
   caps->fragment_shader_texture_lod = true;
   caps->fragment_shader_derivatives = true;
   caps->min_texel_offset = 0;
   caps->max_texel_offset = 0;
   caps->max_stream_output_separate_components = 0;
   caps->max_stream_output_interleaved_components = 0;
   caps->vertex_color_unclamped = true; /* probably irrelevant for GLES2 */
   caps->glsl_feature_level = 120; /* no clue */
   caps->constant_buffer_offset_alignment = 4; /* DWORD, pure data GATHER */
   caps->min_map_buffer_alignment = 0;
   caps->texture_buffer_offset_alignment = 0;
   caps->max_texel_buffer_elements = 0;
   caps->texture_transfer_modes = PIPE_TEXTURE_TRANSFER_BLIT;
   caps->max_viewports = 1;
   caps->mixed_framebuffer_sizes = true;
   caps->buffer_map_persistent_coherent = false;
   caps->min_texture_gather_offset = 0;
   caps->max_texture_gather_offset = 0;
   caps->vendor_id = 0x10de;
   caps->device_id = 0xffffffff;
   caps->uma = true;
   caps->max_vertex_attrib_stride = (1 << 24) - 1;
   caps->shareable_shaders = true;
   caps->viewport_subpixel_bits = 4; /* minimum for GLES 2.0, might be more */
   caps->mixed_color_depth_bits = 1;
   caps->mixed_colorbuffer_formats = true;
   caps->allow_mapped_buffers_during_execution = false;
   caps->max_varyings = 16;
   /* gl_FragCoord from the position registers, see fp/nir.c */
   caps->fs_position_is_sysval = true;
   caps->fs_coord_origin_upper_left = true;
   caps->fs_coord_pixel_center_half_integer = true;
   /* resource_get_handle/from_handle do both (EGL and GBM) */
   caps->dmabuf = DRM_PRIME_CAP_IMPORT | DRM_PRIME_CAP_EXPORT;

   uint64_t cma;
   caps->video_memory = grate_get_total_cma(&cma) ? cma >> 20 : 0;

   caps->min_line_width =
   caps->min_line_width_aa =
   caps->min_point_size =
   caps->min_point_size_aa = 1;
   caps->point_size_granularity =
   caps->line_width_granularity = 0.1f;
   caps->max_line_width =
   caps->max_line_width_aa =
   caps->max_point_size =
   caps->max_point_size_aa = 8192.0f; /* no clue */
   caps->max_texture_lod_bias = 16.0f;
}

static void
grate_init_shader_caps(struct pipe_screen *pscreen)
{
   struct pipe_shader_caps *caps =
      (struct pipe_shader_caps *)&pscreen->shader_caps[MESA_SHADER_VERTEX];

   /* no vertex texturing, no control flow */
   caps->max_instructions =
   caps->max_alu_instructions = 1024;
   caps->max_inputs =
   caps->max_outputs = 16;
   caps->max_const_buffer0_size = 1024;
   caps->max_const_buffers = 1;
   caps->max_temps = 64 * 4; /* 64 vec4s */
   caps->indirect_const_addr = true; /* can index constant registers */
   caps->tgsi_sqrt_supported = true;
   caps->supported_irs = (1 << PIPE_SHADER_IR_NIR) | (1 << PIPE_SHADER_IR_TGSI);

   caps = (struct pipe_shader_caps *)&pscreen->shader_caps[MESA_SHADER_FRAGMENT];

   /* no control flow, no indirection */
   caps->max_instructions =
   caps->max_alu_instructions = 4 * 128;
   caps->max_tex_instructions =
   caps->max_tex_indirections = 128;
   caps->max_inputs =
   caps->max_outputs = 16;
   /* 32 uniform registers, but the compiler only uploads what it uses */
   caps->max_const_buffer0_size = 64 * 4 * sizeof(float);
   caps->max_const_buffers = 1;
   caps->max_temps = 16; /* scalars */
   caps->max_texture_samplers =
   caps->max_sampler_views = 16;
   caps->tgsi_sqrt_supported = true;
   caps->supported_irs = (1 << PIPE_SHADER_IR_NIR) | (1 << PIPE_SHADER_IR_TGSI);
}

/* What the state tracker lowers before grate turns NIR into TGSI. */
static const nir_shader_compiler_options grate_nir_options = {
   .fuse_ffma32 = true,
   .lower_bitops = true,
   .lower_extract_byte = true,
   .lower_extract_word = true,
   .lower_fdiv = true,
   .lower_fsat = true,
   .lower_insert_byte = true,
   .lower_insert_word = true,
   .lower_fdph = true,
   .lower_flrp32 = true,
   .lower_fmod = true,
   .lower_fpow = true,
   .lower_uniforms_to_ubo = true,
   .force_indirect_unrolling = nir_var_all,
   .force_indirect_unrolling_sampler = true,
   .max_unroll_iterations = 32,
   .no_integers = true,
};


static bool
grate_screen_is_format_supported(struct pipe_screen *pscreen,
                                 enum pipe_format format,
                                 enum pipe_texture_target target,
                                 unsigned sample_count,
                                 unsigned storage_sample_count,
                                 unsigned usage)
{
   /*
    * Combined depth/stencil isn't a GR3D format, but some EGL clients
    * insist on a stencil-8 config. Expose it; the buffer is never bound,
    * so depth and stencil tests are no-ops.
    */
   if ((usage & PIPE_BIND_DEPTH_STENCIL) &&
       (format == PIPE_FORMAT_Z24_UNORM_S8_UINT ||
        format == PIPE_FORMAT_S8_UINT_Z24_UNORM))
      return true;

   if (usage & (PIPE_BIND_RENDER_TARGET | PIPE_BIND_DEPTH_STENCIL |
                PIPE_BIND_SAMPLER_VIEW)) {
      if (grate_pixel_format(format) < 0)
         return false;
   }

   return true;
}

static void
grate_screen_fence_reference(struct pipe_screen *pscreen,
                             struct pipe_fence_handle **ptr,
                             struct pipe_fence_handle *fence)
{
   /* fences are a static dummy, see grate_context_flush() */
   *ptr = fence;
}

static bool
grate_screen_fence_finish(struct pipe_screen *screen,
                          struct pipe_context *ctx,
                          struct pipe_fence_handle *fence,
                          uint64_t timeout)
{
   return true;
}

static const uint64_t grate_available_modifiers[] = {
   DRM_FORMAT_MOD_LINEAR,
};

static void
grate_screen_query_dmabuf_modifiers(struct pipe_screen *pscreen,
                                    enum pipe_format format, int max,
                                    uint64_t *modifiers,
                                    unsigned int *external_only,
                                    int *count)
{
   int num_modifiers = ARRAY_SIZE(grate_available_modifiers);

   if (!modifiers) {
      *count = num_modifiers;
      return;
   }

   *count = MIN2(max, num_modifiers);
   for (int i = 0; i < *count; i++) {
      modifiers[i] = grate_available_modifiers[i];
      if (external_only)
         external_only[i] = false;
   }
}

static bool
grate_screen_is_dmabuf_modifier_supported(struct pipe_screen *pscreen,
                                          uint64_t modifier,
                                          enum pipe_format format,
                                          bool *external_only)
{
   for (int i = 0; i < ARRAY_SIZE(grate_available_modifiers); i++) {
      if (grate_available_modifiers[i] == modifier) {
         if (external_only)
            *external_only = false;

         return true;
      }
   }

   return false;
}

struct pipe_screen *
grate_screen_create(int fd)
{
   struct drm_tegra_channel *drm_channel;
   struct grate_screen *screen;
   int err;

   screen = CALLOC_STRUCT(grate_screen);
   if (!screen)
      return NULL;

   err = drm_tegra_new(&screen->drm, fd);
   if (err) {
      FREE(screen);
      return NULL;
   }

   err = drm_tegra_channel_open(&drm_channel, screen->drm, DRM_TEGRA_GR3D);
   if (err) {
      drm_tegra_close(screen->drm);
      FREE(screen);
      return NULL;
   }

   drm_tegra_channel_close(drm_channel);

   grate_debug = debug_get_option_grate_debug();

   screen->base.destroy = grate_screen_destroy;
   screen->base.get_name = grate_screen_get_name;
   screen->base.get_vendor = grate_screen_get_vendor;
   screen->base.get_device_vendor = grate_screen_get_device_vendor;
   screen->base.context_create = grate_screen_context_create;
   screen->base.is_format_supported = grate_screen_is_format_supported;
   screen->base.query_dmabuf_modifiers = grate_screen_query_dmabuf_modifiers;
   screen->base.is_dmabuf_modifier_supported = grate_screen_is_dmabuf_modifier_supported;

   /* fence functions */
   screen->base.fence_reference = grate_screen_fence_reference;
   screen->base.fence_finish = grate_screen_fence_finish;

   for (unsigned i = 0; i <= MESA_SHADER_COMPUTE; i++)
      screen->base.nir_options[i] = &grate_nir_options;

   grate_init_screen_caps(&screen->base);
   grate_init_shader_caps(&screen->base);

   grate_screen_resource_init(&screen->base);

   slab_create_parent(&screen->transfer_pool, sizeof(struct pipe_transfer), 16);

   return &screen->base;
}
