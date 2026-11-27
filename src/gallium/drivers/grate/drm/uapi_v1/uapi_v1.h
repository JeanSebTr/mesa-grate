/*
 * Copyright © 2012, 2013 Thierry Reding
 * Copyright © 2013 Erik Faye-Lund
 * Copyright © 2014 NVIDIA Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE COPYRIGHT HOLDER(S) OR AUTHOR(S) BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef __DRM_TEGRA_PRIVATE_UAPI_V1_H__
#define __DRM_TEGRA_PRIVATE_UAPI_V1_H__ 1

#include "private.h"

struct drm_tegra_channel {
	struct drm_tegra *drm;
	enum host1x_class class;
	uint64_t context;
	uint32_t syncpt;

	/* channel UAPI: entry in drm->channels, and the ID BO mappings use */
	bool channel_uapi;
	struct list_head link;
	uint32_t serial;
};

struct drm_tegra_fence {
	struct drm_tegra *drm;
	uint32_t syncpt;
	uint32_t value;
	bool channel_uapi;
};

struct drm_tegra_pushbuf_private {
	struct drm_tegra_pushbuf base;
	struct drm_tegra_job *job;
	struct list_head list;
	struct list_head bos;

	struct drm_tegra_bo *bo;
	uint32_t *start;
	uint32_t *end;
	/* channel UAPI: the commands, in memory the kernel copies them from */
	uint32_t *words;
};

static inline struct drm_tegra_pushbuf_private *
drm_tegra_pushbuf(struct drm_tegra_pushbuf *pb)
{
	struct drm_tegra_pushbuf_private *priv;
	priv = container_of(pb, struct drm_tegra_pushbuf_private, base);
	return priv;
}

int drm_tegra_pushbuf_queue(struct drm_tegra_pushbuf_private *pushbuf);

struct drm_tegra_job {
	struct drm_tegra_channel *channel;

	unsigned int increments;
	uint32_t syncpt;

	struct drm_tegra_reloc *relocs;
	unsigned int num_relocs;
	/* channel UAPI: the target BO of each relocation */
	struct drm_tegra_bo **reloc_bos;
	/* channel UAPI: syncobjs to wait for and to put the job's fence in */
	uint32_t syncobj_in, syncobj_out;

	struct drm_tegra_cmdbuf *cmdbufs;
	unsigned int num_cmdbufs;

	struct drm_tegra_pushbuf_private *pushbuf;
	struct list_head pushbufs;
};

int drm_tegra_job_add_reloc(struct drm_tegra_job *job,
			    const struct drm_tegra_reloc *reloc,
			    struct drm_tegra_bo *target);
int drm_tegra_job_add_cmdbuf(struct drm_tegra_job *job,
			     const struct drm_tegra_cmdbuf *cmdbuf);

#endif /* __DRM_TEGRA_PRIVATE_UAPI_V1_H__ */
