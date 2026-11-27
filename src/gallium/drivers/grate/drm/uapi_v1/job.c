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

#include <errno.h>
#include <stdlib.h>
#include <string.h>

#include <xf86drm.h>

#include "uapi_v1.h"

int drm_tegra_job_add_reloc(struct drm_tegra_job *job,
			    const struct drm_tegra_reloc *reloc,
			    struct drm_tegra_bo *target)
{
	struct drm_tegra_reloc *relocs;
	struct drm_tegra_bo **bos;
	size_t size;

	size = (job->num_relocs + 1) * sizeof(*reloc);

	relocs = realloc(job->relocs, size);
	if (!relocs)
		return -ENOMEM;

	job->relocs = relocs;

	bos = realloc(job->reloc_bos, (job->num_relocs + 1) * sizeof(*bos));
	if (!bos)
		return -ENOMEM;

	job->reloc_bos = bos;
	job->reloc_bos[job->num_relocs] = target;

	job->relocs[job->num_relocs++] = *reloc;

	return 0;
}

int drm_tegra_job_add_cmdbuf(struct drm_tegra_job *job,
			     const struct drm_tegra_cmdbuf *cmdbuf)
{
	struct drm_tegra_cmdbuf *cmdbufs;
	size_t size;

	size = (job->num_cmdbufs + 1) * sizeof(*cmdbuf);

	cmdbufs = realloc(job->cmdbufs, size);
	if (!cmdbufs)
		return -ENOMEM;

	cmdbufs[job->num_cmdbufs++] = *cmdbuf;
	job->cmdbufs = cmdbufs;

	return 0;
}

int drm_tegra_job_new(struct drm_tegra_job **jobp,
		      struct drm_tegra_channel *channel)
{
	struct drm_tegra_job *job;

	if (!jobp || !channel)
		return -EINVAL;

	job = calloc(1, sizeof(*job));
	if (!job)
		return -ENOMEM;

	list_inithead(&job->pushbufs);
	job->channel = channel;
	job->syncpt = channel->syncpt;

	*jobp = job;

	return 0;
}

int drm_tegra_job_free(struct drm_tegra_job *job)
{
	struct drm_tegra_pushbuf_private *pushbuf;
	struct drm_tegra_pushbuf_private *temp;

	if (!job)
		return -EINVAL;

	LIST_FOR_EACH_ENTRY_SAFE(pushbuf, temp, &job->pushbufs, list)
		drm_tegra_pushbuf_free(&pushbuf->base);

	free(job->cmdbufs);
	free(job->relocs);
	free(job->reloc_bos);
	free(job);

	return 0;
}

/*
 * The channel UAPI: the kernel copies the commands from memory, patches in
 * the addresses of the BOs' mappings and returns the syncpoint threshold the
 * job reaches.
 */
static int drm_tegra_job_submit_uapi(struct drm_tegra_job *job,
				     struct drm_tegra_fence **fencep)
{
	struct drm_tegra_pushbuf_private *pushbuf = job->pushbuf;
	struct drm_tegra_channel *channel = job->channel;
	struct drm_tegra *drm = channel->drm;
	struct drm_tegra_channel_submit args;
	struct drm_tegra_submit_buf *bufs;
	struct drm_tegra_submit_cmd *cmds;
	struct drm_tegra_fence *fence = NULL;
	unsigned int words, num_cmds, i;
	int err;

	words = pushbuf && pushbuf->words ? pushbuf->base.ptr - pushbuf->start : 0;
	if (!words)
		return -EINVAL;

	/* a gather is at most 16383 words */
	num_cmds = DIV_ROUND_UP(words, 16383);
	cmds = calloc(num_cmds, sizeof(*cmds));
	bufs = calloc(job->num_relocs ? job->num_relocs : 1, sizeof(*bufs));
	if (fencep)
		fence = calloc(1, sizeof(*fence));
	if (!cmds || !bufs || (fencep && !fence)) {
		err = -ENOMEM;
		goto out;
	}

	for (i = 0; i < num_cmds; i++) {
		cmds[i].type = DRM_TEGRA_SUBMIT_CMD_GATHER_UPTR;
		cmds[i].gather_uptr.words = MIN2(words - i * 16383, 16383);
	}

	for (i = 0; i < job->num_relocs; i++) {
		err = drm_tegra_bo_channel_map(job->reloc_bos[i], channel,
					       &bufs[i].mapping);
		if (err < 0)
			goto out;

		bufs[i].reloc.target_offset = job->relocs[i].target.offset;
		bufs[i].reloc.gather_offset_words = job->relocs[i].cmdbuf.offset / 4;
		bufs[i].reloc.shift = job->relocs[i].shift;
	}

	memset(&args, 0, sizeof(args));
	args.context = channel->context;
	args.num_bufs = job->num_relocs;
	args.num_cmds = num_cmds;
	args.gather_data_words = words;
	args.bufs_ptr = (uintptr_t)bufs;
	args.cmds_ptr = (uintptr_t)cmds;
	args.gather_data_ptr = (uintptr_t)pushbuf->start;
	args.syncpt.id = job->syncpt;
	args.syncpt.increments = job->increments;

	err = drm_tegra_ioctl(drm->fd, DRM_IOCTL_TEGRA_CHANNEL_SUBMIT, &args);
	if (err < 0)
		goto out;

	if (fence) {
		fence->drm = drm;
		fence->syncpt = job->syncpt;
		fence->value = args.syncpt.value;
		fence->channel_uapi = true;
		*fencep = fence;
		fence = NULL;
	}

out:
	free(fence);
	free(bufs);
	free(cmds);
	return err;
}

int drm_tegra_job_submit(struct drm_tegra_job *job,
			 struct drm_tegra_fence **fencep)
{
	struct drm_tegra *drm;
	struct drm_tegra_fence *fence = NULL;
	struct drm_tegra_syncpt *syncpts;
	struct drm_tegra_submit args;
	int err;

	if (!job)
		return -EINVAL;

	if (job->channel->channel_uapi)
		return drm_tegra_job_submit_uapi(job, fencep);

	/*
	 * Make sure the current command stream buffer is queued for
	 * submission.
	 */
	err = drm_tegra_pushbuf_queue(job->pushbuf);
	if (err < 0)
		return err;

	job->pushbuf = NULL;

	if (fencep) {
		fence = calloc(1, sizeof(*fence));
		if (!fence)
			return -ENOMEM;
	}

	syncpts = calloc(1, sizeof(*syncpts));
	if (!syncpts) {
		free(fence);
		return -ENOMEM;
	}

	syncpts[0].id = job->syncpt;
	syncpts[0].incrs = job->increments;

	memset(&args, 0, sizeof(args));
	args.context = job->channel->context;
	args.num_syncpts = 1;
	args.num_cmdbufs = job->num_cmdbufs;
	args.num_relocs = job->num_relocs;
	args.num_waitchks = 0;
	args.waitchk_mask = 0;
	args.timeout = 1000;

	args.syncpts = (uintptr_t)syncpts;
	args.cmdbufs = (uintptr_t)job->cmdbufs;
	args.relocs = (uintptr_t)job->relocs;
	args.waitchks = 0;

	drm = job->channel->drm;
	err = drmCommandWriteRead(drm->fd, DRM_TEGRA_SUBMIT, &args,
				  sizeof(args));
	if (err < 0) {
		free(syncpts);
		free(fence);
		return err;
	}

	if (fence) {
		fence->syncpt = job->syncpt;
		fence->value = args.fence;
		fence->drm = drm;
		*fencep = fence;
	}

	free(syncpts);

	return 0;
}
