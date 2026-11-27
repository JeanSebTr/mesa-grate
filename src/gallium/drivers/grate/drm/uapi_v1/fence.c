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
#include <string.h>
#include <stdint.h>
#include <time.h>

#include <xf86drm.h>

#include "uapi_v1.h"

int drm_tegra_fence_wait_timeout(struct drm_tegra_fence *fence,
				 unsigned long timeout)
{
	struct drm_tegra_syncpt_wait args;

	if (!fence)
		return -EINVAL;

	if (fence->channel_uapi) {
		struct drm_tegra_syncpoint_wait wait;
		struct timespec now;
		int err;

		/* an absolute CLOCK_MONOTONIC deadline; ~0: wait for ever */
		memset(&wait, 0, sizeof(wait));
		wait.id = fence->syncpt;
		wait.threshold = fence->value;
		if (timeout == ~0ul) {
			wait.timeout_ns = INT64_MAX;
		} else {
			clock_gettime(CLOCK_MONOTONIC, &now);
			wait.timeout_ns = (int64_t)now.tv_sec * 1000000000ll +
					  now.tv_nsec + (int64_t)timeout * 1000000ll;
		}

		err = drm_tegra_ioctl(fence->drm->fd,
				      DRM_IOCTL_TEGRA_SYNCPOINT_WAIT, &wait);
		/* a wait that times out reports -EAGAIN, as the legacy one */
		return err == -ETIMEDOUT ? -EAGAIN : err;
	}

	memset(&args, 0, sizeof(args));
	args.id = fence->syncpt;
	args.thresh = fence->value;
	args.timeout = timeout;

	return drmCommandWriteRead(fence->drm->fd, DRM_TEGRA_SYNCPT_WAIT,
				   &args, sizeof(args));
}

struct drm_tegra_fence *drm_tegra_fence_dup(const struct drm_tegra_fence *fence)
{
	struct drm_tegra_fence *dup = malloc(sizeof(*dup));

	if (dup)
		*dup = *fence;
	return dup;
}

void drm_tegra_fence_free(struct drm_tegra_fence *fence)
{
	free(fence);
}
