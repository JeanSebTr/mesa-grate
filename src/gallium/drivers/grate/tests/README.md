# grate fragment compiler tests

piglit `shader_runner` tests (GLES 2), each drawing one quad and probing
pixels. They pass on llvmpipe; on GR3D all but two pass: `skia-blur`
(drawn in three passes) is one tap off at a probe that sits exactly on a
texel edge (coordinates go through fp16 spill slots), and
`skia-blur-decal` doesn't compile yet (it runs out of spill slots).

Run on the device with a GLES 2 build of piglit on surfaceless EGL:

    PIGLIT_PLATFORM=surfaceless_egl shader_runner_gles2 texture.shader_test -auto -fbo

`texture-blend` needs fixed-function blending in shader_runner, which it
doesn't have yet: `shader_runner-blend.patch` adds `enable GL_BLEND` and
`blend func <src> <dst>`.

- `texture*`, `uniform-colour`, `varying`: the basics (sampling,
  texture x colour, src-over blending).
- `alu-*`, `sfu-*`: ALU operations, modifiers, select, and the MFU special
  functions.
- `fragcoord*`: `gl_FragCoord`, and rounded-rect distance maths on it.
- `uniform-array`: a blur-sized uniform array (16 weights), spilled.
- `texture-sum-*`: texture taps summed, more live values than the four
  row registers: register spilling to render targets.
- `skia-*`: fragment shaders from Skia's GL backend (antialiased quad,
  rounded-rect clip, gaussian blur, and the blur with decal edges,
  quadratic-edge coverage with derivatives);
  `skia-aa-quad-vs` uses Skia's vertex shader and attribute layout too.

Avoid running untested programs while a compositor is drawing: a bad
program can hang GR3D, and only a reboot clears that.
