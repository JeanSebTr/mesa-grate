# grate fragment compiler tests

piglit `shader_runner` tests (GLES 2), each drawing one quad and probing
pixels. They pass on llvmpipe; on GR3D all but two pass: `skia-blur`
and `skia-blur-decal` (drawn in several passes) are one tap off at a
probe that sits exactly on a texel edge (coordinates go through fp16
spill slots).

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
  quadratic-edge coverage with derivatives, a multi-stop gradient);
  `skia-aa-quad-vs` and `skia-gradient` use Skia's vertex shader and
  attribute layout too. `skia-gradient-grid`: the gradient probed on a 20 px grid, 133 points; `skia-gradient-blend`: the same, translucent and blended src-over (a blended multi-pass 4-row program). `skia-rrect-blur-clip*`: an analytic blurred
  rounded rect under a rounded-rect clip-out, alone and blended: enough
  uniforms to fill the table past its 41st entry, with `gl_FragCoord`.

Avoid running untested programs while a compositor is drawing: a bad
program can hang GR3D, and only a reboot clears that.
