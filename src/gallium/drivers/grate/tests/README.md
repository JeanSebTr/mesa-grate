# grate fragment compiler tests

piglit `shader_runner` tests (GLES 2), each drawing one quad and probing
pixels. They pass on llvmpipe; on GR3D all but `uniform-array` and
`skia-blur` pass (those need more than the four row registers).

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
- `uniform-array`: a blur-sized uniform array (16 weights).
- `skia-*`: fragment shaders from Skia's GL backend (antialiased quad,
  rounded-rect clip, gaussian blur); `skia-aa-quad-vs` uses Skia's vertex
  shader and attribute layout too.

Avoid running untested programs while a compositor is drawing: a bad
program can hang GR3D, and only a reboot clears that.
