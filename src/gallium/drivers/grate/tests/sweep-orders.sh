#!/bin/sh
# sweep-orders.sh [shader_runner_gles2] [tests...]: run each shader test
# under each fragment-compiler candidate order alone (GRATE_FP_ORDERS), and
# print the wrong results. A test an order can't compile at all is drawn as
# nothing and fails too: those are reported as "cannot compile", not as
# wrong results. Needs MESA_SHADER_CACHE_DISABLE so each order compiles.
RUNNER=${1:-shader_runner_gles2}
[ $# -gt 0 ] && shift
cd "$(dirname "$0")"
TESTS=${*:-*.shader_test}
wrong=0
for o in $(seq 0 15); do
  for t in $TESTS; do
    out=$(GRATE_FP_ORDERS=$((1 << o)) MESA_SHADER_CACHE_DISABLE=1 \
          PIGLIT_PLATFORM=${PIGLIT_PLATFORM:-surfaceless_egl} \
          timeout 30 "$RUNNER" "$t" -auto -fbo 2>&1)
    case "$out" in
    *'"result": "pass"'*) ;;
    *'not supported'*) echo "order $o $t: cannot compile" ;;
    *) echo "order $o $t: WRONG"; wrong=$((wrong + 1)) ;;
    esac
  done
done
echo "$wrong wrong results"
[ $wrong -eq 0 ]
