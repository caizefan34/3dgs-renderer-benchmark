#!/bin/bash
# gcc-10 wrapper without gcc-11 include-fixed (causes keylockerintrin.h errors)
# Instead, create a minimal limits.h fix
export COMPILER_PATH=/tmp/gcc10/usr/lib/gcc/x86_64-linux-gnu/10:/tmp/gcc10/usr/lib/gcc/x86_64-linux-gnu:/tmp/gcc10/usr/lib/gcc
exec /tmp/gcc10/usr/bin/gcc-10 \
  -isystem /tmp/gcc10/usr/include/c++/10 \
  -isystem /tmp/gcc10/usr/include/c++/10/x86_64-linux-gnu \
  -isystem /tmp/gcc10/usr/include/x86_64-linux-gnu \
  -isystem /tmp/gcc10/usr/include \
  -isystem /usr/include \
  -B/tmp/gcc10/usr/lib/gcc/x86_64-linux-gnu/10 \
  "$@"
