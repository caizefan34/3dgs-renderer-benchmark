#!/bin/bash
# Simple g++-10 wrapper: use system include-fixed, gcc-10 C++ headers
export COMPILER_PATH=/tmp/gcc10/usr/lib/gcc/x86_64-linux-gnu/10:/tmp/gcc10/usr/lib/gcc/x86_64-linux-gnu:/tmp/gcc10/usr/lib/gcc
exec /tmp/gcc10/usr/bin/g++-10 \
  -isystem /tmp/gcc10/usr/include/c++/10 \
  -isystem /tmp/gcc10/usr/include/c++/10/x86_64-linux-gnu \
  -isystem /tmp/gcc10/usr/include/x86_64-linux-gnu \
  -isystem /tmp/gcc10/usr/include \
  -isystem /usr/lib/gcc/x86_64-linux-gnu/11/include-fixed \
  -isystem /usr/lib/gcc/x86_64-linux-gnu/11/include \
  "$@"
