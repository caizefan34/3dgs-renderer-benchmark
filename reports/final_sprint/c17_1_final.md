# C17-1 final status

## Verdict

**PROMISING_BUT_INCOMPLETE.** The exact-allocation implementation and opt-in
mode are supplied as a reproducible patch, but no compiled renderer result is
available. It must not be called an optimization.

## Limiting factor

The A100's resident gsplat 1.5.3 environment has a broken `torch` install. A
temporary torch recovery made CUDA available, but its CUDA 12.4 runtime and
available CCCL headers were incompatible during full gsplat JIT compilation.
The failure occurred in the surrounding extension build, before any C17
renderer validation.

Implementation and evidence only were committed as `ec6bbf4`; unrelated dirty
and untracked workspace content was left untouched.
