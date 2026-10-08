#!/bin/bash
# P16-RB step 2: clone exact pinned upstream sources (identity freeze; no builds).
set -x
SRC=/root/epic05_p16rb/src
mkdir -p "$SRC"
export GIT_TERMINAL_PROMPT=0

clone_pin () {  # url dir commit
  local url="$1" dir="$2" commit="$3"
  cd "$SRC" || return 1
  if [ ! -d "$dir/.git" ]; then
    git clone "$url" "$dir" || { echo "CLONE_FAIL $dir"; return 1; }
  fi
  cd "$dir" || return 1
  git fetch --all --tags --quiet
  git checkout --detach "$commit" || { echo "CHECKOUT_FAIL $dir"; return 1; }
  echo "===== $dir ====="
  git rev-parse HEAD
  git config --get remote.origin.url
  echo "-- worktree status (must be empty) --"
  git status --porcelain=v1
  echo "-- submodule init --"
  git submodule update --init --recursive --quiet
  echo "-- submodule status --"
  git submodule status --recursive
  echo "-- branches containing HEAD --"
  git branch -a --contains HEAD | head -20
  echo "===== end $dir ====="
}

clone_pin https://github.com/nerficg-project/faster-gaussian-splatting fastergs_main 117bdbacfd7df6c1f06d9d959b1f5ab02d4d3847
clone_pin https://github.com/nerficg-project/faster-gaussian-splatting fastergs_fused ae665af2f91a4973bb92c40fdc4cba8d12fe3c96
clone_pin https://github.com/nerficg-project/nerficg nerficg e458ac6e41810ad6d9405687c1e698fadd28dac5
clone_pin https://github.com/fastgs/FastGS FastGS 44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f
clone_pin https://github.com/LinjieLyu/SADGS SADGS 029814fb5ed773e4a794cec5115951a64f610ab8

echo "CLONE_PINNED_DONE $(date -Is)"
