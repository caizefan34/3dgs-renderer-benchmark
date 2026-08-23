# Automatic-ready renderer source clone log

- Timestamp: `2026-07-17T09:14:56+08:00`
- Git: `git version 2.53.0.windows.2`
- Source root: `C:\Users\36570\Documents\Codex\3dgs-renderer-benchmark\artifacts\renderer-sources`
- Operation: source clone and verification only; no installation or compilation performed.
- Clone strategy: `--filter=blob:none --no-checkout`, followed by an explicit detached checkout of the pinned commit.

## Commands executed

The following command pattern was run independently for each repository:

```powershell
git clone --filter=blob:none --no-checkout <source-uri> <destination>
git -C <destination> checkout --detach <source-commit>
git -C <destination> remote -v
git -C <destination> rev-parse HEAD
git -C <destination> symbolic-ref --short -q HEAD
git -C <destination> status --porcelain=v1
git -C <destination> status --short --branch
git -C <destination> config --get remote.origin.partialclonefilter
```

All four destination directories were absent before execution, so no existing user files were overwritten. If a destination had existed, the procedure was configured to verify it without cloning over it.

## Verification results

### original_3dgs / diff-gaussian-rasterization

- Destination: `artifacts/renderer-sources/original-diff-gaussian-rasterization`
- Remote: `https://github.com/graphdeco-inria/diff-gaussian-rasterization`
- Expected commit: `9c5c2028f6fbee2be239bc4c9421ff894fe4fbe0`
- HEAD: `9c5c2028f6fbee2be239bc4c9421ff894fe4fbe0`
- Checkout: detached (`## HEAD (no branch)`)
- Worktree: clean
- Partial clone filter: `blob:none`
- Clone exit: `0`; checkout exit: `0`
- Result: PASS

### gsplat (shared by gsplat and gsplat_higs)

- Destination: `artifacts/renderer-sources/gsplat`
- Remote: `https://github.com/nerfstudio-project/gsplat`
- Expected commit: `77ab983ffe43420b2131669cb35776b883ca4c3c`
- HEAD: `77ab983ffe43420b2131669cb35776b883ca4c3c`
- Checkout: detached (`## HEAD (no branch)`)
- Worktree: clean
- Partial clone filter: `blob:none`
- Clone exit: `0`; checkout exit: `0`
- Result: PASS

### speedy_splat / speedy-splat

- Destination: `artifacts/renderer-sources/speedy-splat`
- Remote: `https://github.com/j-alex-hanson/speedy-splat`
- Expected commit: `34c45c6d9b8bd6110231864f2f358b6d3abbf73d`
- HEAD: `34c45c6d9b8bd6110231864f2f358b6d3abbf73d`
- Checkout: detached (`## HEAD (no branch)`)
- Worktree: clean
- Partial clone filter: `blob:none`
- Clone exit: `0`; checkout exit: `0`
- Result: PASS

### tcgs / 3DGSTensorCore

- Destination: `artifacts/renderer-sources/3DGSTensorCore`
- Remote: `https://github.com/DeepLink-org/3DGSTensorCore`
- Expected commit: `0bb82f88fde211c34b42e1497f0fc7265461592b`
- HEAD: `0bb82f88fde211c34b42e1497f0fc7265461592b`
- Checkout: detached (`## HEAD (no branch)`)
- Worktree: clean
- Partial clone filter: `blob:none`
- Clone exit: `0`; checkout exit: `0`
- Result: PASS

## Summary

All four unique automatic-ready upstream repositories were cloned successfully at their required immutable commits. Remote URLs match the benchmark manifest, every checkout is detached, and all worktrees are clean. No failures occurred.
