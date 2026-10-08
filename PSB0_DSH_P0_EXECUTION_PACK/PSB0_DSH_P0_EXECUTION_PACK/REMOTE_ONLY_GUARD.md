# REMOTE-ONLY HARD GATE

Local Windows PC = CONTROL-PLANE ONLY.

Allowed locally:
- source/text inspection
- git status/diff/log
- packaging and file transfer
- inspection of small JSON/Markdown outputs

Forbidden locally:
- 3DGS training
- CUDA build/JIT/kernel execution
- GPU benchmark/profiling
- ncu/nsys
- scene-scale replay/oracle
- large checkpoint execution
- heavy NumPy/PyTorch experiments

All experimental work MUST run on EPIC-05 or another remote node explicitly approved by the user.

Before every experimental command record:
- hostname
- GPU model
- cwd
- source/git SHA

If remote identity is not proven:
`STOP_REMOTE_GUARD`.
