"""Load the isolated diagnostic CUDA shadow pass without altering FastGS."""

from pathlib import Path

from torch.utils.cpp_extension import load


_extension = None


def extension():
    global _extension
    if _extension is None:
        _extension = load(
            name="fastgs_shadow_gate_diagnostic_v3",
            sources=[str(Path(__file__).with_name("fastgs_shadow_gate_cuda.cu"))],
            extra_cuda_cflags=["-O3", "-lineinfo"],
            verbose=True,
        )
    return _extension
