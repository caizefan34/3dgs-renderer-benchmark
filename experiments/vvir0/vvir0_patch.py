# VRIR0 build bootstrap: fixes MSVC codepage decode bug in torch cpp_extension.
# Build-tooling only; does NOT touch renderer kernels or training semantics.
import os
def apply():
    try:
        import torch.utils.cpp_extension as ce
        ce.SUBPROCESS_DECODE_ARGS = ('cp936', 'replace')
        return True
    except Exception as e:
        return False
if __name__ == '__main__':
    print('patch applied:', apply())
