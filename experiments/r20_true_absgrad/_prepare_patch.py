"""Create a surgical overlay from the actual MX source export (never mutate it)."""
from pathlib import Path
import difflib
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / 'HIGS_SOURCE_HANDOFF_ACTUAL/source'
KERNEL = 'gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference/HigsNativeBackward'
FILES = [KERNEL + '.cu', KERNEL + '.h',
         'gsplat/experimental/render/kernels/cuda/ext.cpp',
         'gsplat/experimental/render/functional/gaussian_inference.py']


def replace(text, old, new, count=1):
    assert text.count(old) == count, (old[:100], text.count(old), count)
    return text.replace(old, new)


def source_diff(before, after, rel):
    """Avoid difflib's autojunk heuristic hiding repeated CUDA macro context."""
    a, b = before.splitlines(True), after.splitlines(True)
    def interval(start, end):
        count = end-start
        return str(start+1) if count == 1 else f'{start+(1 if count else 0)},{count}'
    yield '--- a/'+rel+'\n'
    yield '+++ b/'+rel+'\n'
    for group in difflib.SequenceMatcher(None, a, b, autojunk=False).get_grouped_opcodes(3):
        first, last = group[0], group[-1]
        yield f'@@ -{interval(first[1],last[2])} +{interval(first[3],last[4])} @@\n'
        for tag, i, j, k, l in group:
            if tag == 'equal':
                yield from (' '+line for line in a[i:j])
            else:
                if tag in ('delete','replace'): yield from ('-'+line for line in a[i:j])
                if tag in ('insert','replace'): yield from ('+'+line for line in b[k:l])


def cuda_patch(s):
    s = replace(s, 'template<uint32_t CDIM>\n__global__ void __launch_bounds__(256, 5) higs_blend_bwd_kernel(',
                'template<uint32_t CDIM, bool ABSGRAD>\n__global__ void __launch_bounds__(256, 5) higs_blend_bwd_kernel(')
    s = replace(s, 'template<uint32_t CDIM, uint32_t PX>\n__global__ void',
                'template<uint32_t CDIM, uint32_t PX, bool ABSGRAD>\n__global__ void')
    s = replace(s, '    vec2 *__restrict__ v_means2d,\n',
                '    vec2 *__restrict__ v_means2d,\n    vec2 *__restrict__ v_means2d_abs, // optional [I*N, 2]\n', 2)
    s = replace(s, '            vec2 v_xy_local         = {0.f, 0.f};',
                '            vec2 v_xy_local         = {0.f, 0.f};\n            vec2 v_xy_abs_local     = {0.f, 0.f};', 2)
    # Derive absolute values from the per-pixel signed output, BEFORE PX/warp sums.
    anchor = '            warpSum<CDIM>(v_rgb_local, warp);'
    pos = s.index(anchor)
    s = s[:pos] + '            if constexpr(ABSGRAD)\n            {\n                v_xy_abs_local = {fabsf(v_xy_local.x), fabsf(v_xy_local.y)};\n            }\n' + s[pos:]
    s = replace(s, '                        v_xy_local.x += v_xy_q.x;',
                '                        if constexpr(ABSGRAD)\n                        {\n                            v_xy_abs_local.x += fabsf(v_xy_q.x);\n                            v_xy_abs_local.y += fabsf(v_xy_q.y);\n                        }\n                        v_xy_local.x += v_xy_q.x;')
    s = replace(s, '            warpSum(v_xy_local, warp);',
                '            warpSum(v_xy_local, warp);\n            if constexpr(ABSGRAD)\n            {\n                warpSum(v_xy_abs_local, warp);\n            }', 2)
    s = replace(s, '                atomicAdd(v_xy_ptr + 1, v_xy_local.y);',
                '                atomicAdd(v_xy_ptr + 1, v_xy_local.y);\n                if constexpr(ABSGRAD)\n                {\n                    float *v_abs_ptr = (float *)v_means2d_abs + 2 * (int64_t)g;\n                    atomicAdd(v_abs_ptr, v_xy_abs_local.x);\n                    atomicAdd(v_abs_ptr + 1, v_xy_abs_local.y);\n                }', 2)
    s = replace(s, '    at::Tensor  // v_means2d [I*N, 2]\n',
                '    at::Tensor, // v_means2d [I*N, 2]\n    at::Tensor  // v_means2d_abs [I*N, 2] or undefined when disabled\n')
    s = replace(s, 'higs_rasterize_backward(\n', 'higs_rasterize_backward_with_absgrad(\n')
    s = replace(s, '    const at::Tensor &grad_colors   // [M, K, D] (SH) or [M, D] (RGB)\n',
                '    const at::Tensor &grad_colors,  // [M, K, D] (SH) or [M, D] (RGB)\n    bool absgrad\n')
    s = replace(s, '    at::Tensor v_means2d     = at::zeros({I * N, 2}, opts);',
                '    at::Tensor v_means2d     = at::zeros({I * N, 2}, opts);\n    TORCH_CHECK(!absgrad || tile_size == 16, "True AbsGrad requires tile_size == 16");\n    at::Tensor v_means2d_abs = absgrad ? at::zeros({I * N, 2}, opts) : at::Tensor();')
    s = replace(s, '#define LAUNCH_BLEND_BWD(CDIM)', '#define LAUNCH_BLEND_BWD(CDIM, ABSGRAD)')
    s = s.replace('higs_blend_bwd_kernel<(CDIM)>', 'higs_blend_bwd_kernel<(CDIM), (ABSGRAD)>')
    s = re.sub(r'higs_blend_bwd_px_kernel<\(CDIM\), ([124])>', r'higs_blend_bwd_px_kernel<(CDIM), \1, (ABSGRAD)>', s)
    line = '                reinterpret_cast<vec2 *>(v_means2d.data_ptr<float>()),'
    lines = s.splitlines(keepends=True)
    found = 0
    for i in range(len(lines)-1, -1, -1):
        if lines[i].startswith(line):
            lines.insert(i+1, '                absgrad ? reinterpret_cast<vec2 *>(v_means2d_abs.data_ptr<float>()) : nullptr, \\\n')
            found += 1
    assert found == 4
    s = ''.join(lines)
    for dim in (1, 3, 4):
        s = replace(s, f'            LAUNCH_BLEND_BWD({dim});',
                    f'            if(absgrad) {{ LAUNCH_BLEND_BWD({dim}, true); }}\n            else {{ LAUNCH_BLEND_BWD({dim}, false); }}')
    s = replace(s, '        v_means2d\n', '        v_means2d,\n        v_means2d_abs\n')
    # Preserve the old C++ and Python seven-tensor ABI through a thin wrapper.
    original = (BASE / FILES[0]).read_text(encoding='utf-8')
    start = original.index('std::tuple<\n    at::Tensor, // v_means [N, 3]')
    end = original.index('\n{\n    check_float_cuda', start)
    signature = original[start:end]
    header = (BASE / FILES[1]).read_text(encoding='utf-8')
    params = header[header.index('    higs_rasterize_backward('):]
    names = re.findall(r'(?:const at::(?:optional<at::Tensor>|Tensor) &|int64_t |double )(\w+)', params)
    assert len(names) == 31, names
    body = '\n{\n    auto out = higs_rasterize_backward_with_absgrad(\n        ' + ', '.join(names) + ', false\n    );\n    return {' + ', '.join(f'std::get<{i}>(out)' for i in range(7)) + '};\n}\n\n'
    point = s.rindex('} // namespace gaussian_render_inference_scene')
    s = s[:point] + signature + body + s[point:]
    return s


def header_patch(s):
    start = s.index('    std::tuple<')
    end = s.index('\n} // namespace', start)
    decl = s[start:end]
    extra = decl.replace('higs_rasterize_backward(', 'higs_rasterize_backward_with_absgrad(')
    extra = replace(extra, '        at::Tensor  // v_means2d [I*N, 2]',
                    '        at::Tensor, // v_means2d [I*N, 2]\n        at::Tensor  // v_means2d_abs [I*N, 2] or undefined')
    extra = replace(extra, '        const at::Tensor &grad_colors     // [M, K, D] (SH) or [M, D] (RGB) float32',
                    '        const at::Tensor &grad_colors,    // [M, K, D] (SH) or [M, D] (RGB) float32\n        bool absgrad')
    return s[:end] + '\n\n    // R20 opt-in API: first seven outputs retain the legacy contract.\n' + extra + s[end:]


def binding_patch(s):
    start = s.index('    m.def(\n        "higs_rasterize_backward",')
    end = s.index('\n\n    m.def(', start)
    binding = s[start:end].replace('higs_rasterize_backward', 'higs_rasterize_backward_with_absgrad')
    binding = replace(binding, '        py::arg("grad_colors")',
                      '        py::arg("grad_colors"),\n        py::arg("absgrad") = false')
    return s[:end] + '\n\n    m.attr("higs_true_absgrad_version") = 1;\n' + binding + s[end:]


def python_patch(s):
    s = replace(s, '        cull_cache_key="default",\n    ):',
                '        cull_cache_key="default",\n        absgrad=False,\n    ):')
    s = replace(s, '        ctx.native = native',
                '        ctx.absgrad = bool(absgrad)\n        if ctx.absgrad:\n            from ..kernels import _backend as _abs_backend\n            if (not native or\n                    getattr(_abs_backend._C, "higs_true_absgrad_version", 0) != 1):\n                raise RuntimeError("absgrad=True requires the R20 native HiGS extension; "\n                                   "rebuild it before requesting true absgrad.")\n        ctx.native = native')
    s = replace(s, '        if ctx.N_visible == 0:\n            # All Gaussians invisible',
                '        if ctx.N_visible == 0:\n            if ctx.absgrad:\n                means2d_proxy.absgrad = torch.zeros_like(means2d_proxy)\n            # All Gaussians invisible')
    s = replace(s, '        out = backend.higs_rasterize_backward(',
                '        backward_fn = (backend.higs_rasterize_backward_with_absgrad\n                       if ctx.absgrad else backend.higs_rasterize_backward)\n        abs_kwargs = {"absgrad": True} if ctx.absgrad else {}\n        out = backward_fn(')
    s = replace(s, '            grad_colors=grad_colors,\n        )',
                '            grad_colors=grad_colors,\n            **abs_kwargs,\n        )')
    s = replace(s, '        ) = out\n\n        grad_means2d',
                '        ) = out[:7]\n\n        if ctx.absgrad:\n            abs_full = torch.zeros_like(means2d_proxy)\n            abs_full.index_copy_(1, visible_ids, out[7].reshape(I, ctx.N_visible, 2))\n            means2d_proxy.absgrad = abs_full\n\n        grad_means2d')
    s = replace(s, '            None,  # cull_cache_key (non-tensor input)',
                '            None,  # cull_cache_key (non-tensor input)\n            None,  # absgrad (non-tensor input)')
    # Only the four Stage B/C public/internal entry points receive the new keyword.
    start = s.index('def _higs_frozen_forward(')
    front, tail = s[:start], s[start:]
    tail = replace(tail, '    scene=None,\n) -> dict:', '    scene=None,\n    absgrad: bool = False,\n) -> dict:', 4)
    tail = replace(tail, '        scene=scene,\n    )', '        scene=scene,\n        absgrad=absgrad,\n    )', 2)
    tail = replace(tail, '        sampling_mode, tile_mask, cull_refresh_interval, "default",',
                   '        sampling_mode, tile_mask, cull_refresh_interval, "default", absgrad,')
    tail = replace(tail, '        cull_cache_key,\n    )', '        cull_cache_key, absgrad,\n    )')
    tail = replace(tail, '            "native_available": fwd_meta["native_available"],',
                   '            "native_available": fwd_meta["native_available"],\n            "absgrad": absgrad,', 2)
    tail = tail.replace('        backward_mode: "higs_native"',
                        '        absgrad: Compute native per-pixel componentwise absolute gradients in\n            backward and expose means2d.absgrad (requires the R20 extension).\n        backward_mode: "higs_native"')
    return front + tail


def main():
    changes, manifest = [], {}
    for rel, patch in zip(FILES, (cuda_patch, header_patch, binding_patch, python_patch)):
        src = BASE / rel
        before = src.read_text(encoding='utf-8')
        after = patch(before)
        target = HERE / 'source' / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(after, encoding='utf-8', newline='\n')
        manifest[rel] = {'original_sha256': hashlib.sha256(src.read_bytes()).hexdigest(),
                         'patched_sha256': hashlib.sha256(target.read_bytes()).hexdigest()}
        changes.extend(source_diff(before, after, rel))
    preamble = 'R20_SOURCE_PATCH.diff\nStatus: SOURCE_PATCH_COMPLETE; FROZEN_MX_PARITY_NOT_EXECUTED\n\n'
    (HERE / 'R20_SOURCE_PATCH.diff').write_text(preamble + ''.join(changes), encoding='utf-8', newline='\n')
    (HERE / 'R20_SOURCE_MANIFEST.json').write_text(json.dumps({'status': 'SOURCE_PATCH_COMPLETE_FROZEN_MX_NOT_EXECUTED',
        'source_export': 'HIGS_SOURCE_HANDOFF_ACTUAL/SOURCE_MANIFEST.json', 'files': manifest}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
