import sys
P = '/root/epic05_dpc0/src_full/gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference/HigsNativeBackwardDpc.cu'
s = open(P, encoding='utf-8').read()
before = len(s)
idx = s.find('// DPC0 Stage 2 reduce:')
banner = s.rfind('// ====', 0, idx)
fi = s.find('higs_rasterize_backward_dpc(', idx)
if banner < 0 or fi < 0 or fi < banner:
    print('MARK NOT FOUND banner=%d fi=%d' % (banner, fi)); sys.exit(1)
block = s[banner:fi].rstrip('\n')
s = s[:banner] + s[fi:]
rt = s.find('std::tuple<')
if rt < 0:
    print('RT NOT FOUND'); sys.exit(1)
s = s[:rt] + block + '\n\n' + s[rt:]
open(P, 'w', encoding='utf-8').write(s)
print('MOVED block=%d chars; file %d -> %d' % (len(block), before, len(s)))