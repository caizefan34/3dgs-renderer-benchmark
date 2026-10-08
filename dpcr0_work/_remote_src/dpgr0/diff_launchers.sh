A=/root/epic05_dpgr0/src_dpgr0/gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference/HigsNativeBackwardDpgr0.cu
B=/root/epic05_dpc0/src_full/gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference/HigsNativeBackwardDpc.cu
diff <(sed 's/dpgr0/dpc/g;s/DPGR0/DPC/g' $A) $B > /tmp/dpgr0_launcher.diff 2>&1
echo "diff lines: $(grep -c '^[<>]' /tmp/dpgr0_launcher.diff)"
cat /tmp/dpgr0_launcher.diff
