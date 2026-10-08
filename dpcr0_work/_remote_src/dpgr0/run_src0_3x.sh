cd /root/epic05_dpgr0
for i in 1 2 3; do
  CUDA_VISIBLE_DEVICES=0 /root/miniforge3/envs/gsplat/bin/python dpgr0_src0probe.py 2>&1 | grep -E 'A retain|B fresh'
  echo "--- run done ---"
done
/root/miniforge3/envs/gsplat/bin/python -c "import json; d=json.load(open('runs/dpgr0_phase/room@10000.json')); r=d['rows'][0]; print('JSON source0 n_unique=',r['n_unique_same_gpu'],'det=',r['same_gpu_deterministic'])"
