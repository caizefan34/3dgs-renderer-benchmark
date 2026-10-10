#!/usr/bin/env python3
"""Copy R15 package to MX and run capture + oracle"""
import subprocess
import os

# R15 package location
r15_local = r'C:\Users\36570\3dgs-renderer-benchmark\HIGS_R15\HIGS_R15_CAPTURE_ORACLE_MX'
r15_mx = '/mnt/storage_pool/liaoyuanjun/HIGS_R15_CAPTURE_ORACLE_MX'

print("Step 1: Creating tar archive...")
# Create tar archive
subprocess.run(['tar', '-czf', 'r15_package.tar.gz', '-C', 
                r'C:\Users\36570\3dgs-renderer-benchmark\HIGS_R15', 
                'HIGS_R15_CAPTURE_ORACLE_MX'], check=True)

# Copy to MX
print("Step 2: Transferring to MX...")
subprocess.run(['scp', 'r15_package.tar.gz', f'mx:{r15_mx}.tar.gz'], check=True)

# Extract on MX
print("Step 3: Extracting on MX...")
ssh_cmd = f'''
mkdir -p {r15_mx}
tar -xzf {r15_mx}.tar.gz -C /mnt/storage_pool/liaoyuanjun/
cd {r15_mx}
ls -la
'''
subprocess.run(['ssh', 'mx', ssh_cmd], check=True)

print("Step 4: Running capture script on MX with conda environment...")
capture_cmd = f'''
cd {r15_mx}
source /home/liaoyuanjun/miniforge3/bin/activate /mnt/storage_pool/liaoyuanjun/higs-13scene-env
export PYTHONPATH=/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx
python3 capture_higs.py \
  --source-root /home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx \
  --entrypoint reference/R8_WORKING_ENTRYPOINT.py \
  --out {r15_mx}/captured_projected_inputs.pt \
  --samples 1000 --seed 42
'''
subprocess.run(['ssh', 'mx', capture_cmd], check=True)

print("Capture complete!")
print(f"Results saved to: {r15_mx}/captured_projected_inputs.pt")
