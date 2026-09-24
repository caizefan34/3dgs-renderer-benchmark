@echo off
REM Upload and run validation on mx

set KEY=%USERPROFILE%\.ssh\mx_public.key
set PORT=26372
set HOST=liaoyuanjun@36.140.146.31
set SCRIPT=%CD%\tmp_systems_validation_v2.py
set REMOTE=/tmp/gsplat_systems_revalidation/tmp_systems_validation_v2.py

REM Upload the script
scp -P %PORT% -i "%KEY%" "%SCRIPT%" %HOST%:%REMOTE%

REM Run it with proper env vars
ssh -p %PORT% -i "%KEY%" %HOST% "export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions_systems; export CUDA_CACHE_PATH=/tmp/cuda_cache_systems; export CUDA_VISIBLE_DEVICES=1; cd /tmp/gsplat_systems_revalidation && python3 tmp_systems_validation_v2.py --stage c44"
