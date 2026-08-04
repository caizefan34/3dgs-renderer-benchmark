@echo off
rem HiGS training-path benchmark against the patched gsplat source (Windows).
rem Usage: scripts\higs\run_benchmark.cmd [benchmark args...]
rem   e.g. scripts\higs\run_benchmark.cmd --scene tanks_and_temples/train --backends std higs_native higs_native_ts --tile-sampling-ratio 0.5
call "%~dp0env.cmd" || exit /b 1
cd /d "%HIGS_ROOT%"
%PYTHON% benchmark\run_higs_train_benchmark.py %*
exit /b %ERRORLEVEL%
