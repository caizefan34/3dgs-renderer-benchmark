@echo off
chcp 437 > nul
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 > nul
set FAST_COMPILE=1
set PYTHONUTF8=1
python -c "from gsplat.cuda._backend import _C; print('C1 RESTORED OK')"
