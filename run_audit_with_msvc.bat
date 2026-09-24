@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
echo MSVC environment activated.
set PYTHONIOENCODING=utf-8
set CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING=1
echo CCCL warning suppressed.
python training_workset_audit.py --data-dir "data\datasets\mipnerf360\room" --max-steps 500 --tile-size 16 --output-json "results\phase-a100\training_workset_stability_audit.json" --output-report "reports\phase-a100\training_workset_stability_audit.md"
echo Exit code: %ERRORLEVEL%
