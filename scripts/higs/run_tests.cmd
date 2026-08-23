@echo off
rem Full repo pytest suite against the patched gsplat source (Windows).
rem Usage: scripts\higs\run_tests.cmd [pytest args...]
rem   e.g. scripts\higs\run_tests.cmd
rem        scripts\higs\run_tests.cmd tests/test_higs_native_backward.py -q
call "%~dp0env.cmd" || exit /b 1
cd /d "%HIGS_ROOT%"
if "%~1"=="" (
    %PYTHON% -m pytest tests -q
) else (
    %PYTHON% -m pytest %*
)
exit /b %ERRORLEVEL%
