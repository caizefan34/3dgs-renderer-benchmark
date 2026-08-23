@echo off
rem ===========================================================================
rem  HiGS dev environment for the patched gsplat source tree (Windows).
rem
rem  Sets PYTHONPATH so `import gsplat` resolves to the locally-built patched
rem  source tree, and loads the MSVC environment required by torch's JIT
rem  fallback for `gsplat_scene_cuda` (without `cl` on PATH, 15 HiGS tests fail
rem  with "Failed to load gsplat_scene_cuda via JIT build/load").
rem
rem  Usage (from anywhere):
rem      call scripts\higs\env.cmd
rem
rem  Overridable environment variables (set before calling this script):
rem      GSPLAT_SRC   - patched gsplat source tree (default: artifacts/renderer-sources/gsplat)
rem      HIGS_PYFIX   - dir holding the Windows sitecustomize shim (default: .build_tmp/pyfix)
rem      HIGS_VCVARS  - full path to vcvars64.bat (default: detected via vswhere)
rem      PYTHON       - python executable (default: python)
rem ===========================================================================

rem --- repo root = two levels above this script ---------------------------------
set "HIGS_ROOT=%~dp0..\.."

if not defined GSPLAT_SRC set "GSPLAT_SRC=%HIGS_ROOT%\artifacts\renderer-sources\gsplat"
if not defined HIGS_PYFIX set "HIGS_PYFIX=%HIGS_ROOT%\.build_tmp\pyfix"
if not defined PYTHON set "PYTHON=python"

if not exist "%GSPLAT_SRC%\gsplat\__init__.py" (
    echo [higs] ERROR: patched gsplat source not found at "%GSPLAT_SRC%"
    echo [higs] Set GSPLAT_SRC to your checkout of gsplat with patches/higs-differentiable.patch applied.
    exit /b 1
)

rem --- MSVC environment via vswhere (fall back to the common VS2022 path) -------
rem NOTE: %HIGS_VSROOT% must be consumed on its own line (not inside the same
rem parenthesized block), because cmd expands %var% at block parse time.
set "HIGS_VCVARS_FOUND="
if not defined HIGS_VCVARS (
    if defined VCToolsInstallDir set "HIGS_VCVARS=%VCToolsInstallDir%..\..\..\Auxiliary\Build\vcvars64.bat"
)
set "HIGS_VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not defined HIGS_VCVARS (
    if exist "%HIGS_VSWHERE%" (
        for /f "usebackq tokens=*" %%i in (`"%HIGS_VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "HIGS_VSROOT=%%i"
    )
)
if defined HIGS_VSROOT set "HIGS_VCVARS=%HIGS_VSROOT%\VC\Auxiliary\Build\vcvars64.bat"
if not defined HIGS_VCVARS set "HIGS_VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if exist "%HIGS_VCVARS%" (
    call "%HIGS_VCVARS%" >nul 2>&1
    set "HIGS_VCVARS_FOUND=1"
)
if not defined HIGS_VCVARS_FOUND (
    echo [higs] WARNING: MSVC vcvars64.bat not found; if gsplat_scene_cuda is not
    echo [higs]         already built, JIT builds will fail. Set HIGS_VCVARS to
    echo [higs]         your VS install's vcvars64.bat.
)

set "PYTHONPATH=%HIGS_PYFIX%;%GSPLAT_SRC%"
set "GSPLAT_SKIP_FROM_WORLD=1"
echo [higs] PYTHONPATH=%PYTHONPATH%
