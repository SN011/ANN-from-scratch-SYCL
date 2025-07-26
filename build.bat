@echo off
setlocal

REM ==== adjust to your install ====
set "ONEAPI=C:\Program Files (x86)\Intel\oneAPI"
set "INTEL_VER=2024.2"

REM 1) Clean & prepare
cd /d "%~dp0"
if exist build rmdir /s /q build
mkdir build
cd build

REM 2) Build in one shot:
"%ONEAPI%\compiler\%INTEL_VER%\bin\icx-cl.exe" ^
  -v ^
  /fsycl ^
  /std:c++20 ^
  /EHsc ^
  /Zi ^
  /Od ^
  /MDd ^
  /Qopenmp ^
  /fno-sycl-rdc ^
  /fsycl-device-code-split=per_kernel ^
  /fno-sycl-device-lib=all ^
  /fsycl-device-lib=libm-fp32 ^
  /fsycl-max-parallel-link-jobs=4 ^
  /I"%ONEAPI%\compiler\%INTEL_VER%\include" ^
  /I"%ONEAPI%\mkl\%INTEL_VER%\include" ^
  ..\ImageNew.cpp ..\MNIST_experiment.cpp ^
  /link ^
    /LIBPATH:"%ONEAPI%\mkl\%INTEL_VER%\lib" ^
    mkl_sycld.lib mkl_intel_ilp64.lib mkl_sequential.lib mkl_core.lib ^
    OpenCL.lib ^
    vcompd.lib ^
    msvcrtd.lib ^
    /OUT:ann_app.exe

if errorlevel 1 (
  echo.
  echo ********** BUILD FAILED **********
) else (
  echo.
  echo ********** BUILD SUCCEEDED: %CD%\ann_app.exe **********
)

pause
endlocal
