@echo off
setlocal

REM ============================================================
REM  run_seg_test.bat -- run seg_test.exe with the right PATH.
REM
REM  seg_test.exe dynamically loads segformer_Dll_func.dll, which in turn
REM  depends on nvinfer.dll / nvinfer_plugin.dll / opencv_world480[d].dll.
REM  This script puts those on PATH first.
REM
REM  Usage:
REM    run_seg_test.bat <engine> <image or dir> <out dir> [--mask]
REM    run_seg_test.bat <engine> --eval <val_full.txt>
REM
REM  Config: picks build\Release first, then build\Debug.
REM  Build first by compiling the sln (segformer_dll + seg_test).
REM
REM  Note: Debug DLL links opencv_world480d.dll, Release links opencv_world480.dll.
REM        Both live in the OpenCV bin dir added below.
REM
REM  ASCII-only on purpose (cmd.exe parses .bat with OEM codepage 936/GBK).
REM ============================================================

set "HERE=%~dp0"
set "DEPLOY=%HERE%.."

set "TRT=E:\Tensorrt\TensorRT-8.6.1.6"
set "OCV=E:\opencv_exe\opencv\build"
set "CUDA=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"

set "BUILD="
if exist "%DEPLOY%\build\Release\seg_test.exe" set "BUILD=%DEPLOY%\build\Release"
if "%BUILD%"=="" if exist "%DEPLOY%\build\Debug\seg_test.exe" set "BUILD=%DEPLOY%\build\Debug"
if "%BUILD%"=="" (
    echo [ERROR] seg_test.exe not found in build\Release or build\Debug -- compile the sln first.
    exit /b 1
)
if not exist "%BUILD%\segformer_Dll_func.dll" (
    echo [ERROR] segformer_Dll_func.dll not found in %BUILD% -- compile the sln first.
    exit /b 1
)

set "PATH=%BUILD%;%TRT%\bin;%TRT%\lib;%OCV%\x64\vc16\bin;%CUDA%\bin;%PATH%"

echo [run_seg_test] using %BUILD%
"%BUILD%\seg_test.exe" %*
exit /b %errorlevel%
