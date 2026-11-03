@echo off
setlocal

REM ============================================================
REM  run_seg_test.bat -- run seg_test.exe (RTX50 / TensorRT 10) with PATH set.
REM
REM  seg_test.exe dynamically loads segformer_Dll_func.dll, which depends on
REM  nvinfer_10.dll / nvinfer_plugin_10.dll / cudnn*_9.dll / cublas*.dll /
REM  cudart64_12.dll / opencv_world480.dll.
REM
REM  Usage:
REM    run_seg_test.bat <engine> <image or dir> <out dir> [--mask]
REM    run_seg_test.bat <engine> --eval <val_full.txt>
REM
REM  The exe lives in x64\Release\ by default. For a Debug build run:
REM    set SEG_CFG=Debug
REM    run_seg_test.bat ...
REM
REM  Paths to adjust when moving to another machine:
REM    TRT      = TensorRT 10.10 SDK root
REM    CUDNN    = cuDNN 9 runtime DLL directory (bin\12.9)
REM    CUDA     = CUDA 12.9 runtime DLL directory (bin)
REM    OCV      = OpenCV 4.8 DLL directory (has opencv_world480[d].dll)
REM
REM  ASCII-only on purpose (cmd.exe parses .bat with OEM codepage 936/GBK).
REM ============================================================

set "HERE=%~dp0"

set "TRT=C:\Users\Univo\TensorRT-10.10.0.31"
set "CUDNN=C:\Program Files\NVIDIA\CUDNN\v9.11\bin\12.9"
set "CUDA=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin"
set "OCV=C:\Users\Univo\opencv\opencv\build\x64\vc16\bin"

if not defined SEG_CFG set "SEG_CFG=Release"
set "BIN=%HERE%x64\%SEG_CFG%"
set "EXE=%BIN%\seg_test.exe"
if not exist "%EXE%" (
    echo [ERROR] %EXE% not found -- build segformer_dll.sln ^(%SEG_CFG%^|x64^) first.
    exit /b 1
)

set "PATH=%BIN%;%TRT%\lib;%TRT%\bin;%CUDNN%;%CUDA%;%OCV%;%PATH%"

"%EXE%" %*
exit /b %errorlevel%
