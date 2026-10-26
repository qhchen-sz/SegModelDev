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
REM  Paths to adjust when moving to another machine:
REM    TRT      = TensorRT 10.10 SDK root
REM    CUDART   = CUDA 12.9 + cuDNN 9 runtime DLL directory
REM               (on this machine: AIRTX50\Debug50 bundle, which has all of them)
REM    OCV      = OpenCV build directory (vc16)
REM
REM  ASCII-only on purpose (cmd.exe parses .bat with OEM codepage 936/GBK).
REM ============================================================

set "HERE=%~dp0"

set "TRT=D:\Rtx50Test\TensorRT-10.10.0.31"
set "CUDART=D:\CaoJiaoJie\AIRTX50\Debug50"
set "OCV=E:\opencv_exe\opencv\build"

set "EXE=%HERE%seg_test.exe"
if not exist "%EXE%" (
    echo [ERROR] %EXE% not found -- build segformer_dll.sln first.
    exit /b 1
)

set "PATH=%HERE%;%TRT%\lib;%TRT%\bin;%CUDART%;%OCV%\x64\vc16\bin;%PATH%"

"%EXE%" %*
exit /b %errorlevel%
