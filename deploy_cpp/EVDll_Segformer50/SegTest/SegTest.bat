@echo off
setlocal

REM ASCII-only (see deploy_cpp\build.bat). Puts the TRT / CUDA / OpenCV runtime
REM DLL directories on PATH, then forwards all arguments to SegTest.exe.

set "HERE=%~dp0"

set "TRT=E:\Tensorrt\TensorRT-8.6.1.6"
set "CUDA=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
set "OCV=E:\opencv_exe\opencv\build"

set "PATH=%TRT%\lib;%CUDA%\bin;%OCV%\x64\vc16\bin;%PATH%"

set "EXE=%HERE%bin\SegTest.exe"
if not exist "%EXE%" (
    echo [ERROR] %EXE% not found -- run build_csharp.bat first
    exit /b 1
)

"%EXE%" %*
exit /b %errorlevel%
