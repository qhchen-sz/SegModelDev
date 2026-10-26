@echo off
setlocal

REM ASCII-only (see build.bat). Puts every runtime DLL directory on PATH, then
REM forwards all arguments to seg_infer.exe.
REM
REM Model selection
REM ---------------
REM The exe resolves "--model <name>" to models\segformer_<name>_b1_fp16_640.engine.
REM SEG_MODEL sets the default (v2_strat). The exe keeps the LAST --model/--engine
REM seen on the command line, so passing --model here simply overrides the default
REM -- no fragile string matching in batch:
REM
REM     run.bat --eval ..\segformer_v2\dataset_strat\val_full.txt          (v2_strat)
REM     run.bat --model sw_s4 --eval ..\segformer_v2\dataset_strat\val_full.txt
REM
REM Build a model's engine first with:  tools\build_engine.bat <name>

set "HERE=%~dp0"

set "TRT=E:\Tensorrt\TensorRT-8.6.1.6"
set "CUDA=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
set "OCV=E:\opencv_exe\opencv\build"

set "PATH=%TRT%\lib;%CUDA%\bin;%OCV%\x64\vc16\bin;%PATH%"

set "EXE=%HERE%build\Release\seg_infer.exe"

if "%SEG_MODEL%"=="" set "SEG_MODEL=v2_strat"

if not exist "%EXE%" (
    echo [ERROR] %EXE% not found -- run build.bat first
    exit /b 1
)

"%EXE%" --model "%SEG_MODEL%" %*
exit /b %errorlevel%
