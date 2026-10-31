@echo off
setlocal

REM ============================================================
REM  build_engine50.bat <model> [--fp32] [--force] [--onnx <path>]
REM
REM  Builds a TensorRT 10.10 engine for RTX 50 series from an ONNX file.
REM
REM  *** RUN THIS ON THE RTX 50 MACHINE ***
REM  TensorRT engines are tied to the exact TensorRT version + GPU. An engine
REM  built on another card (e.g. RTX 3090) will fail to load on the RTX 50.
REM
REM  <model>   = model name used in the output filename, e.g. sw_s4, v2_strat
REM  --onnx    = explicit ONNX path (otherwise auto-detected, see below)
REM  --fp32    = build FP32 instead of the default FP16
REM  --force   = rebuild even if the engine already exists
REM
REM  Default ONNX lookup:
REM    1. ..\models\segformer_<model>_b1_640.onnx
REM    2. ..\..\segformer_v2\runs_strat\<model>\segformer_<model>_best.onnx
REM  Output:
REM    ..\models50\segformer_<model>_b1_<fp16|fp32>_640.engine
REM
REM  Paths to adjust when moving to another machine:
REM    TRT     = TensorRT 10.10 SDK root
REM    CUDART  = CUDA 12.9 + cuDNN 9 runtime DLL directory
REM    PY      = python (only needed if the ONNX has to be exported)
REM
REM  ASCII-only on purpose (cmd.exe parses .bat with OEM codepage 936/GBK).
REM ============================================================

set "HERE=%~dp0"
set "DEPLOY=%HERE%.."
set "ROOT=%DEPLOY%\.."
set "TRT=D:\Rtx50Test\TensorRT-10.10.0.31"
set "CUDART=D:\CaoJiaoJie\AIRTX50\Debug50"
set "PY=F:\HymsonSoftWare\MiniForge\envs\yolov10\python.exe"
set "OUTDIR=%DEPLOY%\models50"

set "MODEL="
set "PREC=fp16"
set "FORCE=0"
set "ONNX="

:parse
if "%~1"=="" goto parsed
if /I "%~1"=="--fp32" (
    set "PREC=fp32"
    shift
    goto parse
)
if /I "%~1"=="--force" (
    set "FORCE=1"
    shift
    goto parse
)
if /I "%~1"=="--onnx" (
    set "ONNX=%~2"
    shift
    shift
    goto parse
)
if "%MODEL%"=="" (
    set "MODEL=%~1"
    shift
    goto parse
)
echo [ERROR] unknown argument: %~1
exit /b 1
:parsed

if "%MODEL%"=="" (
    echo [ERROR] usage: build_engine50.bat ^<model^> [--fp32] [--force] [--onnx ^<path^>]
    echo         e.g.  build_engine50.bat sw_s4
    echo               build_engine50.bat v2_strat
    exit /b 1
)

if /I "%PREC%"=="fp32" (set "PRECFLAG=") else (set "PRECFLAG=--fp16")

if "%ONNX%"=="" (
    if exist "%DEPLOY%\models\segformer_%MODEL%_b1_640.onnx" (
        set "ONNX=%DEPLOY%\models\segformer_%MODEL%_b1_640.onnx"
    )
)
if "%ONNX%"=="" (
    if exist "%ROOT%\segformer_v2\runs_strat\%MODEL%\segformer_%MODEL%_best.onnx" (
        set "ONNX=%ROOT%\segformer_v2\runs_strat\%MODEL%\segformer_%MODEL%_best.onnx"
    )
)

set "ENGINE=%OUTDIR%\segformer_%MODEL%_b1_%PREC%_640.engine"

if not exist "%TRT%\bin\trtexec.exe" (
    echo [ERROR] trtexec.exe not found at "%TRT%\bin\trtexec.exe"
    echo         Set TRT in this script to the TensorRT 10.10 SDK root.
    exit /b 1
)
if "%ONNX%"=="" (
    echo [ERROR] no ONNX found for model "%MODEL%"
    echo         Tried:
    echo           %DEPLOY%\models\segformer_%MODEL%_b1_640.onnx
    echo           %ROOT%\segformer_v2\runs_strat\%MODEL%\segformer_%MODEL%_best.onnx
    echo         Export it first, or pass --onnx ^<path^>.
    exit /b 1
)
if not exist "%ONNX%" (
    echo [ERROR] ONNX not found: "%ONNX%"
    exit /b 1
)

if exist "%ENGINE%" (
    if "%FORCE%"=="0" (
        echo [skip] engine already exists -- pass --force to rebuild
        echo        %ENGINE%
        exit /b 0
    )
)

if not exist "%OUTDIR%" mkdir "%OUTDIR%"

set "PATH=%TRT%\lib;%TRT%\bin;%CUDART%;%PATH%"

echo [build_engine50] model     = %MODEL%
echo [build_engine50] precision = %PREC%
echo [build_engine50] onnx      = %ONNX%
echo [build_engine50] engine    = %ENGINE%
echo [build_engine50] NOTE      = engine is built for THIS GPU only.
echo.

REM RTX50 + TensorRT 10: this model is numerically WRONG for batch>1
REM (logits shift, the whole image becomes background); only batch=1 is
REM correct. The DLL therefore runs batch=1 and uses multi-stream
REM concurrency (SEGFORMER_WORKERS) for throughput.
REM Fixed batch=1: spatial 640x640 fixed, batch static 1.
"%TRT%\bin\trtexec.exe" --onnx="%ONNX%" --saveEngine="%ENGINE%" %PRECFLAG% ^
    --minShapes=pixel_values:1x3x640x640 ^
    --optShapes=pixel_values:1x3x640x640 ^
    --maxShapes=pixel_values:1x3x640x640 ^
    --memPoolSize=workspace:1024MiB
if errorlevel 1 (
    echo.
    echo [ERROR] trtexec failed.
    echo         If the parser rejected LayerNormalization, re-export the ONNX
    echo         at a lower opset and retry, e.g.:
    echo             "%PY%" segformer_v2\export_best_onnx.py --src ^<best^> --out ^<onnx^> --opset 16
    exit /b 1
)

echo.
echo [OK] %ENGINE%
echo     Copy this engine to the RTX 50 machine only if it was built there.
exit /b 0
