@echo off
setlocal

REM ============================================================
REM  build_engine.bat <model> [--force] [--fp32]
REM
REM  <model> = a run directory under segformer_v2\runs_strat\, i.e. the
REM            weights live at segformer_v2\runs_strat\<model>\best
REM            Examples: sw_s4, v2_strat, sw_s1, sw_s2, sw_s3, v2b_strat
REM
REM  Steps:
REM    1. export ONNX from that best/ via segformer_v2\export_best_onnx.py
REM    2. compile it to a TensorRT engine with trtexec (FP16 by default)
REM
REM  Output: models\segformer_<model>_b1_640.onnx          (one ONNX serves both precisions)
REM          models\segformer_<model>_b1_<prec>_640.engine
REM
REM  Precision: the ONNX graph does not encode precision -- --fp16/--fp32 is
REM  purely a trtexec build flag, hence no precision suffix on the .onnx.
REM
REM  --fp32 is worth knowing about. Measured on sw_s4, the FP16 engine's logit
REM  error vs PyTorch is ~19x larger than FP32's (max|d| 2.31 vs 0.12) because
REM  these particular weights are more fp16-sensitive (v2_strat's are not).
REM  It did NOT move the metric that matters -- negative-image false positives
REM  2016 vs 2003 px, i.e. +0.65%, same >100px/clean counts -- so FP16 stays
REM  the default; it is 1.9x faster (154 vs 296 ms/image). Build fp32 when you
REM  want tighter parity with the PyTorch numbers.
REM
REM  The engine is NOT portable across TensorRT versions or GPUs.
REM  Rebuild it on the target machine.
REM
REM  ASCII-only on purpose: cmd.exe parses .bat with the OEM codepage
REM  (936/GBK), so UTF-8 non-ASCII comments break command parsing.
REM ============================================================

set "HERE=%~dp0"
set "DEPLOY=%HERE%.."
set "ROOT=%DEPLOY%\.."

set "MODEL=%~1"
if "%MODEL%"=="" (
    echo [ERROR] usage: build_engine.bat ^<model^> [--force] [--fp32]
    echo         e.g.  build_engine.bat sw_s4
    echo         ^<model^> is a directory under segformer_v2\runs_strat\
    exit /b 1
)

set "FORCE=0"
set "PREC=fp16"
for %%A in (%*) do (
    if /I "%%A"=="--force" set "FORCE=1"
    if /I "%%A"=="--fp32" set "PREC=fp32"
)
if /I "%PREC%"=="fp32" (set "PRECFLAG=") else (set "PRECFLAG=--fp16")

set "TRT=E:\Tensorrt\TensorRT-8.6.1.6"
set "PY=F:\HymsonSoftWare\MiniForge\envs\yolov10\python.exe"
set "BEST=%ROOT%\segformer_v2\runs_strat\%MODEL%\best"
set "ONNX=%DEPLOY%\models\segformer_%MODEL%_b1_640.onnx"
set "ENGINE=%DEPLOY%\models\segformer_%MODEL%_b1_%PREC%_640.engine"

if not exist "%BEST%\config.json" (
    echo [ERROR] no weights at "%BEST%"
    echo         expected segformer_v2\runs_strat\%MODEL%\best\config.json
    exit /b 1
)
if not exist "%TRT%\bin\trtexec.exe" (
    echo [ERROR] trtexec.exe not found at "%TRT%\bin\trtexec.exe"
    exit /b 1
)
if not exist "%PY%" (
    echo [ERROR] python not found at "%PY%"
    echo         note: the default "python" on this machine is a base env without torch
    exit /b 1
)

echo [build_engine] model     = %MODEL%
echo [build_engine] precision = %PREC%
echo [build_engine] best      = %BEST%
echo [build_engine] onnx      = %ONNX%
echo [build_engine] engine    = %ENGINE%
echo.

if exist "%ENGINE%" (
    if "%FORCE%"=="0" (
        echo [skip] engine already exists -- pass --force to rebuild
        exit /b 0
    )
)

if "%FORCE%"=="0" (
    if exist "%ONNX%" (
        echo [skip] reusing existing ONNX
        goto :build
    )
)

REM ---------- 1. export ONNX ------------------------------------------
REM export_best_onnx.py defaults --out to runs\v2_strat\segformer_v2_strat_best.onnx.
REM Not passing --out would silently OVERWRITE the v2 ONNX, so always pass it.
echo [1/2] exporting ONNX from %BEST% ...
"%PY%" "%ROOT%\segformer_v2\export_best_onnx.py" --src "%BEST%" --out "%ONNX%"
if errorlevel 1 (
    echo [ERROR] ONNX export failed
    exit /b 1
)

:build
REM ---------- 2. build engine -----------------------------------------
REM Fixed 640x640: infer_fast_strat.py:39-41 pads every edge tile with
REM BORDER_REFLECT up to tile x tile, so every sliding-window tile is exactly
REM 640x640 and the spatial dims never need to be dynamic. Fixing them lets TRT
REM constant-fold the 84 Shape + 76 Reshape nodes from the dynamic H/W graph.
REM Batch stays dynamic 1-16 so the last partial batch fits.
set "PATH=%TRT%\lib;%TRT%\bin;%PATH%"
echo.
echo [2/2] building %PREC% engine ...
"%TRT%\bin\trtexec.exe" --onnx="%ONNX%" --saveEngine="%ENGINE%" %PRECFLAG% --minShapes=pixel_values:1x3x640x640 --optShapes=pixel_values:16x3x640x640 --maxShapes=pixel_values:16x3x640x640 --memPoolSize=workspace:4096
if errorlevel 1 (
    echo.
    echo [ERROR] trtexec failed.
    echo         If the ONNX parser rejected LayerNormalization, re-export at a
    echo         lower opset and retry:
    echo             python segformer_v2\export_best_onnx.py --src ^<best^> --out ^<onnx^> --opset 16
    exit /b 1
)

echo.
echo [OK] %ENGINE%
exit /b 0
