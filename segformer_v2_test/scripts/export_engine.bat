@echo off
setlocal

REM Export ONNX + TensorRT engine for this dataset.
REM Default: fp16, output <DS>\export\segformer_<dataset>_b1_fp16_640.engine,
REM and a copy in deploy_cpp\models for run.bat --model <dataset>.
REM Pass --fp32 / --force / --deploy-dir ... through if needed.

set "PY=F:\HymsonSoftWare\MiniForge\envs\yolov10\python.exe"
set "DS=%~dp0.."
set "DEPLOY=D:\qhchen\NewModel\newsiliec9\deploy_cpp\models"

"%PY%" "%~dp0export_engine.py" --config "%DS%\config_sw_s4.yaml" --run sw_s4 --deploy-dir "%DEPLOY%" %*
exit /b %ERRORLEVEL%
