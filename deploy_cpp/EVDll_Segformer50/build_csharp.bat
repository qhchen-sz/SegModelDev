@echo off
setlocal

REM ASCII-only (see deploy_cpp\build.bat).
REM Builds EVDll_Segformer.dll (-> dll\) and SegTest.exe (-> SegTest\bin\).
REM The native segformer_Dll_func.dll must already exist under deploy_cpp\build\Release\
REM (run deploy_cpp\build.bat first); SegTest copies it next to itself.

set "HERE=%~dp0"
set "MSBUILD=F:\HymsonSoftWare\VS2022\MSBuild\Current\Bin\MSBuild.exe"

if not exist "%MSBUILD%" (
    echo [ERROR] MSBuild not found at "%MSBUILD%"
    exit /b 1
)

echo [1/2] building EVDll_Segformer.dll ...
"%MSBUILD%" "%HERE%EVDll_Segformer.csproj" /p:Configuration=Release /p:Platform=x64 /v:minimal /nologo
if errorlevel 1 exit /b 1

echo.
echo [2/2] building SegTest.exe ...
"%MSBUILD%" "%HERE%SegTest\SegTest.csproj" /p:Configuration=Release /p:Platform=x64 /v:minimal /nologo
if errorlevel 1 exit /b 1

echo.
echo [OK] %HERE%dll\EVDll_Segformer.dll
echo [OK] %HERE%SegTest\bin\SegTest.exe
exit /b 0
