@echo off
setlocal

REM ASCII-only on purpose: cmd.exe parses .bat with the OEM codepage (936/GBK),
REM so UTF-8 non-ASCII comments would get mangled and break the commands.

set "HERE=%~dp0"
set "VCVARS=F:\HymsonSoftWare\VS2022\VC\Auxiliary\Build\vcvars64.bat"
set "CMAKE=F:\HymsonSoftWare\Cmake\bin\cmake.exe"

if not exist "%VCVARS%" (
    echo [ERROR] vcvars64.bat not found at "%VCVARS%"
    exit /b 1
)
if not exist "%CMAKE%" (
    echo [ERROR] cmake.exe not found at "%CMAKE%"
    exit /b 1
)

call "%VCVARS%" >nul
if errorlevel 1 (
    echo [ERROR] vcvars64 failed
    exit /b 1
)

echo [build] configuring...
"%CMAKE%" -S "%HERE%." -B "%HERE%build" -G "Visual Studio 17 2022" -A x64
if errorlevel 1 exit /b 1

echo [build] compiling...
"%CMAKE%" --build "%HERE%build" --config Release --parallel
if errorlevel 1 exit /b 1

echo.
echo [OK] %HERE%build\Release\seg_infer.exe
exit /b 0
