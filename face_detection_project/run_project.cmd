@echo off
setlocal

set "PROJECT_ROOT=%~dp0"
set "PYTHON=%PROJECT_ROOT%..\venv\Scripts\python.exe"
set "TF_CPP_MIN_LOG_LEVEL=2"

if not exist "%PYTHON%" (
    echo Error: Virtual environment Python not found: %PYTHON%
    exit /b 1
)

"%PYTHON%" "%PROJECT_ROOT%recognise.py"
exit /b %errorlevel%
