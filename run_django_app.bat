@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found. Create it first:
    echo python -m venv .venv
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo Installing required packages...
    ".venv\Scripts\python.exe" -m pip install -r "Django Application\requirements-web.txt"
    if errorlevel 1 (
        echo Failed to install requirements.
        pause
        exit /b 1
    )
)

cd /d "%~dp0Django Application"
"%~dp0.venv\Scripts\python.exe" manage.py runserver
