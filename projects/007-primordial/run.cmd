@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Set up the app first using the three commands in README.md.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m primordial
if errorlevel 1 pause
