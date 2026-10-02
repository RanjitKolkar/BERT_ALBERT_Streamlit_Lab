@echo off
setlocal
title Easy Setup - Document Chatbot + Local LLM
cd /d "%~dp0"

echo.
echo ============================================================
echo   EASY SETUP  (one download for Local LLM)
echo ============================================================
echo.
echo   This will:
echo     1. Install required Python packages
echo     2. Download the Local LLM (Qwen2.5-0.5B) once
echo.
echo   Internet required. First run may take several minutes.
echo   You can leave this window open and wait.
echo.
echo   For ALL tutorial models later, use:  setup_all_models.bat
echo ============================================================
echo.

where python >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Python was not found on PATH.
  echo.
  echo Install Python 3.10+ from https://www.python.org/downloads/
  echo During install, tick: "Add python.exe to PATH"
  echo Then double-click this file again.
  echo.
  pause
  exit /b 1
)

python --version
echo.
python scripts\easy_setup.py
set ERR=%ERRORLEVEL%
echo.
if not "%ERR%"=="0" (
  echo Setup did not finish successfully.
  echo Read the messages above, then try again.
  pause
  exit /b %ERR%
)

echo.
echo You can now start the app with run_app.bat
echo.
pause
endlocal
