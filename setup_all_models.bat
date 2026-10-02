@echo off
setlocal
title Full Model Setup - Tutorial Lab
cd /d "%~dp0"

echo.
echo ============================================================
echo   FULL SETUP  (packages + ALL models)
echo ============================================================
echo.
echo   Use this for classroom / offline Tutorial Lab.
echo   Downloads NER, BERT, ALBERT, RoBERTa, and Local LLM.
echo   Larger and slower than setup.bat (Local LLM only).
echo ============================================================
echo.

where python >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Python was not found on PATH.
  echo Install Python 3.10+ and tick "Add python.exe to PATH".
  pause
  exit /b 1
)

python --version
echo.
python scripts\easy_setup.py --all
set ERR=%ERRORLEVEL%
echo.
if not "%ERR%"=="0" (
  echo Full setup did not finish successfully.
  pause
  exit /b %ERR%
)

echo.
echo Done. Start the app with run_app.bat
echo.
pause
endlocal
