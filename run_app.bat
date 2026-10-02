@echo off
setlocal
title Document Chatbot + Tutorial Lab
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Python not found. Run setup.bat first.
  pause
  exit /b 1
)

echo Starting app...
echo Browser should open automatically.
echo Keep this window open while using the app.
echo.
python -m streamlit run app.py
if errorlevel 1 (
  echo.
  echo App failed to start. Try running setup.bat once, then try again.
  pause
)
endlocal
