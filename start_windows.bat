@echo off
setlocal
cd /d "%~dp0"
where ffmpeg >nul 2>nul
if errorlevel 1 (
 echo FFmpeg is not on PATH. Run: winget install --id Gyan.FFmpeg -e
 pause
 exit /b 1
)
where ffprobe >nul 2>nul
if errorlevel 1 (
 echo ffprobe is not on PATH. Restart PowerShell after installing FFmpeg.
 pause
 exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
 py -3.12 -m venv .venv
 if errorlevel 1 (
   py -3.11 -m venv .venv
   if errorlevel 1 python -m venv .venv
   if errorlevel 1 (
     echo Could not create Python virtual environment. Install Python 3.11 or 3.12.
     pause
     exit /b 1
   )
 )
)
".venv\Scripts\python.exe" -c "import sys; sys.exit(0 if (3, 11) <= sys.version_info[:2] < (3, 14) else 1)"
if errorlevel 1 (
 echo This tool requires Python 3.11 - 3.13. Remove .venv and use Python 3.12.
 pause
 exit /b 1
)
if not exist ".venv\.installed-v4" (
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
 echo Python package installation failed. Check internet access and Python version.
 pause
 exit /b 1
)
echo installed > ".venv\.installed-v4"
)
".venv\Scripts\python.exe" app.py
pause
