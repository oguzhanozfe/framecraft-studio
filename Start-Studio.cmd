@echo off
setlocal
cd /d "%~dp0"
where ffmpeg >nul 2>&1 || (echo Install FFmpeg and add it to PATH, then run again. & pause & exit /b 1)
if not exist ".venv\Scripts\python.exe" (py -3 -m venv .venv || exit /b 1)
if not exist ".venv\Scripts\framecraft.exe" (".venv\Scripts\python.exe" -m pip install -e . || exit /b 1)
".venv\Scripts\framecraft.exe" studio --open

