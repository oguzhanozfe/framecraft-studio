@echo off
setlocal
cd /d "%~dp0"
if not defined FRAMECRAFT_HOME set "FRAMECRAFT_HOME=%~dp0.local"
if not defined FRAMECRAFT_PROJECTS set "FRAMECRAFT_PROJECTS=%~dp0projects"
if not defined UGC_CREATOR_PORT set "UGC_CREATOR_PORT=7867"
where ffmpeg >nul 2>&1 || (echo Install FFmpeg and add it to PATH. & pause & exit /b 1)
if not exist ".venv\Scripts\python.exe" (py -3 -m venv .venv || exit /b 1)
if not exist ".venv\Scripts\ugc-creator.exe" (".venv\Scripts\python.exe" -m pip install -e ".[campaign,speech]" || exit /b 1)
".venv\Scripts\ugc-creator.exe" studio --port %UGC_CREATOR_PORT% --open

