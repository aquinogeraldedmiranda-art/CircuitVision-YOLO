@echo off
setlocal
cd /d "%~dp0"
set PORT=8000
where py >nul 2>&1
if %errorlevel%==0 (
 set "PY=py -3.11"
) else (
 set "PY=python"
)
%PY% -m venv .venv
if errorlevel 1 (
 echo Python 3.11 required. Download from https://python.org
 pause & exit /b 1
)
call .venv\Scripts\activate.bat
python -m pip install -r requirements-local-pt.txt
if errorlevel 1 (pause & exit /b 1)
echo.
echo The frontend will open at http://127.0.0.1:8000
start "" "http://127.0.0.1:8000"
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
pause
