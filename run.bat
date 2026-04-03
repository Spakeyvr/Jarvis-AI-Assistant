@echo off

if not exist ".venv\Scripts\activate.bat" (
    echo Creating virtual environment...
    py -3.12 -m venv .venv
    call .venv\Scripts\activate.bat
    echo Installing dependencies...
    pip install -r requirements.txt
) else (
    call .venv\Scripts\activate.bat
)

python jarvis\main.py
pause
