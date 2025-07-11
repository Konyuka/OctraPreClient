@echo off
where python >nul 2>nul || exit /b 1
if not exist "venv" python -m venv venv
call venv\Scripts\activate.bat
pip install -r requirements.txt
python cli.py
