@echo off
echo Starting WPARK Simulation Server...
echo.
echo Access the application at: http://127.0.0.1:5000
echo.
cd /d "%~dp0"
python flask_app\app.py
pause
