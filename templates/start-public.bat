@echo off
echo ========================================
echo    SPOTECTION - PUBLIC ACCESS
echo ========================================
echo.
echo Step 1: Authenticating ngrok...
ngrok authtoken 35RL8oQ75bx0IgQnn3HkhpP5h8Q_57YBCFFWfGFrtaubacH35
echo.
echo Step 2: Starting Flask application...
start cmd /k "python app.py"
timeout /t 5
echo.
echo Step 3: Starting public tunnel...
echo.
echo WAITING FOR PUBLIC URL...
echo.
ngrok http 5000
pause