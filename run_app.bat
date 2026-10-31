@echo off
title BrickVision AI — Industrial Brick Material Inspection
echo =======================================================================
echo   BRICKVISION AI — INDUSTRIAL BRICK MATERIAL INSPECTION
echo =======================================================================
echo.
echo Starting FastAPI Web Server...
echo Loading Brick SVM Model package (best_model_package.pkl)...
echo.
echo Once the server is running, open your web browser and go to:
echo http://127.0.0.1:8000  (or http://127.0.0.1:8001 if port 8000 is in use)
echo.

python app.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] The server failed to start or crashed.
    echo Please make sure you have the required packages installed:
    echo pip install fastapi uvicorn numpy opencv-python-headless scikit-image scipy joblib
    echo.
    pause
)
