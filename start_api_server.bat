@echo off
echo Starting RoadVision AI Real ML API Server on http://localhost:8000...
set PYTHONPATH=%~dp0model cobination pipeline -autocar simulation\.venv\Lib\site-packages;%PYTHONPATH%
"C:\Program Files\Python311\python.exe" "%~dp0model cobination pipeline -autocar simulation\server.py"
pause
