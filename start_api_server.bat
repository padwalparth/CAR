@echo off
echo Starting RoadVision AI Real ML API Server on http://localhost:8000...
python "%~dp0model cobination pipeline -autocar simulation\server.py"
pause
