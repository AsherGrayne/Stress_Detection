@echo off
echo Starting ML Inference Service on port 8081...
start "Inference Service" cmd /k "cd /d %~dp0 && python -m uvicorn inference_service.main:app --host 127.0.0.1 --port 8081"

echo Starting Node.js Dashboard/API on port 8090...
start "Stress Dashboard" cmd /k "cd /d %~dp0\node-app && npm start"

echo All services launched. Open http://localhost:8090
pause
