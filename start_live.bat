@echo off
rem Starts the Flask web app and opens a free Cloudflare quick tunnel so the
rem app is reachable from anywhere at a public https://...trycloudflare.com URL.
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
  echo venv not found - run: python -m venv .venv ^&^& .venv\Scripts\python -m pip install -r requirements.txt
  pause
  exit /b 1
)
start "darknet-classifier" /min cmd /c ".venv\Scripts\python.exe webapp\app.py"
timeout /t 5 /nobreak >nul
start "darknet-tunnel" /min cmd /c "webapp\start_tunnel.bat"
echo.
echo Local app : http://127.0.0.1:8000
echo Public URL: see webapp\tunnel_url.txt  (takes ~15s to appear)
echo.
