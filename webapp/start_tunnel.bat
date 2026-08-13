@echo off
"C:\Users\karthik\AppData\Local\cloudflared\cloudflared.exe" tunnel --url http://127.0.0.1:8000 --no-autoupdate --loglevel info 2>"%~dp0tunnel_url.txt"
