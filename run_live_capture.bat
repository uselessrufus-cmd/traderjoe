@echo off
setlocal
set BASE=%~dp0
if not exist "%BASE%data\bitunix\live" mkdir "%BASE%data\bitunix\live"
"%BASE%.python\python.exe" -u "%BASE%scripts\bitunix_ws_capture.py" --symbol BTCUSDT --intervals 1m,5m,15m,1h,4h,12h,1d,1w,1M --retain-days 7 ^> "%BASE%data\bitunix\live\bitunix_ws.log" 2^>^&1
