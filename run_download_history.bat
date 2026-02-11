@echo off
setlocal
set BASE=%~dp0
"%BASE%.python\python.exe" "%BASE%scripts\download_bitstamp_data.py" --intervals 1m,5m,15m,1h,4h,12h,1d,1w,1mo
