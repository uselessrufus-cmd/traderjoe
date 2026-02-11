@echo off
setlocal
set BASE=%~dp0

REM 1) Ensure history exists
if not exist "%BASE%data\bitstamp\ohlc\bitstamp_1m.csv" (
  echo Downloading Bitstamp history...
  call "%BASE%run_download_history.bat"
)

REM 2) Launch chart (Bitstamp history only)
start "" /min "%BASE%.python\python.exe" -m streamlit run "%BASE%chart_app.py" --server.headless true --server.port 8501
timeout /t 2 >nul
start "" "http://localhost:8501"
