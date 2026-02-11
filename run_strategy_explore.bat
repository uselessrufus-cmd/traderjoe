@echo off
setlocal
set BASE=%~dp0
"%BASE%.python\python.exe" "%BASE%scripts\strategy_explore.py" --max-combos 300 --sleep 0.05
