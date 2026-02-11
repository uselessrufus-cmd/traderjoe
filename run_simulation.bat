@echo off
setlocal
set BASE=%~dp0
"%BASE%.python\python.exe" "%BASE%scripts\sim_train_loop.py"
"%BASE%.python\python.exe" "%BASE%scripts\sim_summary.py"
