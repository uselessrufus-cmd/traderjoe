@echo off
setlocal
set BASE=%~dp0
"%BASE%.python\python.exe" "%BASE%scripts\self_assess.py"
"%BASE%.python\python.exe" "%BASE%scripts\performance_summary.py"
