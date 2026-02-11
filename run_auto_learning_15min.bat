@echo off
setlocal
set BASE=%~dp0
"%BASE%.python\python.exe" "%BASE%scripts\auto_learning.py" --minutes 15 --cycle 300
