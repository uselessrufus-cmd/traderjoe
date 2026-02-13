@echo off
setlocal
set BASE=%~dp0
"%BASE%.python\python.exe" "%BASE%scripts\reset_training_state.py" --delete-db
