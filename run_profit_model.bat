@echo off
setlocal
set BASE=%~dp0
"%BASE%.python\python.exe" "%BASE%scripts\train_profit_model.py"
"%BASE%.python\python.exe" "%BASE%scripts\predict_profit_model.py"
