@echo off
setlocal
set BASE=%~dp0
"%BASE%.python\python.exe" "%BASE%scripts\ml_features.py"
"%BASE%.python\python.exe" "%BASE%scripts\train_lgbm.py"
