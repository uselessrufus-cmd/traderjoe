@echo off
setlocal
set BASE=%~dp0
"%BASE%.python\python.exe" "%BASE%scripts\ml_features_mtf.py"
"%BASE%.python\python.exe" "%BASE%scripts\train_lgbm_mtf.py"
