@echo off
setlocal
set BASE=%~dp0
"%BASE%.python\python.exe" "%BASE%scripts\rl_train_historical.py" --episodes 40 --fee-rate 0.001
