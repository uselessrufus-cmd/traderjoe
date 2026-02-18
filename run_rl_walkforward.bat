@echo off
setlocal
set BASE=%~dp0
"%BASE%.python\python.exe" "%BASE%scripts\rl_train_historical.py" --walkforward --episodes 10 --optimize-trials 6 --train-bars 8760 --test-bars 2160 --step-bars 720 --max-minutes 15 --reward-clip 0.10 --fee-rate 0.001
