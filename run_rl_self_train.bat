@echo off
setlocal
set BASE=%~dp0
"%BASE%.python\python.exe" "%BASE%scripts\rl_self_train.py" --rounds 4 --max-minutes 15 --per-round-minutes 4 --episodes 8 --optimize-trials 6 --reward-clip 0.05 --fee-rate 0.001
