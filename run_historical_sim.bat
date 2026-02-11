@echo off
setlocal
set BASE=%~dp0
"%BASE%.python\python.exe" "%BASE%scripts\historical_sim.py" --sl-atr 2.0 --tp-atr 4.0 --stagger-tp --tp1-atr 2.0 --tp2-atr 4.0 --tp1-pct 0.5 --dca --martingale 1.5 --dca-atr 1.0 --max-dca 2
