@echo off
setlocal
set BASE=%~dp0
"%BASE%.python\python.exe" "%BASE%scripts\rl_self_train.py" --rounds 4 --max-minutes 15 --per-round-minutes 4 --episodes 8 --optimize-trials 6 --reward-clip 0.05 --fee-rate 0.001 --fee-schedule 0.0005,0.0010,0.0015,0.0020 --promote-min-fee 0.0020 --cpu-target-pct 50 --status-out data/models/fee_ladder/rl_self_train_status.txt --pool-out data/models/fee_ladder/rl_strategy_pool.csv --recheck-out data/models/fee_ladder/rl_strategy_recheck.csv --champions-out data/models/fee_ladder/rl_champions.csv --champions-summary-out data/models/fee_ladder/rl_champions_summary.txt --exp-tag fee_ladder
