# TraderJoe Indicators + Chart

This repo pulls historical BTC data (Bitstamp) and renders a GUI chart.

## Setup

```
The repo includes a portable Python under `.python` so you can run without
installing Python system-wide.

If you need to reinstall dependencies:
.\.python\python.exe -m pip install -r requirements.txt
```

## OpenAI (optional)

If you want AI analysis on the chart, set your API key as an environment variable:

```
setx OPENAI_API_KEY "your_key_here"
```

Close and reopen PowerShell after setting it.

If you prefer a local key file (no environment variables), run:
```
set_api_key.bat
```
This stores the key in `config/openai_key.txt` (plain text).

## Historical download (Bitstamp BTCUSD)

This downloads Bitstamp BTCUSD 1‑minute history (2012‑2025) plus the latest
updates CSV, then resamples into the intervals you want.

```
python scripts/download_bitstamp_data.py --intervals 1m,5m,15m,1h,4h,12h,1d,1w,1mo
```

Outputs:
- `data/bitstamp/ohlc/bitstamp_1m.csv`
- `data/bitstamp/ohlc/bitstamp_5m.csv`
- `data/bitstamp/ohlc/bitstamp_15m.csv`
- `data/bitstamp/ohlc/bitstamp_1h.csv`
- `data/bitstamp/ohlc/bitstamp_4h.csv`
- `data/bitstamp/ohlc/bitstamp_12h.csv`
- `data/bitstamp/ohlc/bitstamp_1d.csv`
- `data/bitstamp/ohlc/bitstamp_1w.csv`
- `data/bitstamp/ohlc/bitstamp_1mo.csv`

## Input CSV format (for indicators)

Required columns (case-insensitive):
- `timestamp` (seconds or milliseconds)
- `open`, `high`, `low`, `close`, `volume`

If your CSV uses other column names, use the `--col-*` overrides.

## Run

```
python indicators.py --input data.csv --output data.indicators.csv \
  --rsi 14 --mfi 14 --macd --bb 20 --ma 20,50,200 --ema 12,26 \
  --order-blocks --ob-atr 14 --ob-mult 1.8 --ob-lookahead 6
```

## Chart GUI

```
TraderJoeChart.bat
```

Use the sidebar to select interval and CSVs. By default, it loads:
`data/bitstamp/ohlc/bitstamp_1h.csv` and computes indicators/signals automatically.

The chart uses a simple 1h-only composite by default to show:
- Market direction (trend/range)
- A suggested position size based on composite score

To show buy/sell markers and order blocks, point the chart to:
`data/signals/btc_1h_signals.csv`

## Phase 1 (1h rule-based baseline)

Generate signals and a quick backtest report:

```
run_phase1_backtest.bat
```

Outputs:
- `data/signals/btc_1h_signals.csv`
- `data/signals/btc_1h_report.txt`

## Phase 2 (Walk-forward testing)

```
run_phase2_walkforward.bat
```

Outputs:
- `data/signals/btc_1h_walkforward.txt`

## Live signal logging (hourly)

```
run_live_logger.bat
```

Outputs:
- `data/signals/live_signals.csv`

## ML Training (LightGBM)

This builds a labeled dataset from 1h candles and trains a LightGBM model.

```
run_ml_train.bat
```

Outputs:
- `data/ai_memory.db`
- `data/models/lgbm_model.txt`

To get the latest ML signal:

```
run_ml_predict.bat
```

Output:
- `data/models/lgbm_latest_signal.txt`

## Self assessment + performance summary

```
run_self_assess.bat
```

Outputs:
- `data/ai_memory.db` (signal_history table)
- `data/models/performance_summary.txt`

## Paper trading (single position, no long+short overlap)

Run an hourly loop that:
- Updates ML signal
- Executes a single-position paper trade

```
run_auto_loop.bat
```

Live loop now runs `paper_trade.py --rl` (online Q-learning). It writes status to:
- `data/models/rl_status.txt`

Paper trading summary:
```
run_paper_summary.bat
```

Outputs:
- `data/paper_trades.db`
- `data/models/paper_summary.txt`

## Adaptive Parameters (self-tuning)

```
run_adaptive_params.bat
```

Outputs:
- `data/models/adaptive_params.txt`

## Historical Paper-Trading Simulation (fast)

Trains on 1 year, paper-trades the next 3 months on historical data.
Uses ATR-based stop-loss/take-profit with optional DCA, optional martingale,
and optional staggered take-profit.

```
run_historical_sim.bat
```

Output:
- `data/models/historical_sim_report.txt`

Optional flags (run from `.\.python\python.exe scripts\historical_sim.py`):
- `--dca` enables DCA
- `--martingale 1.5` scales DCA adds and also scales new trade size after losses (set `1.0` to disable martingale)
- `--stagger-tp` enables TP1/TP2 partial take-profit
- `--tp1-atr`, `--tp2-atr`, `--tp1-pct` to tune the staggered TP

## Auto-learning (15-minute test run)

Runs the full self-learning cycle for 15 minutes:
simulation → strategy explore → historical sim → train → predict → adaptive params.

```
run_auto_learning_15min.bat
```

## Historical Simulation (self-improvement loop)

This runs a rolling train/test simulation on historical data (1h) so the
model “self‑improves” across windows.

```
run_simulation.bat
```

Output:
- `data/models/sim_report.txt`
- `data/models/sim_summary.txt`

## Simulation Explorer (many runs)

Runs many simulations across horizons/thresholds and ranks stability.

```
run_sim_explore.bat
```

Outputs:
- `data/models/sim_explore.csv`
- `data/models/sim_explore_summary.txt`

## Strategy Explorer (rule-based variants)

Runs multiple rule-based strategies with different:
MFI thresholds, order-block lookback, trend filter, stop-loss, take-profit.

```
run_strategy_explore.bat
```

Outputs:
- `data/models/strategy_explore.csv`
- `data/models/strategy_explore_summary.txt`

By default the batch script runs a subset of combos for speed. To run all:
```
.\.python\python.exe scripts\strategy_explore.py
```

## Profit-optimized model (regression)

```
run_profit_model.bat
```

Outputs:
- `data/models/profit_model.txt`
- `data/models/profit_signal.txt`

## Regime report

```
run_regime_report.bat
```

Output:
- `data/models/regime_report.txt`

## Output

Adds columns for:
- `rsi_14`
- `mfi_14`
- `bb_mid_20`, `bb_up_20`, `bb_dn_20`, `bb_bw_20`
- `sma_20`, `sma_50`, `sma_200`
- `ema_12`, `ema_26`
- `ob_bull`, `ob_bear` (simple order block heuristic)
- `macd_line`, `macd_signal`, `macd_hist`

## Notes on order blocks

Order blocks are not a standardized indicator. This script uses a pragmatic heuristic:
- A candle is a bull order block if its real body >= `ob_mult * ATR(ob_atr)`
  and within the next `ob_lookahead` candles, price breaks above that candle's high.
- Bear order block is symmetric for downside breaks.

Tune `--ob-mult` and `--ob-lookahead` to fit your market and timeframe.
