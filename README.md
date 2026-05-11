# Reinforcement Learning Trading Environment

PPO-based agent trained to trade BTC/USDT on hourly data from August 2017 to April 2026.
Uses a semi-Markov gymnasium environment with stop-loss, take-profit, and maximum holding
period rules. Walk-forward validation across four expanding folds.

## Project Structure

```
code/
    main.py              # single entrypoint - runs the full pipeline
    eda.py               # exploratory data analysis
    features.py          # feature engineering and fold definitions
    environment.py       # semi-Markov gymnasium trading environment
    train.py             # PPO training with hyperparameter search
    evaluate.py          # metrics, equity curves, trade logs
    benchmarks.py        # BTC buy-and-hold and S&P500 benchmarks
    walkthrough.pdf      # code structure and design decisions

artifacts/
    btcusdt_eda.pdf
    training_curves_fold_{1,2,3,4}.png
    equity_curves_test_fold_{1,2,3,4}.png
    validation_results_fold_{1,2,3,4}.csv
    feature_analysis_fold_{1,2,3,4}.csv
    trade_log_fold_{1,2,3,4}.csv
    metrics_summary.csv

report.pdf
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Place `Binance_BTCUSDT_1h.csv` in the `data/` folder, then:

```bash
python -m code.main
```

## Results Summary

| Fold | Test Period   | Sharpe | Total Return |
|------|---------------|--------|--------------|
| 1    | 2023          | -0.39  | -14.6%       |
| 2    | 2024          | -1.13  | -54.0%       |
| 3    | 2025          | -1.28  | -51.9%       |
| 4    | Jan-Apr 2026  | -1.66  | -35.8%       |
| Mean |               | -1.12  | -39.1%       |
