import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from code.benchmarks import btc_buy_and_hold, sp500_buy_and_hold
from code.environment import BTCTradingEnv
from code.features import compute_features, feature_columns, folds, get_fold_splits, normalize_features

data_path  = "data/Binance_BTCUSDT_1h.csv"
models_dir = "artifacts/models"
output_dir = "artifacts"


def load_and_prepare():
    df = pd.read_csv(data_path, skiprows=1)
    df.columns = [
        "unix", "date", "symbol", "open", "high",
        "low", "close", "volume_btc", "volume_usdt", "tradecount"
    ]
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    df = df.sort_values("date").reset_index(drop=True)
    df = df.drop(columns=["symbol"])
    df = compute_features(df)
    df = df.dropna().reset_index(drop=True)
    return df


def run_model_on_test(model, df):
    env = BTCTradingEnv(df, episode_length=len(df) - 50)
    env.reset(seed=0)
    env.current_step = 0
    env.episode_end  = len(df) - 2
    obs = env._obs()

    trade_log    = []
    equity       = 1.0
    equity_dates = {}

    done = False
    while not done:
        action, _  = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        if "entry_price" in info:
            equity *= np.exp(info["net_pnl"])
            exit_ts = pd.Timestamp(info["exit_timestamp"])
            equity_dates[exit_ts] = equity

            duration = (
                pd.Timestamp(info["exit_timestamp"]) -
                pd.Timestamp(info["entry_timestamp"])
            ).total_seconds() / 3600

            trade_log.append({
                "entry_timestamp": info["entry_timestamp"],
                "exit_timestamp":  info["exit_timestamp"],
                "exit_reason":     info["exit_reason"],
                "entry_price":     info["entry_price"],
                "exit_price":      info["exit_price"],
                "direction":       "long" if info["direction"] == 1 else "short",
                "gross_pnl":       info["gross_pnl"],
                "net_pnl":         info["net_pnl"],
                "duration_hours":  duration,
            })

        done = terminated or truncated

    date_index   = pd.to_datetime(df["date"].values)
    equity_curve = pd.Series(np.nan, index=date_index)
    equity_curve.iloc[0] = 1.0

    for date, val in equity_dates.items():
        if date in equity_curve.index:
            equity_curve[date] = val

    equity_curve = equity_curve.ffill().fillna(1.0)

    return trade_log, equity_curve


def compute_metrics(equity_curve, trade_log, n_hours):
    log_returns       = np.log(equity_curve / equity_curve.shift(1)).dropna()
    total_return      = float(equity_curve.iloc[-1] - 1.0)
    annualized_return = float((1.0 + total_return) ** (8760.0 / n_hours) - 1.0)
    annualized_vol    = float(log_returns.std() * np.sqrt(8760))
    sharpe            = annualized_return / annualized_vol if annualized_vol > 0 else 0.0

    neg_returns  = log_returns[log_returns < 0]
    downside_vol = float(neg_returns.std() * np.sqrt(8760)) if len(neg_returns) > 0 else 1e-9
    sortino      = annualized_return / downside_vol if downside_vol > 0 else 0.0

    cummax       = equity_curve.cummax()
    drawdown     = (equity_curve - cummax) / cummax
    max_drawdown = float(drawdown.min())
    calmar       = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

    if trade_log:
        trades_df    = pd.DataFrame(trade_log)
        hit_rate     = float((trades_df["net_pnl"] > 0).mean())
        avg_duration = float(trades_df["duration_hours"].mean())
        turnover     = len(trade_log) / n_hours
    else:
        hit_rate     = 0.0
        avg_duration = 0.0
        turnover     = 0.0

    return {
        "total_return":       total_return,
        "annualized_return":  annualized_return,
        "annualized_vol":     annualized_vol,
        "sharpe":             sharpe,
        "sortino":            sortino,
        "max_drawdown":       max_drawdown,
        "calmar":             calmar,
        "hit_rate":           hit_rate,
        "avg_duration_hours": avg_duration,
        "turnover":           turnover,
    }


def compute_feature_analysis(train_df, fold_num):
    from sklearn.feature_selection import mutual_info_regression

    target   = train_df["log_return"].shift(-1).dropna()
    features = train_df[feature_columns].iloc[:len(target)]

    correlations = features.corrwith(target)
    mi_scores    = mutual_info_regression(features, target, random_state=42)

    result = pd.DataFrame({
        "feature":     feature_columns,
        "correlation": correlations.values,
        "mutual_info": mi_scores,
    })
    result = result.sort_values("mutual_info", ascending=False).reset_index(drop=True)
    result.to_csv(f"{output_dir}/feature_analysis_fold_{fold_num}.csv", index=False)


def plot_equity_curve(strategy_curve, btc_curve, sp500_curve, fold_num, test_start, test_end):
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(strategy_curve.index, strategy_curve.values, linewidth=1,   label="RL Strategy")
    ax.plot(btc_curve.index,      btc_curve.values,      linewidth=0.8, label="BTC Buy & Hold", alpha=0.8)

    if sp500_curve is not None:
        ax.plot(sp500_curve.index, sp500_curve.values, linewidth=0.8, label="S&P500 (SPY)", alpha=0.8)

    ax.set_title(f"Equity Curve - Test Fold {fold_num}  ({test_start[:10]} to {test_end[:10]})")
    ax.set_ylabel("Portfolio Value (normalized)")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{output_dir}/equity_curves_test_fold_{fold_num}.png", dpi=150)
    plt.close(fig)


def run_evaluation():
    os.makedirs(output_dir, exist_ok=True)
    df = load_and_prepare()

    all_metrics = []

    for fold_config in folds:
        fold_num = fold_config["fold"]
        print(f"evaluating fold {fold_num}...")

        train_df, val_df, test_df = get_fold_splits(df, fold_config)
        train_df, val_df, test_df, _, _ = normalize_features(train_df, val_df, test_df)

        compute_feature_analysis(train_df, fold_num)

        model = PPO.load(f"{models_dir}/ppo_fold_{fold_num}")

        trade_log, strategy_curve = run_model_on_test(model, test_df)

        n_hours = len(test_df)
        metrics = compute_metrics(strategy_curve, trade_log, n_hours)
        metrics["fold"] = fold_num

        _, _, raw_test_df = get_fold_splits(load_and_prepare(), fold_config)
        btc_curve, _    = btc_buy_and_hold(raw_test_df)
        sp500_curve, _  = sp500_buy_and_hold(
            fold_config["test_start"], fold_config["test_end"]
        )

        btc_metrics   = compute_metrics(btc_curve, [], n_hours)
        btc_metrics["fold"] = fold_num

        sp500_metrics = {}
        if sp500_curve is not None:
            sp500_metrics = compute_metrics(
                sp500_curve.reindex(btc_curve.index).ffill().fillna(1.0),
                [], n_hours
            )
            sp500_metrics["fold"] = fold_num

        metrics["alpha_vs_btc"]   = metrics["annualized_return"] - btc_metrics["annualized_return"]
        metrics["alpha_vs_sp500"] = (
            metrics["annualized_return"] - sp500_metrics.get("annualized_return", 0.0)
        )

        if trade_log:
            trade_df = pd.DataFrame(trade_log)
            trade_df.to_csv(f"{output_dir}/trade_log_fold_{fold_num}.csv", index=False)

        plot_equity_curve(
            strategy_curve, btc_curve, sp500_curve,
            fold_num,
            fold_config["test_start"],
            fold_config["test_end"]
        )

        all_metrics.append({"label": f"strategy_fold_{fold_num}", **metrics})
        all_metrics.append({"label": f"btc_fold_{fold_num}",      **btc_metrics})
        if sp500_metrics:
            all_metrics.append({"label": f"sp500_fold_{fold_num}", **sp500_metrics})

        print(f"  trades={len(trade_log)}  sharpe={metrics['sharpe']:.3f}  return={metrics['total_return']:.3f}")

    metrics_df    = pd.DataFrame(all_metrics)
    strategy_rows = metrics_df[metrics_df["label"].str.startswith("strategy")]
    numeric_cols  = strategy_rows.select_dtypes(include=np.number).columns
    summary_mean  = strategy_rows[numeric_cols].mean()
    summary_std   = strategy_rows[numeric_cols].std()
    summary_mean["label"] = "strategy_mean"
    summary_std["label"]  = "strategy_std"

    metrics_df = pd.concat(
        [metrics_df,
         pd.DataFrame([summary_mean]),
         pd.DataFrame([summary_std])],
        ignore_index=True
    )
    metrics_df.to_csv(f"{output_dir}/metrics_summary.csv", index=False)
    print("evaluation complete")


if __name__ == "__main__":
    run_evaluation()