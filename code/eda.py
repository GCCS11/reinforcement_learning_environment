# EDA

import os
import warnings
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore")

data_path = "data/Binance_BTCUSDT_1h.csv"
output_path = "artifacts/btcusdt_eda.pdf"

regimes = [
    ("2017-08-17", "2018-01-06", "Bull Run I",    "#d4edda"),
    ("2018-01-07", "2019-01-01", "Crypto Winter", "#f8d7da"),
    ("2019-01-02", "2020-10-01", "Accumulation",  "#fff3cd"),
    ("2020-10-02", "2021-11-09", "Bull Run II",   "#d4edda"),
    ("2021-11-10", "2022-12-31", "Bear Market",   "#f8d7da"),
    ("2023-01-01", "2024-03-01", "Recovery",      "#fff3cd"),
    ("2024-03-02", "2026-04-26", "Bull Run III",  "#d4edda"),
]


def load_data(path):
    df = pd.read_csv(path, skiprows=1)
    df.columns = [
        "unix", "date", "symbol", "open", "high",
        "low", "close", "volume_btc", "volume_usdt", "tradecount"
    ]
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    df = df.sort_values("date").reset_index(drop=True)
    df = df.drop(columns=["symbol"])
    return df


def compute_returns(df):
    df = df.copy()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    return df.dropna(subset=["log_return"]).reset_index(drop=True)


def run_stationarity_tests(series):
    series = series.dropna()
    adf = adfuller(series, autolag="AIC")
    kpss_res = kpss(series, regression="c", nlags="auto")
    return {
        "ADF Statistic":  adf[0],
        "ADF p-value":    adf[1],
        "KPSS Statistic": kpss_res[0],
        "KPSS p-value":   kpss_res[1],
    }


def shade_regimes(ax):
    for start, end, label, color in regimes:
        ax.axvspan(
            pd.Timestamp(start), pd.Timestamp(end),
            alpha=0.25, color=color, label=label
        )


def plot_page_one(pdf, df):
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle("BTC/USDT Exploratory Data Analysis", fontsize=15)

    ax = axes[0]
    ax.plot(df["date"], df["close"], color="#1f77b4", linewidth=0.5)
    shade_regimes(ax)
    ax.set_title("Price with Market Regimes")
    ax.set_ylabel("Price (USDT)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper left", fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    vol_24  = df["log_return"].rolling(24).std()  * np.sqrt(8760)
    vol_168 = df["log_return"].rolling(168).std() * np.sqrt(8760)
    ax.plot(df["date"], vol_24,  linewidth=0.5, label="24h window",  alpha=0.8)
    ax.plot(df["date"], vol_168, linewidth=0.8, label="168h window")
    ax.set_title("Annualized Rolling Volatility")
    ax.set_ylabel("Volatility")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.bar(df["date"], df["volume_usdt"] / 1e9, width=0.04, color="#1f77b4", alpha=0.5)
    ax.set_title("Hourly Trading Volume")
    ax.set_ylabel("Volume (Billions USDT)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_page_two(pdf, returns):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Return Distributions", fontsize=14)

    ax = axes[0]
    ax.hist(returns, bins=200, density=True, color="#1f77b4", alpha=0.7)
    mu, sigma = returns.mean(), returns.std()
    x = np.linspace(returns.min(), returns.max(), 500)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), "r-", linewidth=1.5, label="Normal fit")
    ax.set_title(f"Log Return Distribution  (kurt={returns.kurt():.2f})")
    ax.set_xlabel("Log Return")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    (osm, osr), (slope, intercept, _) = stats.probplot(returns, dist="norm")
    ax.plot(osm, osr, ".", markersize=1, color="#1f77b4")
    ax.plot(osm, slope * np.array(osm) + intercept, "r-", linewidth=1.5)
    ax.set_title("Q-Q Plot vs Normal")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_page_three(pdf, returns):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Autocorrelation Analysis", fontsize=14)

    plot_acf( returns,        lags=48, ax=axes[0, 0], title="ACF - Log Returns")
    plot_pacf(returns,        lags=48, ax=axes[0, 1], title="PACF - Log Returns")
    plot_acf( returns ** 2,   lags=48, ax=axes[1, 0], title="ACF - Squared Returns (Volatility Clustering)")
    plot_acf( returns.abs(),  lags=48, ax=axes[1, 1], title="ACF - Absolute Returns")

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_page_four(pdf, price_res, return_res):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")

    rows = [
        ["ADF",  "Price",       f"{price_res['ADF Statistic']:.4f}",  f"{price_res['ADF p-value']:.4f}",
         "Non-stationary" if price_res["ADF p-value"] > 0.05 else "Stationary"],
        ["KPSS", "Price",       f"{price_res['KPSS Statistic']:.4f}", f"{price_res['KPSS p-value']:.4f}",
         "Non-stationary" if price_res["KPSS p-value"] < 0.05 else "Stationary"],
        ["ADF",  "Log Returns", f"{return_res['ADF Statistic']:.4f}", f"{return_res['ADF p-value']:.4f}",
         "Non-stationary" if return_res["ADF p-value"] > 0.05 else "Stationary"],
        ["KPSS", "Log Returns", f"{return_res['KPSS Statistic']:.4f}", f"{return_res['KPSS p-value']:.4f}",
         "Non-stationary" if return_res["KPSS p-value"] < 0.05 else "Stationary"],
    ]
    cols = ["Test", "Series", "Statistic", "p-value", "Conclusion"]

    table = ax.table(cellText=rows, colLabels=cols, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.5)
    ax.set_title("Stationarity Tests (ADF + KPSS)", fontsize=13, pad=20)

    pdf.savefig(fig)
    plt.close(fig)


def run_eda():
    os.makedirs("artifacts", exist_ok=True)

    df      = load_data(data_path)
    df      = compute_returns(df)
    returns = df["log_return"]

    price_res  = run_stationarity_tests(df["close"])
    return_res = run_stationarity_tests(returns)

    with PdfPages(output_path) as pdf:
        plot_page_one(pdf, df)
        plot_page_two(pdf, returns)
        plot_page_three(pdf, returns)
        plot_page_four(pdf, price_res, return_res)

        info = pdf.infodict()
        info["Title"]  = "BTC/USDT Exploratory Data Analysis"
        info["Author"] = "RL Trading Project"


if __name__ == "__main__":
    run_eda()