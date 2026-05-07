import numpy as np
import pandas as pd
import yfinance as yf

transaction_cost = 0.00125


def btc_buy_and_hold(df):

    entry_price  = float(df.iloc[0]["open"])
    equity_curve = df["close"] / entry_price
    equity_curve = pd.Series(equity_curve.values, index=pd.to_datetime(df["date"].values))

    # apply transaction costs only at entry and exit
    equity_curve = equity_curve * (1.0 - transaction_cost)
    equity_curve.iloc[-1] *= (1.0 - transaction_cost)

    log_returns = np.log(df["close"] / df["close"].shift(1)).dropna()
    log_returns = pd.Series(log_returns.values)

    return equity_curve, log_returns


def sp500_buy_and_hold(start_date, end_date):

    hist = yf.download("SPY", start=str(start_date)[:10], end=str(end_date)[:10],
                       interval="1d", progress=False, auto_adjust=True)

    if hist.empty:
        return None, None

    hist.index = pd.to_datetime(hist.index).tz_localize(None)

    entry_price  = float(hist["Open"].iloc[0])
    daily_equity = hist["Close"] / entry_price
    daily_equity = daily_equity * (1.0 - transaction_cost)
    daily_equity.iloc[-1] *= (1.0 - transaction_cost)

    hourly_index  = pd.date_range(start=daily_equity.index[0],
                                  end=daily_equity.index[-1], freq="1h")
    equity_hourly = daily_equity.reindex(hourly_index).ffill()

    log_returns = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()

    return equity_hourly, log_returns