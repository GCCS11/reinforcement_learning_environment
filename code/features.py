import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange

folds = [
    {
        "fold":       1,
        "train_end":  "2021-12-31 23:00:00",
        "val_start":  "2022-01-01 00:00:00",
        "val_end":    "2022-12-31 23:00:00",
        "test_start": "2023-01-01 00:00:00",
        "test_end":   "2023-12-31 23:00:00",
    },
    {
        "fold":       2,
        "train_end":  "2022-12-31 23:00:00",
        "val_start":  "2023-01-01 00:00:00",
        "val_end":    "2023-12-31 23:00:00",
        "test_start": "2024-01-01 00:00:00",
        "test_end":   "2024-12-31 23:00:00",
    },
    {
        "fold":       3,
        "train_end":  "2023-12-31 23:00:00",
        "val_start":  "2024-01-01 00:00:00",
        "val_end":    "2024-12-31 23:00:00",
        "test_start": "2025-01-01 00:00:00",
        "test_end":   "2025-12-31 23:00:00",
    },
    {
        "fold":       4,
        "train_end":  "2024-12-31 23:00:00",
        "val_start":  "2025-01-01 00:00:00",
        "val_end":    "2025-12-31 23:00:00",
        "test_start": "2026-01-01 00:00:00",
        "test_end":   "2026-04-26 23:00:00",
    },
]

train_start = "2017-08-17 04:00:00"


def compute_features(df):
    df = df.copy()

    #returns
    df["log_return"]    = np.log(df["close"] / df["close"].shift(1))
    df["log_return_2h"] = np.log(df["close"] / df["close"].shift(2))
    df["log_return_4h"] = np.log(df["close"] / df["close"].shift(4))
    df["log_return_8h"] = np.log(df["close"] / df["close"].shift(8))
    df["log_return_24h"]= np.log(df["close"] / df["close"].shift(24))

    # rolling volatility — justified by ACF of squared returns in EDA
    df["vol_24h"]  = df["log_return"].rolling(24).std()
    df["vol_72h"]  = df["log_return"].rolling(72).std()
    df["vol_168h"] = df["log_return"].rolling(168).std()

    # rsi
    df["rsi_14"] = RSIIndicator(close=df["close"], window=14).rsi()
    df["rsi_28"] = RSIIndicator(close=df["close"], window=28).rsi()

    # macd
    macd = MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd_diff"] = macd.macd_diff()

    #bollinger bands
    bb = BollingerBands(close=df["close"], window=20, window_dev=2)
    df["bb_pct"] = bb.bollinger_pband()
    df["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / df["close"]

    #atr
    atr = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14)
    df["atr_pct"] = atr.average_true_range() / df["close"]

    # volume
    df["volume_ratio"] = df["volume_usdt"] / df["volume_usdt"].rolling(24).mean()


    df["hour_sin"] = np.sin(2 * np.pi * df["date"].dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["date"].dt.hour / 24)

    return df


feature_columns = [
    "log_return",
    "log_return_2h",
    "log_return_4h",
    "log_return_8h",
    "log_return_24h",
    "vol_24h",
    "vol_72h",
    "vol_168h",
    "rsi_14",
    "rsi_28",
    "macd_diff",
    "bb_pct",
    "bb_width",
    "atr_pct",
    "volume_ratio",
    "hour_sin",
    "hour_cos",
]


def get_fold_splits(df, fold_config):
    train = df[
        (df["date"] >= train_start) &
        (df["date"] <= fold_config["train_end"])
    ].copy()

    val = df[
        (df["date"] >= fold_config["val_start"]) &
        (df["date"] <= fold_config["val_end"])
    ].copy()

    test = df[
        (df["date"] >= fold_config["test_start"]) &
        (df["date"] <= fold_config["test_end"])
    ].copy()

    return train, val, test


def normalize_features(train, val, test):
    means = train[feature_columns].mean()
    stds  = train[feature_columns].std().replace(0, 1)

    for split in [train, val, test]:
        split[feature_columns] = (split[feature_columns] - means) / stds

    return train, val, test, means, stds