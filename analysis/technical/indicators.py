"""
analysis/technical/indicators.py — Indicateurs techniques via pandas-ta.

Indicateurs calculés : RSI, MACD, EMA (20/50/200), Bollinger Bands, ATR.
"""
from __future__ import annotations

import pandas as pd
import pandas_ta as ta  # type: ignore[import]


class TechnicalIndicators:
    """Calcule les indicateurs techniques sur un DataFrame OHLCV."""

    @staticmethod
    def add_all(df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajoute tous les indicateurs au DataFrame.
        df doit avoir les colonnes : open, high, low, close, volume.
        Retourne une copie enrichie.
        """
        df = df.copy()

        # RSI
        df["rsi_14"] = ta.rsi(df["close"], length=14)

        # EMA
        df["ema_20"]  = ta.ema(df["close"], length=20)
        df["ema_50"]  = ta.ema(df["close"], length=50)
        df["ema_200"] = ta.ema(df["close"], length=200)

        # MACD (12, 26, 9)
        macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
        if macd is not None:
            df["macd"]        = macd["MACD_12_26_9"]
            df["macd_signal"] = macd["MACDs_12_26_9"]
            df["macd_hist"]   = macd["MACDh_12_26_9"]

        # Bollinger Bands (20, 2)
        bbands = ta.bbands(df["close"], length=20, std=2)
        if bbands is not None:
            df["bb_upper"] = bbands["BBU_20_2.0"]
            df["bb_mid"]   = bbands["BBM_20_2.0"]
            df["bb_lower"] = bbands["BBL_20_2.0"]
            df["bb_pct"]   = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-8)

        # ATR (Average True Range, 14)
        df["atr_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)

        # ADX (Average Directional Index, 14) — force et direction de la tendance
        adx_df = ta.adx(df["high"], df["low"], df["close"], length=14)
        if adx_df is not None:
            df["adx_14"] = adx_df.get("ADX_14", adx_df.iloc[:, 0])
            df["dmp_14"] = adx_df.get("DMP_14", adx_df.iloc[:, 1])  # +DI
            df["dmn_14"] = adx_df.get("DMN_14", adx_df.iloc[:, 2])  # -DI

        # Volume SMA
        df["volume_sma_20"] = df["volume"].rolling(20).mean()
        df["volume_ratio"]  = df["volume"] / (df["volume_sma_20"] + 1e-8)

        return df

    @staticmethod
    def rsi_signal(rsi: float, oversold: float = 30.0, overbought: float = 70.0) -> str:
        """Classifie le RSI en signal qualitatif."""
        if rsi < oversold:
            return "oversold"
        elif rsi > overbought:
            return "overbought"
        return "neutral"

    @staticmethod
    def ema_trend(close: float, ema_20: float, ema_50: float, ema_200: float) -> str:
        """Détermine la tendance par alignement des EMA."""
        if close > ema_20 > ema_50 > ema_200:
            return "strong_uptrend"
        elif close < ema_20 < ema_50 < ema_200:
            return "strong_downtrend"
        elif close > ema_50:
            return "uptrend"
        elif close < ema_50:
            return "downtrend"
        return "sideways"

    @staticmethod
    def atr_stops(close: float, atr: float, sl_mult: float = 2.0, tp_mult: float = 3.0) -> dict:
        """Calcule SL et TP dynamiques basés sur l'ATR."""
        return {
            "stop_loss_long":   close - atr * sl_mult,
            "take_profit_long": close + atr * tp_mult,
            "stop_loss_short":   close + atr * sl_mult,
            "take_profit_short": close - atr * tp_mult,
        }
