import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from pandas_datareader import data as web


def today() -> pd.Timestamp:
    return pd.Timestamp(datetime.date.today())


def fetch_latest_close(ticker: str, price_column: str = "Adj Close") -> float:
    for period in ("5d", "10d", "1mo"):
        data = yf.download(ticker, period=period, progress=False, auto_adjust=False)
        if not data.empty and price_column in data.columns:
            col = data[price_column]
            if isinstance(col, pd.DataFrame):
                col = col.iloc[:, 0]
            col = col.dropna()
            if not col.empty:
                return float(col.iloc[-1])

    raise ValueError(
        f"Aucune donnée récupérée pour le ticker '{ticker}'."
    )


def fetch_latest_equity_price(ticker: str) -> float:
    return fetch_latest_close(ticker, price_column="Adj Close")


def fetch_latest_future_price(ticker: str) -> float:
    return fetch_latest_close(ticker, price_column="Close")


def fetch_latest_treasury_yield(series: str) -> float:
    end = datetime.date.today()
    start = end - datetime.timedelta(days=30)
    data = web.DataReader(series, "fred", start, end).dropna()
    if data.empty:
        raise ValueError(f"Aucune donnée FRED récupérée.")
    return float(data[series].iloc[-1]) / 100


def fetch_risk_free_rate() -> float:
    return fetch_latest_treasury_yield("DGS3MO")


def bond_price(face_value: float, coupon_rate: float, yield_rate: float,
                maturity: pd.Timestamp, valuation_date: pd.Timestamp) -> float:
    years_to_maturity = (maturity - valuation_date).days / 365
    if years_to_maturity <= 0:
        return face_value
    n = max(1, int(np.ceil(years_to_maturity * 2)))
    coupon = face_value * coupon_rate / 2
    y = yield_rate / 2
    return coupon * (1 - (1 + y) ** (-n)) / y + face_value * (1 + y) ** (-n)