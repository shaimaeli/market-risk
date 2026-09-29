import pandas as pd

EQUITIES = {
    "AAPL": {"ticker": "AAPL", "qty": 4765},
    "MSFT": {"ticker": "MSFT", "qty": 3905},
    "LVMH": {"ticker": "MC.PA", "qty": 1811},
}

FUTURES = {
    "S&P 500 Future": {"ticker": "ES=F", "multiplier": 50, "qty": 3},
    "Gold Future": {"ticker": "GC=F", "multiplier": 100, "qty": 2},
}

BONDS = {
    "Treasury 5Y": {
        "coupon": 0.0425,
        "maturity": pd.Timestamp("2031-06-30"),
        "yield_series": "DGS5",
        "qty": 1504,
    },
    "Treasury 10Y": {
        "coupon": 0.0425,
        "maturity": pd.Timestamp("2035-05-15"),
        "yield_series": "DGS10",
        "qty": 1506,
    },
}

OPTION_STRIKE = 315.0
OPTION_EXPIRATION = pd.Timestamp("2027-09-17")
OPTION_MULTIPLIER = 100
SIGMA_AAPL_IMPLICITE = 0.2734  
OPTION_QUANTITIES = {"AAPL 315 Call": 321, "AAPL 315 Put": 367}

FACTEURS = [
    "AAPL", "MSFT", "LVMH", "S&P 500 Future", "Gold Future",
    "AAPL 315 Call", "AAPL 315 Put", "Treasury 5Y", "Treasury 10Y",
]