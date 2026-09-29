import pandas as pd

from black_scholes import black_scholes_call, black_scholes_put
from contrat_specs import (
    EQUITIES, FUTURES, BONDS, FACTEURS,
    OPTION_STRIKE, OPTION_EXPIRATION, OPTION_MULTIPLIER,
    OPTION_QUANTITIES, SIGMA_AAPL_IMPLICITE,
)
import market_data as md


def build_positions(valuation_date: pd.Timestamp = None, r: float = None) -> dict:
    if valuation_date is None:
        valuation_date = md.today()

    if r is None:
        r = md.fetch_risk_free_rate()

    positions = {}

    for name, info in EQUITIES.items():
        price = md.fetch_latest_equity_price(info["ticker"])
        positions[name] = {
            "kind": "equity", "qty": info["qty"], "price": price, "multiplier": 1,
        }

    for name, info in FUTURES.items():
        price = md.fetch_latest_future_price(info["ticker"])
        positions[name] = {
            "kind": "future", "qty": info["qty"], "price": price,
            "multiplier": info["multiplier"],
        }

    
    for name, info in BONDS.items():
        yield_rate = md.fetch_latest_treasury_yield(info["yield_series"])
        price = md.bond_price(
            face_value=1000, coupon_rate=info["coupon"], yield_rate=yield_rate,
            maturity=info["maturity"], valuation_date=valuation_date,
        )
        positions[name] = {
            "kind": "bond", "qty": info["qty"], "price": price, "multiplier": 1,
        }

    S = positions["AAPL"]["price"]
    K = OPTION_STRIKE
    T = max((OPTION_EXPIRATION - valuation_date).days, 0) / 365
    if T <= 0:
        raise ValueError(
            f"L'option AAPL 315 a expiré ({OPTION_EXPIRATION.date()}) par "
            f"rapport à la date de valorisation demandée ({valuation_date.date()})."
        )
    q_div = 1 / S  

    prix_call = black_scholes_call(S, K, T, r, q_div, SIGMA_AAPL_IMPLICITE)
    prix_put = black_scholes_put(S, K, T, r, q_div, SIGMA_AAPL_IMPLICITE)

    positions["AAPL 315 Call"] = {
        "kind": "option", "qty": OPTION_QUANTITIES["AAPL 315 Call"],
        "multiplier": OPTION_MULTIPLIER, "price": prix_call,
        "S": S, "K": K, "T": T, "q": q_div, "sigma": SIGMA_AAPL_IMPLICITE,
    }
    positions["AAPL 315 Put"] = {
        "kind": "option", "qty": OPTION_QUANTITIES["AAPL 315 Put"],
        "multiplier": OPTION_MULTIPLIER, "price": prix_put,
        "S": S, "K": K, "T": T, "q": q_div, "sigma": SIGMA_AAPL_IMPLICITE,
    }

    positions["_meta"] = {
        "valuation_date": valuation_date,
        "risk_free_rate": r,
        "time_to_maturity_call_put": T,
    }

    return positions


def market_values(positions: dict) -> pd.Series:
    mv = {
        name: p["qty"] * p["price"] * p["multiplier"]
        for name, p in positions.items()
        if name != "_meta"
    }
    return pd.Series(mv).reindex(FACTEURS)