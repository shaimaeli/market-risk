import datetime

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import yfinance as yf
from pandas_datareader import data as web

from black_scholes import black_scholes_call, black_scholes_put
from contrat_specs import EQUITIES, FUTURES, BONDS, FACTEURS

NB_JOURS_OUVRES = 252
OUTPUT_FILE = "Matrices_Risque_Portefeuille.xlsx"

OPTION_STRIKE = 315.0
OPTION_EXPIRATION = pd.Timestamp("2026-09-18")
VOL_WINDOW = 30
LAMBDA_EWMA_DEFAUT = 0.94

EQUITY_TICKERS = {info["ticker"]: name for name, info in EQUITIES.items()}


def _valider_start_date(start_date) -> str:
    if start_date is None or str(start_date).strip() == "":
        raise ValueError(
            "La date de début est obligatoire."
        )
    d = pd.Timestamp(start_date)
    aujourdhui = pd.Timestamp(datetime.date.today())
    if d >= aujourdhui:
        raise ValueError(
            f"La date de début ({d.date()}) doit être antérieure à "
            f"aujourd'hui ({aujourdhui.date()})."
        )
    if (aujourdhui - d).days < 400:
        raise ValueError(
            f"Historique trop court."
        )
    return d.date().isoformat()


def bond_price(face_value, coupon_rate, yield_rate, maturity, valuation_date):
    years_to_maturity = (maturity - valuation_date).days / 365
    if years_to_maturity <= 0:
        return face_value
    n = max(1, int(np.ceil(years_to_maturity * 2)))
    coupon = face_value * coupon_rate / 2
    y = yield_rate / 2
    return coupon * (1 - (1 + y) ** (-n)) / y + face_value * (1 + y) ** (-n)


def ewma_covariance_matrix(returns_df: pd.DataFrame, lambda_: float = 0.94) -> pd.DataFrame:
    assets = returns_df.columns
    r = returns_df.values
    n_obs = r.shape[0]
    cov = np.cov(r, rowvar=False)
    for t in range(n_obs):
        r_t = r[t, :].reshape(-1, 1)
        cov = lambda_ * cov + (1 - lambda_) * (r_t @ r_t.T)
    return pd.DataFrame(cov, index=assets, columns=assets)


def construire_donnees(start_date: str, output_file: str = OUTPUT_FILE,
                        lambda_ewma: float = LAMBDA_EWMA_DEFAUT,
                        generer_heatmap: bool = True,
                        log_callback=None) -> dict:
    def log(msg):
        if log_callback:
            log_callback(msg)

    start_date = _valider_start_date(start_date)
    end_date = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()

    log(f"Période demandée : {start_date} → {end_date}")

    log("Téléchargement des actions (AAPL, MSFT, LVMH)...")
    data_eq = yf.download(
        list(EQUITY_TICKERS.keys()), start=start_date, end=end_date,
        auto_adjust=False, progress=False,
    )
    if data_eq.empty:
        raise ValueError("Aucune donnée actions téléchargée.")

    prices_eq = data_eq["Adj Close"].rename(columns=EQUITY_TICKERS)
    prices_eq = prices_eq[list(EQUITY_TICKERS.values())].dropna(how="any")
    returns_eq = prices_eq.pct_change().dropna()
    n_theorique_eq = len(prices_eq)

    log("Téléchargement des yields Treasury (FRED)...")
    yields = web.DataReader(["DGS5", "DGS10"], "fred", start_date, end_date).dropna()
    yields_decimal = yields / 100

    prices_bonds = pd.DataFrame(index=yields_decimal.index)
    for bond_name, info in BONDS.items():
        prices_bonds[bond_name] = [
            bond_price(1000, info["coupon"], y, info["maturity"], date)
            for date, y in yields_decimal[info["yield_series"]].items()
        ]
    prices_bonds = prices_bonds.dropna()
    returns_bonds = prices_bonds.pct_change().dropna()

    log("Téléchargement des futures (S&P 500, Gold)...")
    tickers_fut = [info["ticker"] for info in FUTURES.values()]
    data_fut = yf.download(
        tickers_fut, start=start_date, end=end_date,
        auto_adjust=False, progress=False,
    )
    prices_fut = pd.DataFrame(index=data_fut.index)
    for name, info in FUTURES.items():
        prices_fut[name] = data_fut["Close"][info["ticker"]]
    prices_fut = prices_fut.dropna()
    returns_fut = prices_fut.pct_change().dropna()

    log("Construction des prix d'options AAPL 315 (Black-Scholes)...")
    aapl_close = data_eq["Adj Close"]["AAPL"].dropna()
    aapl_returns_daily = aapl_close.pct_change()
    volatility = aapl_returns_daily.rolling(window=VOL_WINDOW).std() * np.sqrt(NB_JOURS_OUVRES)

    rf_data = web.DataReader("DGS3MO", "fred", start_date, end_date).rename(
        columns={"DGS3MO": "Risk_Free_Rate"}
    ) / 100

    opt_data = pd.DataFrame({"AAPL": aapl_close, "Volatility": volatility})
    opt_data = opt_data.join(rf_data, how="left")
    opt_data["Risk_Free_Rate"] = opt_data["Risk_Free_Rate"].ffill()
    opt_data = opt_data.dropna()

    opt_data["Time_to_Maturity"] = [
        max((OPTION_EXPIRATION - date).days / 365, 0) for date in opt_data.index
    ]
    opt_data["q"] = 1 / opt_data["AAPL"]

    opt_data["AAPL 315 Call"] = [
        black_scholes_call(S, OPTION_STRIKE, T, r, q, sigma)
        for S, T, r, q, sigma in zip(
            opt_data["AAPL"], opt_data["Time_to_Maturity"],
            opt_data["Risk_Free_Rate"], opt_data["q"], opt_data["Volatility"],
        )
    ]
    opt_data["AAPL 315 Put"] = [
        black_scholes_put(S, OPTION_STRIKE, T, r, q, sigma)
        for S, T, r, q, sigma in zip(
            opt_data["AAPL"], opt_data["Time_to_Maturity"],
            opt_data["Risk_Free_Rate"], opt_data["q"], opt_data["Volatility"],
        )
    ]

    prices_opt = opt_data[["AAPL 315 Call", "AAPL 315 Put"]].dropna()
    returns_opt = prices_opt.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

    log("Fusion des rendements sur les dates communes...")
    returns = returns_eq.join(returns_fut, how="inner")
    returns = returns.join(returns_opt, how="inner")
    returns = returns.join(returns_bonds, how="inner")
    returns = returns.dropna()

    manquants = [a for a in FACTEURS if a not in returns.columns]
    if manquants:
        raise ValueError(f"Colonnes manquantes par rapport à positions.FACTEURS : {manquants}")
    returns = returns[FACTEURS]

    n_obs = len(returns)
    if n_obs < 60:
        raise ValueError(
            f"Seulement {n_obs} jours communs à toutes les sources après "
            f"fusion : période trop courte ou trop ancienne pour être "
            f"exploitable. Choisissez une date de début plus récente ou "
            f"vérifiez la disponibilité des données."
        )

    completude = {
        "AAPL": len(returns_eq), "MSFT": len(returns_eq), "LVMH": len(returns_eq),
        "S&P 500 Future": len(returns_fut), "Gold Future": len(returns_fut),
        "AAPL 315 Call": len(returns_opt), "AAPL 315 Put": len(returns_opt),
        "Treasury 5Y": len(returns_bonds), "Treasury 10Y": len(returns_bonds),
    }
    completude_pct = {k: round(100 * n_obs / v, 1) if v else 0.0 for k, v in completude.items()}

    log("Calcul des matrices de covariance et de corrélation...")
    covariance_matrix = returns.cov()
    annual_covariance_matrix = covariance_matrix * NB_JOURS_OUVRES
    correlation_matrix = returns.corr()

    daily_vol = returns.std()
    annual_vol = daily_vol * np.sqrt(NB_JOURS_OUVRES)
    volatility_summary = pd.DataFrame({
        "Volatilité journalière (%)": (daily_vol * 100).round(4),
        "Volatilité annualisée (%)": (annual_vol * 100).round(4),
    })

    stats = returns.describe().T
    stats["Rendement moyen annuel (%)"] = (returns.mean() * NB_JOURS_OUVRES * 100).round(4)
    stats["Sharpe approx."] = (returns.mean() / returns.std() * np.sqrt(NB_JOURS_OUVRES)).round(4)

    log(f"Calcul de la covariance EWMA (lambda={lambda_ewma})...")
    ewma_daily_cov = ewma_covariance_matrix(returns, lambda_=lambda_ewma)
    ewma_annual_cov = ewma_daily_cov * NB_JOURS_OUVRES

    heatmap_path = None
    log("Génération de la heatmap de corrélation...")
    plt.figure(figsize=(11, 9))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
    sns.heatmap(
            correlation_matrix, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
            vmin=-1, vmax=1, square=True, linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": "Corrélation"}, mask=mask,
        )
    plt.title(
            f"Corrélation du portefeuille\n({returns.index.min().date()} → "
            f"{returns.index.max().date()} | {n_obs} jours)",
            fontsize=12, fontweight="bold", pad=16,
        )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    heatmap_path = "static/heatmap_correlation.png"
    plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close()

    log(f"Export vers {output_file}...")
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        returns.to_excel(writer, sheet_name="Rendements_fusionnes")
        covariance_matrix.to_excel(writer, sheet_name="Covariance_journaliere")
        annual_covariance_matrix.to_excel(writer, sheet_name="Covariance_annualisee")
        ewma_daily_cov.to_excel(writer, sheet_name="Covariance_EWMA_journaliere")
        ewma_annual_cov.to_excel(writer, sheet_name="Covariance_EWMA_annualisee")
        correlation_matrix.to_excel(writer, sheet_name="Correlation")
        volatility_summary.to_excel(writer, sheet_name="Volatilites")
        stats.to_excel(writer, sheet_name="Statistiques_descriptives")

    return {
        "start_date": start_date,
        "end_date": returns.index.max().date().isoformat(),
        "n_obs": n_obs,
        "actifs": list(returns.columns),
        "completude_pct": completude_pct,
        "heatmap_path": heatmap_path,
        "output_file": output_file,
        "lambda_ewma": lambda_ewma,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        sys.exit(1)
    meta = construire_donnees(sys.argv[1])
    print(meta)
