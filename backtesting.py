import numpy as np
import pandas as pd

from positions import FACTEURS
from var_param import var_es_parametrique
from var_mc import var_es_montecarlo


def historical_pnl(mv: pd.Series, rendements: pd.DataFrame) -> pd.Series:
    rendements = rendements.reindex(columns=FACTEURS)
    mv = mv.reindex(FACTEURS)
    return rendements.dot(mv)


def rolling_backtest(rendements: pd.DataFrame, positions: dict, mv: pd.Series,
                      method: str = "parametrique", window: int = 100,
                      confidence: float = 0.99, n_scenarios: int = 50_000,
                      seed: int = 42, cov_method: str = "rolling",
                      lambda_ewma: float = 0.94) -> pd.DataFrame:
    
    rendements = rendements.reindex(columns=FACTEURS).dropna()
    mv = mv.reindex(FACTEURS)
    n = len(rendements)

    if n <= window:
        raise ValueError(
            f"Pas assez de données ({n} jours) pour une fenêtre de "
            f"{window} jours : il ne resterait aucun jour à tester "
            f"hors-échantillon. Réduis `window` ou utilise un historique "
            f"plus long (voir build_data.py)."
        )

    ewma_cov_par_jour = {}
    if cov_method == "ewma":
        cov_running = rendements.iloc[:window].cov().values
        for t in range(window, n):
            r_prev = rendements.iloc[t - 1].values.reshape(-1, 1)
            cov_running = lambda_ewma * cov_running + (1 - lambda_ewma) * (r_prev @ r_prev.T)
            ewma_cov_par_jour[t] = pd.DataFrame(
                cov_running * 252, index=FACTEURS, columns=FACTEURS
            )

    dates, var_values, pnl_values = [], [], []

    for t in range(window, n):
        if method in ("parametrique", "montecarlo"):
            if cov_method == "ewma":
                cov_t = ewma_cov_par_jour[t]
            else:  # "rolling"
                fenetre = rendements.iloc[t - window:t]
                cov_t = fenetre.cov() * 252

        if method == "parametrique":
            VaR_t = var_es_parametrique(mv, cov_t, confidence)["VaR"]

        elif method == "historique":
            fenetre = rendements.iloc[t - window:t]
            poids = mv / mv.sum()
            rp_fenetre = fenetre.dot(poids)
            VaR_t = -rp_fenetre.quantile(1 - confidence) * mv.sum()

        elif method == "montecarlo":
            VaR_t = var_es_montecarlo(
                positions, cov_t, n_scenarios=n_scenarios,
                confidence=confidence, seed=seed,
            )["VaR"]

        else:
            raise ValueError("method doit être 'parametrique', 'historique' ou 'montecarlo'")

        pnl_t = rendements.iloc[t].dot(mv)  # perte/gain réellement observé ce jour-là

        dates.append(rendements.index[t])
        var_values.append(VaR_t)
        pnl_values.append(pnl_t)

    resultat = pd.DataFrame({"VaR": var_values, "pnl": pnl_values}, index=dates)
    resultat["violation"] = resultat["pnl"] < -resultat["VaR"]

    return resultat


def summarize_backtest(resultat: pd.DataFrame) -> dict:
    n_obs = len(resultat)
    n_violations = int(resultat["violation"].sum())
    dates_violation = resultat.index[resultat["violation"]].tolist()

    return {
        "n_obs": n_obs,
        "n_violations": n_violations,
        "taux_violation": n_violations / n_obs,
        "dates_violation": dates_violation,
    }