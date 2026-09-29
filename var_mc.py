import numpy as np
import pandas as pd

from positions import FACTEURS, market_values


def var_es_montecarlo(positions: dict,
                       matrice_covariance: pd.DataFrame,
                       n_scenarios: int = 500_000,
                       confidence: float = 0.99,
                       seed: int = 42) -> dict:
    cov = matrice_covariance.loc[FACTEURS, FACTEURS]
    cov_jour = cov / 252
    rng = np.random.default_rng(seed)
    chocs = rng.multivariate_normal(
        mean=np.zeros(len(FACTEURS)),
        cov=cov_jour,
        size=n_scenarios,
    )

    valeurs_marche = market_values(positions)  

    pnl_par_facteur = chocs * valeurs_marche.values
    pnl_total = pnl_par_facteur.sum(axis=1)

    quantile = np.quantile(pnl_total, 1 - confidence)
    VaR = -quantile

    queue = pnl_total[pnl_total <= quantile]
    ES = -queue.mean()

    return {
        "VaR": VaR,
        "ES": ES,
        "pnl_total": pnl_total,
        "n_obs_queue": len(queue),
    }