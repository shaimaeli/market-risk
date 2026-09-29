import numpy as np
import pandas as pd

from positions import FACTEURS


def portfolio_metrics(mv: pd.Series, matrice_covariance: pd.DataFrame,
                       rendements_annuels_pct: pd.Series) -> dict:
    mv = mv.reindex(FACTEURS)
    poids = mv / mv.sum()

    rendements_annuels_pct = rendements_annuels_pct.reindex(FACTEURS)
    rendement_annuel_pct = rendements_annuels_pct.dot(poids)

    cov = matrice_covariance.reindex(index=FACTEURS, columns=FACTEURS)
    variance = poids.T @ cov @ poids
    volatilite_annuelle_pct = np.sqrt(variance) * 100

    return {
        "rendement_annuel_pct": rendement_annuel_pct,
        "volatilite_annuelle_pct": volatilite_annuelle_pct,
        "poids": poids,
    }