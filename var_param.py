import numpy as np
import pandas as pd
from scipy.stats import norm


def var_es_parametrique(mv: pd.Series, matrice_covariance: pd.DataFrame,
                         confidence: float = 0.99) :
    mv = mv.reindex(matrice_covariance.index)

    variance_annuelle = mv.T @ matrice_covariance @ mv
    vol_annuelle = np.sqrt(variance_annuelle)
    vol_jour = vol_annuelle / np.sqrt(252)

    z = norm.ppf(confidence)
    VaR = z * vol_jour
    ES = vol_jour * norm.pdf(z) / (1 - confidence)

    return {
        "VaR": VaR,
        "ES": ES,
        "vol_jour": vol_jour,
        "vol_annuelle": vol_annuelle,
    }