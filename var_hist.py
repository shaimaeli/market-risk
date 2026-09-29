import pandas as pd

from positions import FACTEURS


def var_es_historique(mv: pd.Series, rendements: pd.DataFrame,
                       confidence: float = 0.99) -> dict:
    mv = mv.reindex(FACTEURS)
    valeur_portefeuille = mv.sum()
    poids = mv / valeur_portefeuille

    rendements = rendements.reindex(columns=FACTEURS)
    rendement_portefeuille = rendements.dot(poids)

    q = rendement_portefeuille.quantile(1 - confidence)
    VaR_pct = -q
    queue = rendement_portefeuille[rendement_portefeuille <= q]
    ES_pct = -queue.mean()

    return {
        "VaR_pct": VaR_pct,
        "ES_pct": ES_pct,
        "VaR": VaR_pct * valeur_portefeuille,
        "ES": ES_pct * valeur_portefeuille,
        "valeur_portefeuille": valeur_portefeuille,
        "poids": poids,
        "n_obs_queue": len(queue),
    }