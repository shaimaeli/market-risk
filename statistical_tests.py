from scipy.stats import binom


def traffic_light_zone(n_violations: int, n_obs: int = 250,
                       confidence: float = 0.99) -> dict:
    """Classification de Bâle (zone verte, jaune ou rouge).

    Les seuils de Bâle (4 et 9 violations) sont définis pour 250 jours et un
    niveau de confiance de 99 %. Pour une autre durée de test, le nombre de
    violations est ramené à un équivalent sur 250 jours.
    """
    n_violations_250 = n_violations * 250 / n_obs

    if n_violations_250 <= 4:
        zone = "Verte"
        commentaire = "Le nombre de violations est conforme aux attentes."
    elif n_violations_250 <= 9:
        zone = "Jaune"
        commentaire = "Zone de surveillance"
    else:
        zone = "Rouge"
        commentaire = "Le modèle sous-estime significativement le risque."

    p_attendu = 1 - confidence
    p_value_binom = 1 - binom.cdf(n_violations - 1, n_obs, p_attendu)

    return {
        "n_violations": n_violations,
        "n_obs": n_obs,
        "n_violations_equivalent_250j": round(n_violations_250, 2),
        "zone": zone,
        "commentaire": commentaire,
        "p_value_binomiale": p_value_binom,
    }