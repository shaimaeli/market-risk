import numpy as np
from scipy.stats import chi2, binom


def kupiec_test(n_obs: int, n_violations: int, confidence: float = 0.99) -> dict:
    
    p_attendu = 1 - confidence
    x = n_violations
    n = n_obs

    if x == 0:
        taux_observe = 0.0
        log_vraisemblance_h0 = n * np.log(1 - p_attendu)
        log_vraisemblance_h1 = n * np.log(1 - p_attendu)  # x/n = 0 aussi
        LR = 0.0
    else:
        taux_observe = x / n
        log_vraisemblance_h0 = (n - x) * np.log(1 - p_attendu) + x * np.log(p_attendu)
        log_vraisemblance_h1 = (n - x) * np.log(1 - taux_observe) + x * np.log(taux_observe)
        LR = -2 * (log_vraisemblance_h0 - log_vraisemblance_h1)

    p_value = 1 - chi2.cdf(LR, df=1)
    rejet_h0 = p_value < 0.05  

    if rejet_h0:
        conclusion = (
            "Le modèle est REJETÉ : le taux de violation "
            "observé est statistiquement différent du taux attendu."
        )
    else:
        conclusion = (
            "Le modèle N'EST PAS REJETÉ : le taux de "
            "violation observé est statistiquement cohérent avec le "
            "taux attendu."
        )

    return {
        "n_obs": n,
        "n_violations": x,
        "taux_observe": taux_observe,
        "taux_attendu": p_attendu,
        "LR_stat": LR,
        "p_value": p_value,
        "rejet_h0": rejet_h0,
        "conclusion": conclusion,
    }


def traffic_light_zone(n_violations: int, n_obs: int = 250,
                        confidence: float = 0.99) -> dict:
    
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


def christoffersen_independence_test(violations) -> dict:
   
    v = np.asarray(violations).astype(int)
    n00 = n01 = n10 = n11 = 0

    for t in range(1, len(v)):
        prev, curr = v[t - 1], v[t]
        if prev == 0 and curr == 0:
            n00 += 1
        elif prev == 0 and curr == 1:
            n01 += 1
        elif prev == 1 and curr == 0:
            n10 += 1
        elif prev == 1 and curr == 1:
            n11 += 1

    n0 = n00 + n01  
    n1 = n10 + n11  

    if n0 == 0 or n1 == 0 or n01 == 0 or n11 == 0:
       
        return {
            "LR_stat": 0.0,
            "p_value": 1.0,
            "rejet_h0": False,
            "n_transitions": {"n00": n00, "n01": n01, "n10": n10, "n11": n11},
            "conclusion": (
                "Pas assez de transitions violation->violation ou "
                "non-violation->violation pour tester l'indépendance "
                "de façon fiable (trop peu de violations observées)."
            ),
        }

    pi01 = n01 / n0
    pi11 = n11 / n1
    pi = (n01 + n11) / (n0 + n1)

    ll_h0 = n00 * np.log(1 - pi) + n01 * np.log(pi) + n10 * np.log(1 - pi) + n11 * np.log(pi)
    ll_h1 = n00 * np.log(1 - pi01) + n01 * np.log(pi01) + n10 * np.log(1 - pi11) + n11 * np.log(pi11)

    LR = -2 * (ll_h0 - ll_h1)
    p_value = 1 - chi2.cdf(LR, df=1)
    rejet_h0 = p_value < 0.05

    if rejet_h0:
        conclusion = (
            "H0 REJETÉE"
        )
    else:
        conclusion = (
            "H0 non rejetée "
        )

    return {
        "LR_stat": LR,
        "p_value": p_value,
        "rejet_h0": rejet_h0,
        "n_transitions": {"n00": n00, "n01": n01, "n10": n10, "n11": n11},
        "conclusion": conclusion,
    }