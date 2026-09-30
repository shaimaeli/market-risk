import io
import os
import math
import datetime
import traceback

from flask import Flask, request, jsonify, render_template, send_file
import numpy as np
import pandas as pd

from build_data import construire_donnees
from data_loader import load_risk_data
from positions import build_positions, market_values, FACTEURS
from var_param import var_es_parametrique
from var_hist import var_es_historique
from var_mc import var_es_montecarlo
from portefeuille_info import portfolio_metrics
from backtesting import rolling_backtest, summarize_backtest
from test import traffic_light_zone

app = Flask(__name__)

ETAT = {}   # garde en mémoire l'analyse en cours


# =========================================================
# FONCTIONS UTILITAIRES
# =========================================================

def nettoyer(v):
    """Convertit un nombre numpy en nombre Python. Les NaN deviennent None (JSON valide)."""
    if isinstance(v, (np.floating, np.integer)):
        v = v.item()
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def var_en_pct(var_usd, valeur_portefeuille):
    if not valeur_portefeuille:
        return 0.0
    return 100 * var_usd / valeur_portefeuille


def echantillonner(serie, n_max):
    """Garde environ n_max points d'une série (pour alléger les graphiques)."""
    pas = max(1, len(serie) // n_max)
    return serie.iloc[::pas]


def erreur_si_pas_d_analyse():
    if not ETAT:
        return jsonify({"ok": False, "error": "Aucune analyse chargée."}), 409
    return None


def resume_analyse():
    """Infos affichées dans la barre du haut."""
    meta = ETAT["meta"]
    return {
        "ok": True,
        "chargee": True,
        "meta": {
            "start_date": meta["start_date"],
            "end_date": meta["end_date"],
            "n_obs": meta["n_obs"],
            "lambda_ewma": meta["lambda_ewma"],
        },
        "confidence": ETAT["confidence"],
        "n_scenarios_mc": ETAT["n_scenarios_mc"],
        "valeur_portefeuille": float(ETAT["mv"].sum()),
    }


# =========================================================
# ANALYSE
# =========================================================

def lancer_analyse(start_date, confidence, lambda_ewma, n_scenarios_mc):
    """Construit les données, calcule les VaR et stocke tout dans ETAT."""
    journal = []
    meta = construire_donnees(start_date, lambda_ewma=lambda_ewma, log_callback=journal.append)

    matrice_cov, matrice_cov_ewma, rendements = load_risk_data(meta["output_file"])

    positions = build_positions()
    mv = market_values(positions)

    resultats_var = {
        "Paramétrique (simple)": var_es_parametrique(mv, matrice_cov, confidence),
        "Paramétrique (EWMA)": var_es_parametrique(mv, matrice_cov_ewma, confidence),
        "Historique": var_es_historique(mv, rendements, confidence),
        "Monte Carlo (simple)": var_es_montecarlo(
            positions, matrice_cov, n_scenarios=n_scenarios_mc, confidence=confidence),
        "Monte Carlo (EWMA)": var_es_montecarlo(
            positions, matrice_cov_ewma, n_scenarios=n_scenarios_mc, confidence=confidence),
    }

    stats_desc = pd.read_excel(meta["output_file"], sheet_name="Statistiques_descriptives", index_col=0)
    vol_summary = pd.read_excel(meta["output_file"], sheet_name="Volatilites", index_col=0)
    rendements_annuels_pct = stats_desc["Rendement moyen annuel (%)"]

    ETAT.clear()
    ETAT.update({
        "meta": meta,
        "confidence": confidence,
        "n_scenarios_mc": n_scenarios_mc,
        "matrice_cov": matrice_cov,
        "matrice_cov_ewma": matrice_cov_ewma,
        "rendements": rendements,
        "positions": positions,
        "mv": mv,
        "resultats_var": resultats_var,
        "stats_desc": stats_desc,
        "vol_summary": vol_summary,
        "metrics_simple": portfolio_metrics(mv, matrice_cov, rendements_annuels_pct),
        "metrics_ewma": portfolio_metrics(mv, matrice_cov_ewma, rendements_annuels_pct),
        "backtest": None,
    })


# =========================================================
# ROUTES
# =========================================================

@app.route("/")
def accueil():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    if not ETAT:
        return jsonify({"ok": True, "chargee": False})
    return jsonify(resume_analyse())


@app.route("/api/analyse", methods=["POST"])
def api_analyse():
    d = request.get_json(force=True, silent=True) or {}

    start_date = str(d.get("start_date", "")).strip()
    if not start_date:
        return jsonify({"ok": False, "error": "La date de début est obligatoire."}), 400

    try:
        confidence = float(d.get("confidence", 0.99))
        lambda_ewma = float(d.get("lambda_ewma", 0.94))
        n_scenarios_mc = int(d.get("n_scenarios_mc", 50000))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Paramètres numériques invalides."}), 400

    try:
        lancer_analyse(start_date, confidence, lambda_ewma, n_scenarios_mc)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": f"Échec de la construction des données : {e}"}), 400

    return jsonify(resume_analyse())


@app.route("/api/dashboard")
def api_dashboard():
    erreur = erreur_si_pas_d_analyse()
    if erreur:
        return erreur

    mv = ETAT["mv"]
    valeur = float(mv.sum())
    poids = mv / valeur
    vol_summary = ETAT["vol_summary"]
    resultats_var = ETAT["resultats_var"]
    metrics_ewma = ETAT["metrics_ewma"]
    actifs_presents = [a for a in FACTEURS if a in mv.index]

    # un résumé par actif
    actifs = []
    for actif in actifs_presents:
        vol = vol_summary.loc[actif, "Volatilité annualisée (%)"] if actif in vol_summary.index else None
        actifs.append({
            "nom": actif,
            "vol_pct": nettoyer(vol),
            "poids_pct": nettoyer(100 * poids[actif]),
            "exposition": nettoyer(mv[actif]),
        })
    actifs.sort(key=lambda a: a["vol_pct"] or 0, reverse=True)

    # valeur du portefeuille dans le temps (poids actuels appliqués à l'historique)
    rendements = ETAT["rendements"].reindex(columns=FACTEURS)
    rendements_portefeuille = rendements.dot(mv.reindex(FACTEURS) / valeur)
    evolution = echantillonner((1 + rendements_portefeuille).cumprod() * valeur, 180)

    return jsonify({
        "ok": True,
        "valeur_portefeuille": valeur,
        "var_historique": nettoyer(resultats_var["Historique"]["VaR"]),
        "var_parametrique_ewma": nettoyer(resultats_var["Paramétrique (EWMA)"]["VaR"]),
        "vol_annuelle_pct": nettoyer(metrics_ewma["volatilite_annuelle_pct"]),
        "rendement_annuel_pct": nettoyer(metrics_ewma["rendement_annuel_pct"]),
        "actifs": actifs,
        "evolution": {
            "dates": [d.strftime("%Y-%m-%d") for d in evolution.index],
            "valeurs": [nettoyer(round(v, 2)) for v in evolution.values],
        },
        "repartition": {
            "labels": actifs_presents,
            "poids_pct": [nettoyer(round(100 * poids[a], 2)) for a in actifs_presents],
        },
    })


@app.route("/api/donnees")
def api_donnees():
    erreur = erreur_si_pas_d_analyse()
    if erreur:
        return erreur

    stats_desc = ETAT["stats_desc"]
    vol_summary = ETAT["vol_summary"]
    meta = ETAT["meta"]
    actifs = [a for a in FACTEURS if a in stats_desc.index]

    lignes = []
    for actif in actifs:
        lignes.append({
            "actif": actif,
            "completude": nettoyer(meta["completude_pct"].get(actif)),
            "rendement_moyen_annuel": nettoyer(stats_desc.loc[actif, "Rendement moyen annuel (%)"]),
            "vol_journaliere": nettoyer(vol_summary.loc[actif, "Volatilité journalière (%)"]),
            "vol_annualisee": nettoyer(vol_summary.loc[actif, "Volatilité annualisée (%)"]),
            "min": nettoyer(stats_desc.loc[actif, "min"] * 100),
            "max": nettoyer(stats_desc.loc[actif, "max"] * 100),
        })

    # rendement cumulé de chaque actif (base 1)
    cumul = echantillonner((1 + ETAT["rendements"].reindex(columns=FACTEURS)).cumprod(), 150)

    return jsonify({
        "ok": True,
        "actifs": actifs,
        "lignes": lignes,
        "cumul": {
            "dates": [d.strftime("%Y-%m-%d") for d in cumul.index],
            "series": {col: [nettoyer(round(v, 4)) for v in cumul[col].values] for col in cumul.columns},
        },
    })


@app.route("/api/covariance")
def api_covariance():
    erreur = erreur_si_pas_d_analyse()
    if erreur:
        return erreur

    methode = request.args.get("methode", "simple")
    if methode == "simple":
        cov, metrics = ETAT["matrice_cov"], ETAT["metrics_simple"]
    else:
        cov, metrics = ETAT["matrice_cov_ewma"], ETAT["metrics_ewma"]

    cov = cov.reindex(index=FACTEURS, columns=FACTEURS)
    corr = ETAT["rendements"].reindex(columns=FACTEURS).corr()

    def en_liste(matrice, decimales):
        return [[nettoyer(round(matrice.loc[i, j], decimales)) for j in matrice.columns]
                for i in matrice.index]

    return jsonify({
        "ok": True,
        "actifs": FACTEURS,
        "cov_matrix": en_liste(cov, 6),
        "corr_matrix": en_liste(corr, 3),
        "lambda_ewma": ETAT["meta"]["lambda_ewma"],
    })


@app.route("/api/var")
def api_var():
    erreur = erreur_si_pas_d_analyse()
    if erreur:
        return erreur

    valeur = float(ETAT["mv"].sum())

    lignes = []
    for methode, r in ETAT["resultats_var"].items():
        lignes.append({
            "methode": methode,
            "VaR": nettoyer(r["VaR"]),
            "ES": nettoyer(r["ES"]),
            "VaR_pct": nettoyer(var_en_pct(r["VaR"], valeur)),
            "ES_pct": nettoyer(var_en_pct(r["ES"], valeur)),
        })

    return jsonify({"ok": True, "lignes": lignes})


@app.route("/api/backtest", methods=["POST"])
def api_backtest():
    erreur = erreur_si_pas_d_analyse()
    if erreur:
        return erreur

    d = request.get_json(force=True, silent=True) or {}
    try:
        fenetre = int(d.get("window", 250))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Fenêtre invalide."}), 400
    inclure_mc = bool(d.get("inclure_mc", False))

    rendements = ETAT["rendements"]
    confidence = ETAT["confidence"]

    if len(rendements) - fenetre < 30:
        return jsonify({"ok": False, "error": "Réduisez la fenêtre ou allongez l'historique."}), 400

    # (nom affiché, méthode de VaR, méthode de covariance)
    configs = [
        ("Paramétrique (simple)", "parametrique", "rolling"),
        ("Paramétrique (EWMA)", "parametrique", "ewma"),
        ("Historique", "historique", "rolling"),
    ]
    if inclure_mc:
        configs += [
            ("Monte Carlo (simple)", "montecarlo", "rolling"),
            ("Monte Carlo (EWMA)", "montecarlo", "ewma"),
        ]

    resultats = {}
    try:
        for nom, methode, cov_method in configs:
            bt = rolling_backtest(
                rendements, ETAT["positions"], ETAT["mv"], method=methode, window=fenetre,
                confidence=confidence, n_scenarios=10_000, cov_method=cov_method)
            resume = summarize_backtest(bt)
            feu = traffic_light_zone(resume["n_violations"], resume["n_obs"], confidence)
            resultats[nom] = {"bt": bt, "resume": resume, "feu": feu}
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": f"Échec du backtest : {e}"}), 400

    ETAT["backtest"] = {"window": fenetre, "resultats": resultats}

    lignes = []
    for nom, r in resultats.items():
        lignes.append({
            "label": nom,
            "n_obs": r["resume"]["n_obs"],
            "violations": r["resume"]["n_violations"],
            "taux_observe_pct": nettoyer(r["resume"]["taux_violation"] * 100),
            "taux_attendu_pct": nettoyer((1 - confidence) * 100),
            "zone": r["feu"]["zone"],
        })

    # le graphique montre la méthode historique (ou la première disponible)
    nom_graphique = "Historique" if "Historique" in resultats else next(iter(resultats))
    bt = echantillonner(resultats[nom_graphique]["bt"], 200)

    return jsonify({
        "ok": True,
        "lignes": lignes,
        "chart": {
            "methode": nom_graphique,
            "dates": [d.strftime("%Y-%m-%d") for d in bt.index],
            "pnl": [nettoyer(round(v, 2)) for v in bt["pnl"].values],
            "var_neg": [nettoyer(round(-v, 2)) for v in bt["VaR"].values],
        },
    })


# =========================================================
# EXPORT PDF
# =========================================================

STYLE_PDF = """
body { font-family: Helvetica, sans-serif; font-size: 11px; color: #222; }
h1 { font-size: 20px; }
h2 { font-size: 14px; margin-top: 20px; border-bottom: 1px solid #2c5282; padding-bottom: 4px; }
table { width: 100%; border-collapse: collapse; margin-top: 8px; }
th, td { border-bottom: 1px solid #ccc; padding: 5px 8px; text-align: left; font-size: 10px; }
th { background: #eee; }
"""


def fmt_usd(x):
    try:
        return "$" + f"{x:,.0f}".replace(",", " ")
    except (TypeError, ValueError):
        return "—"


def fmt_pct(x, decimales=2):
    try:
        return f"{x:.{decimales}f}%"
    except (TypeError, ValueError):
        return "—"


def tableau_html(entetes, lignes):
    html = "<table>"
    if entetes:
        html += "<tr>" + "".join(f"<th>{e}</th>" for e in entetes) + "</tr>"
    for ligne in lignes:
        html += "<tr>" + "".join(f"<td>{c}</td>" for c in ligne) + "</tr>"
    return html + "</table>"


@app.route("/api/export/pdf")
def api_export_pdf():
    erreur = erreur_si_pas_d_analyse()
    if erreur:
        return erreur

    from xhtml2pdf import pisa   # importé ici pour que l'app démarre même sans cette librairie

    meta = ETAT["meta"]
    mv = ETAT["mv"]
    valeur = float(mv.sum())
    confidence = ETAT["confidence"]
    stats_desc = ETAT["stats_desc"]
    vol_summary = ETAT["vol_summary"]
    backtest = ETAT["backtest"]

    resume = tableau_html([], [
        ["Période couverte", f"{meta['start_date']} → {meta['end_date']}"],
        ["Jours de trading", meta["n_obs"]],
        ["Confiance VaR", fmt_pct(confidence * 100, 1)],
        ["&lambda; EWMA", meta["lambda_ewma"]],
        ["Valeur du portefeuille", fmt_usd(valeur)],
    ])

    repartition = tableau_html(["Actif", "Poids", "Exposition"], [
        [a, fmt_pct(100 * mv[a] / valeur), fmt_usd(mv[a])]
        for a in FACTEURS if a in mv.index
    ])

    var_es = tableau_html(["Méthode", "VaR", "VaR (%)", "Expected Shortfall"], [
        [m, fmt_usd(r["VaR"]), fmt_pct(var_en_pct(r["VaR"], valeur)), fmt_usd(r["ES"])]
        for m, r in ETAT["resultats_var"].items()
    ])

    statistiques = tableau_html(
        ["Actif", "Rendement annuel", "Vol. annualisée", "Sharpe", "Complétude"], [
            [a,
             fmt_pct(stats_desc.loc[a, "Rendement moyen annuel (%)"]),
             fmt_pct(vol_summary.loc[a, "Volatilité annualisée (%)"]),
             round(stats_desc.loc[a, "Sharpe approx."], 2),
             fmt_pct(meta["completude_pct"].get(a), 1)]
            for a in FACTEURS if a in stats_desc.index
        ])

    if backtest:
        table_bt = tableau_html(["Méthode", "Testés", "Violations", "Taux", "Zone Bâle"], [
            [nom, r["resume"]["n_obs"], r["resume"]["n_violations"],
             fmt_pct(r["resume"]["taux_violation"] * 100), r["feu"]["zone"]]
            for nom, r in backtest["resultats"].items()
        ])
        backtest_html = f"{table_bt}<p>Fenêtre glissante : {backtest['window']} jours.</p>"
    else:
        backtest_html = "<p>Aucun backtest exécuté.</p>"

    html = f"""
    <html>
    <head><style>{STYLE_PDF}</style></head>
    <body>
      <h1>Rapport de risque</h1>
      <h2>Résumé de l'analyse</h2>{resume}
      <h2>Répartition du portefeuille</h2>{repartition}
      <h2>VaR &amp; Expected Shortfall</h2>{var_es}
      <h2>Statistiques par actif</h2>{statistiques}
      <h2>Backtesting</h2>{backtest_html}
    </body>
    </html>
    """

    pdf = io.BytesIO()
    pisa.CreatePDF(io.StringIO(html), dest=pdf)
    pdf.seek(0)

    return send_file(
        pdf, mimetype="application/pdf", as_attachment=True,
        download_name=f"rapport_risque_{datetime.date.today().isoformat()}.pdf")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)