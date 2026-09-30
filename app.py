import io
import math
import os
import traceback
import datetime

from flask import Flask, request, jsonify, render_template, send_file,redirect, url_for

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
from statistical_tests import traffic_light_zone
app = Flask(__name__)
app.secret_key = "risk-console-dev-key"  

ETAT = {}


def etat_dispo():
    return bool(ETAT)


def require_etat_json():
    """Retourne un tuple (réponse d'erreur, code) si aucune analyse n'est chargée."""
    if not etat_dispo():
        return jsonify({
            "ok": False,
            "error": "Aucune analyse chargée : lancez d'abord une construction "
                     "des données depuis l'onglet Configuration.",
        }), 409
    return None


def clean(v):
    """Rend une valeur JSON-safe (NaN/inf -> None, numpy -> python natif)."""
    if v is None:
        return None
    if isinstance(v, (np.floating, np.integer)):
        v = v.item()
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    return v


def fmt_usd(x):
    try:
        return f"{x:,.0f}".replace(",", " ")
    except Exception:
        return "—"


def fmt_pct(x, dec=2):
    try:
        return f"{x:.{dec}f}"
    except Exception:
        return "—"



def lancer_analyse(start_date: str, confidence: float, lambda_ewma: float,
                    n_scenarios_mc: int):
    journal = []
    meta = construire_donnees(
        start_date, lambda_ewma=lambda_ewma, log_callback=journal.append
    )

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
    rendements_annuels_pct = stats_desc["Rendement moyen annuel (%)"]

    metrics_simple = portfolio_metrics(mv, matrice_cov, rendements_annuels_pct)
    metrics_ewma = portfolio_metrics(mv, matrice_cov_ewma, rendements_annuels_pct)

    vol_summary = pd.read_excel(meta["output_file"], sheet_name="Volatilites", index_col=0)

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
        "metrics_simple": metrics_simple,
        "metrics_ewma": metrics_ewma,
        "journal": journal,
        "backtest": None,  
    })


def var_pct_du_portefeuille(var_usd, valeur_portefeuille):
    if not valeur_portefeuille:
        return 0.0
    return 100 * var_usd / valeur_portefeuille

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    if not etat_dispo():
        return jsonify({"ok": True, "chargee": False})

    meta = ETAT["meta"]
    mv = ETAT["mv"]
    return jsonify({
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
        "valeur_portefeuille": float(mv.sum()),
    })


@app.route("/api/analyse", methods=["POST"])
def api_analyse():
    data = request.get_json(force=True, silent=True) or {}
    start_date = str(data.get("start_date", "")).strip()
    try:
        confidence = float(data.get("confidence", 0.99))
        lambda_ewma = float(data.get("lambda_ewma", 0.94))
        n_scenarios_mc = int(data.get("n_scenarios_mc", 50000))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Paramètres numériques invalides."}), 400

    if not start_date:
        return jsonify({
            "ok": False,
            "error": "La date de début est obligatoire.",
        }), 400

    try:
        lancer_analyse(start_date, confidence, lambda_ewma, n_scenarios_mc)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": f"Échec de la construction des données : {e}"}), 400

    meta = ETAT["meta"]
    mv = ETAT["mv"]
    return jsonify({
        "ok": True,
        "meta": {
            "start_date": meta["start_date"],
            "end_date": meta["end_date"],
            "n_obs": meta["n_obs"],
            "lambda_ewma": meta["lambda_ewma"],
        },
        "confidence": confidence,
        "n_scenarios_mc": n_scenarios_mc,
        "valeur_portefeuille": float(mv.sum()),
    })


@app.route("/api/dashboard")
def api_dashboard():
    err = require_etat_json()
    if err:
        return err

    mv = ETAT["mv"]
    valeur_portefeuille = float(mv.sum())
    resultats_var = ETAT["resultats_var"]
    vol_summary = ETAT["vol_summary"]
    metrics_ewma = ETAT["metrics_ewma"]

    poids = mv / valeur_portefeuille

    actifs = []
    for actif in FACTEURS:
        if actif not in mv.index:
            continue
        vol_annuelle = (
            vol_summary.loc[actif, "Volatilité annualisée (%)"]
            if actif in vol_summary.index else None
        )
        actifs.append({
            "nom": actif,
            "vol_pct": clean(vol_annuelle),
            "poids_pct": clean(100 * poids.get(actif, 0.0)),
            "exposition": clean(mv.get(actif, 0.0)),
        })
    actifs.sort(key=lambda c: c["vol_pct"] if c["vol_pct"] is not None else -1, reverse=True)

    rendements = ETAT["rendements"]
    poids_hist = mv.reindex(FACTEURS) / valeur_portefeuille
    rp = rendements.reindex(columns=FACTEURS).dot(poids_hist)
    cumul = (1 + rp).cumprod() * valeur_portefeuille
    pas = max(1, len(cumul) // 180)
    cumul_ech = cumul.iloc[::pas]

    return jsonify({
        "ok": True,
        "valeur_portefeuille": valeur_portefeuille,
        "var_historique": clean(resultats_var["Historique"]["VaR"]),
        "var_parametrique_ewma": clean(resultats_var["Paramétrique (EWMA)"]["VaR"]),
        "vol_annuelle_pct": clean(metrics_ewma["volatilite_annuelle_pct"]),
        "rendement_annuel_pct": clean(metrics_ewma["rendement_annuel_pct"]),
        "actifs": actifs,
        "evolution": {
            "dates": [d.strftime("%Y-%m-%d") for d in cumul_ech.index],
            "valeurs": [clean(round(v, 2)) for v in cumul_ech.values],
        },
        "repartition": {
            "labels": [a for a in FACTEURS if a in poids.index],
            "poids_pct": [clean(round(100 * poids[a], 2)) for a in FACTEURS if a in poids.index],
        },
    })


@app.route("/api/donnees")
def api_donnees():
    err = require_etat_json()
    if err:
        return err

    stats_desc = ETAT["stats_desc"]
    vol_summary = ETAT["vol_summary"]
    meta = ETAT["meta"]
    rendements = ETAT["rendements"]

    lignes = []
    for actif in FACTEURS:
        if actif not in stats_desc.index:
            continue
        lignes.append({
            "actif": actif,
            "completude": clean(meta["completude_pct"].get(actif)),
            "rendement_moyen_annuel": clean(stats_desc.loc[actif, "Rendement moyen annuel (%)"]),
            "vol_journaliere": clean(vol_summary.loc[actif, "Volatilité journalière (%)"]),
            "vol_annualisee": clean(vol_summary.loc[actif, "Volatilité annualisée (%)"]),
            "min": clean(stats_desc.loc[actif, "min"] * 100),
            "max": clean(stats_desc.loc[actif, "max"] * 100),
        })

    cumul = (1 + rendements.reindex(columns=FACTEURS)).cumprod()
    pas = max(1, len(cumul) // 150)
    cumul_ech = cumul.iloc[::pas]
    dates = [d.strftime("%Y-%m-%d") for d in cumul_ech.index]
    series = {
        col: [clean(round(v, 4)) for v in cumul_ech[col].values]
        for col in cumul_ech.columns
    }

    return jsonify({
        "ok": True,
        "actifs": [a for a in FACTEURS if a in stats_desc.index],
        "lignes": lignes,
        "cumul": {"dates": dates, "series": series},
        "meta": {
            "start_date": meta["start_date"],
            "end_date": meta["end_date"],
            "n_obs": meta["n_obs"],
        },
    })


@app.route("/api/covariance")
def api_covariance():
    err = require_etat_json()
    if err:
        return err

    methode = request.args.get("methode", "simple")
    cov = ETAT["matrice_cov"] if methode == "simple" else ETAT["matrice_cov_ewma"]
    cov = cov.reindex(index=FACTEURS, columns=FACTEURS)
    corr = ETAT["rendements"].reindex(columns=FACTEURS).corr()
    metrics = ETAT["metrics_simple"] if methode == "simple" else ETAT["metrics_ewma"]

    def to_matrix(df, dec):
        return [[clean(round(df.loc[i, j], dec)) for j in df.columns] for i in df.index]

    return jsonify({
        "ok": True,
        "methode": methode,
        "actifs": FACTEURS,
        "cov_matrix": to_matrix(cov, 6),
        "corr_matrix": to_matrix(corr, 3),
        "lambda_ewma": ETAT["meta"]["lambda_ewma"],
        "metrics": {
            "rendement_annuel_pct": clean(metrics["rendement_annuel_pct"]),
            "volatilite_annuelle_pct": clean(metrics["volatilite_annuelle_pct"]),
        },
    })


@app.route("/api/var")
def api_var():
    err = require_etat_json()
    if err:
        return err

    resultats_var = ETAT["resultats_var"]
    valeur_portefeuille = float(ETAT["mv"].sum())
    confidence = ETAT["confidence"]

    lignes = []
    for methode, r in resultats_var.items():
        lignes.append({
            "methode": methode,
            "VaR": clean(r["VaR"]),
            "ES": clean(r["ES"]),
            "VaR_pct": clean(var_pct_du_portefeuille(r["VaR"], valeur_portefeuille)),
            "ES_pct": clean(var_pct_du_portefeuille(r["ES"], valeur_portefeuille)),
        })

    return jsonify({
        "ok": True,
        "confidence": confidence,
        "valeur_portefeuille": valeur_portefeuille,
        "n_scenarios_mc": ETAT["n_scenarios_mc"],
        "lignes": lignes,
    })


@app.route("/api/backtest", methods=["POST"])
def api_backtest():
    err = require_etat_json()
    if err:
        return err

    data = request.get_json(force=True, silent=True) or {}
    try:
        window = int(data.get("window", 250))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Fenêtre invalide."}), 400
    inclure_mc = bool(data.get("inclure_mc", False))

    rendements = ETAT["rendements"]
    positions = ETAT["positions"]
    mv = ETAT["mv"]
    confidence = ETAT["confidence"]

    if len(rendements) - window < 30:
        return jsonify({
            "ok": False,
            "error": (
                f"Réduisez la fenêtre ou allongez l'historique."
            ),
        }), 400

    
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

    resultats_backtest = {}
    try:
        for label, method, cov_method in configs:
            bt = rolling_backtest(
                rendements, positions, mv, method=method, window=window,
                confidence=confidence, n_scenarios=10_000, cov_method=cov_method,
            )
            resume = summarize_backtest(bt)
            feu = traffic_light_zone(resume["n_violations"], resume["n_obs"], confidence)
            resultats_backtest[label] = {"bt": bt, "resume": resume, "feu": feu}
        ETAT["backtest"] = {"window": window, "resultats": resultats_backtest}
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": f"Échec du backtest : {e}"}), 400

    lignes = []
    for label, r in resultats_backtest.items():
        resume = r["resume"]
        lignes.append({
            "label": label,
            "n_obs": resume["n_obs"],
            "violations": resume["n_violations"],
            "taux_observe_pct": clean(resume["taux_violation"] * 100),
            "taux_attendu_pct": clean((1 - confidence) * 100),
            "zone": r["feu"]["zone"],
        })

    ref_label = "Historique" if "Historique" in resultats_backtest else next(iter(resultats_backtest))
    bt_ref = resultats_backtest[ref_label]["bt"]
    pas = max(1, len(bt_ref) // 200)
    bt_ech = bt_ref.iloc[::pas]

    return jsonify({
        "ok": True,
        "window": window,
        "n_obs_total": len(rendements),
        "lignes": lignes,
        "chart": {
            "methode": ref_label,
            "dates": [d.strftime("%Y-%m-%d") for d in bt_ech.index],
            "pnl": [clean(round(v, 2)) for v in bt_ech["pnl"].values],
            "var_neg": [clean(round(-v, 2)) for v in bt_ech["VaR"].values],
        },
    })


@app.route("/api/export/pdf")
def api_export_pdf():
    err = require_etat_json()
    if err:
        return err

    from xhtml2pdf import pisa

    meta = ETAT["meta"]
    mv = ETAT["mv"]
    valeur_portefeuille = float(mv.sum())
    resultats_var = ETAT["resultats_var"]
    confidence = ETAT["confidence"]
    stats_desc = ETAT["stats_desc"]
    vol_summary = ETAT["vol_summary"]
    backtest = ETAT.get("backtest")
    poids = (mv / valeur_portefeuille)

    lignes_var_html = ""
    for m, r in resultats_var.items():
        lignes_var_html += (
            f"<tr><td>{m}</td>"
            f"<td>${fmt_usd(r['VaR'])}</td>"
            f"<td>{fmt_pct(var_pct_du_portefeuille(r['VaR'], valeur_portefeuille))}%</td>"
            f"<td>${fmt_usd(r['ES'])}</td></tr>"
        )

    lignes_donnees_html = ""
    for actif in FACTEURS:
        if actif not in stats_desc.index:
            continue
        lignes_donnees_html += (
            f"<tr><td>{actif}</td>"
            f"<td>{fmt_pct(stats_desc.loc[actif, 'Rendement moyen annuel (%)'])}%</td>"
            f"<td>{fmt_pct(vol_summary.loc[actif, 'Volatilité annualisée (%)'])}%</td>"
            f"<td>{fmt_pct(stats_desc.loc[actif, 'Sharpe approx.'])}</td>"
            f"<td>{fmt_pct(meta['completude_pct'].get(actif), 1)}%</td></tr>"
        )

    poids_html = "".join(
        f"<tr><td>{actif}</td><td>{fmt_pct(100 * poids[actif])}%</td>"
        f"<td>${fmt_usd(mv[actif])}</td></tr>"
        for actif in FACTEURS if actif in mv.index
    )

    backtest_html = "<p>Aucun backtest exécuté.</p>"
    if backtest:
        lignes_bt = ""
        for label, r in backtest["resultats"].items():
            lignes_bt += (
                f"<tr><td>{label}</td>"
                f"<td>{r['resume']['n_obs']}</td>"
                f"<td>{r['resume']['n_violations']}</td>"
                f"<td>{fmt_pct(r['resume']['taux_violation'] * 100)}%</td>"
                f"<td>{r['feu']['zone']}</td></tr>"
            )
        backtest_html = f"""
        <table>
          <tr><th>Méthode</th><th>Testés</th><th>Violations</th><th>Taux</th>
              <th>Zone Bâle</th></tr>
          {lignes_bt}
        </table>
        <p>Fenêtre glissante : {backtest['window']} jours.</p>
        """

    date_generation = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")

    html = f"""
    <html>
    <head>
    <style>
      body {{ font-family: Helvetica, sans-serif; font-size: 11px; color: #1A1814; }}
      h1 {{ font-size: 20px; margin-bottom: 4px; }}
      h2 {{ font-size: 14px; margin-top: 22px; border-bottom: 1px solid #C9A84C; padding-bottom: 4px; }}
      table {{ width: 100%; border-collapse: collapse; margin-top: 8px; }}
      th, td {{ border-bottom: 1px solid #ccc; padding: 5px 8px; text-align: left; font-size: 10px; }}
      th {{ background: #EDE9DC; text-transform: uppercase; font-size: 8px; letter-spacing: 0.08em; }}
      .meta {{ color: #7A7260; font-size: 10px; margin-bottom: 12px; }}
    </style>
    </head>
    <body>
      <h1>Rapport de risque</h1>

      <h2>Résumé de l'analyse</h2>
      <table>
        <tr><td>Période couverte</td><td>{meta['start_date']} → {meta['end_date']}</td></tr>
        <tr><td>Jours de trading</td><td>{meta['n_obs']}</td></tr>
        <tr><td>Confiance VaR</td><td>{fmt_pct(confidence * 100, 1)}%</td></tr>
        <tr><td>&lambda; EWMA</td><td>{meta['lambda_ewma']}</td></tr>
        <tr><td>Valeur du portefeuille</td><td>${fmt_usd(valeur_portefeuille)}</td></tr>
      </table>

      <h2>Répartition du portefeuille</h2>
      <table>
        <tr><th>Actif</th><th>Poids</th><th>Exposition</th></tr>
        {poids_html}
      </table>

      <h2>VaR &amp; Expected Shortfall — comparaison des méthodes</h2>
      <table>
        <tr><th>Méthode</th><th>VaR</th><th>VaR (%)</th><th>Expected Shortfall</th></tr>
        {lignes_var_html}
      </table>

      <h2>Statistiques descriptives par actif</h2>
      <table>
        <tr><th>Actif</th><th>Rendement annuel</th><th>Vol. annualisée</th><th>Sharpe</th><th>Complétude</th></tr>
        {lignes_donnees_html}
      </table>

      <h2>Backtesting</h2>
      {backtest_html}
    </body>
    </html>
    """

    pdf_buffer = io.BytesIO()
    pisa.CreatePDF(io.StringIO(html), dest=pdf_buffer)
    pdf_buffer.seek(0)

    return send_file(
        pdf_buffer, mimetype="application/pdf", as_attachment=True,
        download_name=f"rapport_risque_{datetime.date.today().isoformat()}.pdf",
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)