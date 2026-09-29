Application web qui mesure le risque d'un portefeuille(AAPL,MSFT,LVMH,S&P 500,OR,bond 5Y et 10Y, CALL AAPL ET PUT AAPL) à partir de données de marché réelles.
Elle calcule la VaR et l'Expected Shortfall par plusieurs méthodes, teste leur fiabilité par backtesting et exporte un rapport PDF.
<img width="927" height="433" alt="image" src="https://github.com/user-attachments/assets/256ebf42-6549-4c57-b822-9f0e96eaa273" />
télécharge les historiques de prix depuis la date choisie, calcule les rendements, les volatilités et les matrices de covariance.
- Deux estimations de la covariance: moyenne simple et EWMA (RiskMetrics, λ réglable, 0,94 par défaut).
- VaR et Expected Shortfall par cinq méthodes :
  - paramétrique (covariance simple et EWMA),
  - historique,
  - Monte Carlo (covariance simple et EWMA).
- Backtesting sur fenêtre glissante, avec trois tests de validation :
  - test de Kupiec (nombre de violations),
  - test de Christoffersen (indépendance des violations),
  - zone verte, jaune ou rouge de Bâle.
- Export PDF d'un rapport complet.
