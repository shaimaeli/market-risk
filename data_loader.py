import pandas as pd

RISK_FILE = "Matrices_Risque_Portefeuille.xlsx"


def _clean_matrix(df: pd.DataFrame) -> pd.DataFrame:
    df.index = df.index.astype(str).str.strip()
    df.columns = df.columns.astype(str).str.strip()
    return df


def load_risk_data(filepath: str = RISK_FILE):
    matrice_covariance = _clean_matrix(
        pd.read_excel(filepath, sheet_name="Covariance_annualisee", index_col=0)
    )
    matrice_covariance_ewma = _clean_matrix(
        pd.read_excel(filepath, sheet_name="Covariance_EWMA_annualisee", index_col=0)
    )

    rendements = pd.read_excel(
        filepath, sheet_name="Rendements_fusionnes", index_col=0
    )
    rendements.columns = rendements.columns.astype(str).str.strip()

    return matrice_covariance, matrice_covariance_ewma, rendements