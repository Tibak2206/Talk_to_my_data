import joblib
import pandas as pd

from src.config import DECISION_THRESHOLD, ID_COLUMN, MODEL_PATH, SCORING_OUTPUT_PATH, TARGET_COLUMN


def score(df, pipeline=None):
    """Calcule proba_default et label_pred pour chaque client de df.

    df doit contenir les colonnes brutes attendues par la pipeline (cf.
    build_pipeline) ainsi que ID_COLUMN. TARGET_COLUMN n'est pas requise et
    est ignoree si presente.
    """
    if pipeline is None:
        pipeline = joblib.load(MODEL_PATH)

    X = df.drop(columns=[TARGET_COLUMN], errors="ignore")
    proba_default = pipeline.predict_proba(X)[:, 1]
    label_pred = (proba_default >= DECISION_THRESHOLD).astype(int)

    return pd.DataFrame({
        "id": df[ID_COLUMN],
        "proba_default": proba_default,
        "label_pred": label_pred,
    })


if __name__ == "__main__":
    from src.data_prep import load_and_prepare_data

    _, df_test = load_and_prepare_data()
    scoring_df = score(df_test)

    SCORING_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    scoring_df.to_csv(SCORING_OUTPUT_PATH, index=False)
    print(f"Fichier de scoring sauvegarde : {SCORING_OUTPUT_PATH}")
