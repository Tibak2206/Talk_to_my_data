import joblib
from sklearn.ensemble import GradientBoostingClassifier

from src.config import FINAL_MODEL_PARAMS, MODEL_PATH, TARGET_COLUMN
from src.data_prep import build_pipeline, load_and_prepare_data


def train_and_save_model():
    """Entraine le modele final (Gradient Boosting, hyperparametres optimises
    dans le notebook 03) sur le train, et sauvegarde la pipeline complete.
    """
    df_train, df_test = load_and_prepare_data()
    X_train = df_train.drop(columns=[TARGET_COLUMN])
    y_train = df_train[TARGET_COLUMN]

    model = GradientBoostingClassifier(**FINAL_MODEL_PARAMS)
    pipeline = build_pipeline(model)
    pipeline.fit(X_train, y_train)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Modele entraine et sauvegarde : {MODEL_PATH}")

    return pipeline


if __name__ == "__main__":
    train_and_save_model()
