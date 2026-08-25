import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from src.config import (
    CATEGORICAL_COLUMNS,
    EDUCATION_LEVEL_MAPPING,
    MARITAL_STATUS_MAPPING,
    NUMERIC_COLUMNS,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
    TYPE_CAST_COLUMNS,
)


def load_and_prepare_data():
    """Charge le CSV brut, type les colonnes categorielles et split train/test."""
    df = pd.read_csv(RAW_DATA_PATH)
    df[TYPE_CAST_COLUMNS] = df[TYPE_CAST_COLUMNS].astype("str")

    df_train, df_test = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COLUMN],
    )
    return df_train, df_test


def clean_categoricals(X):
    """Regroupe les modalites non documentees/rares (regle fixe decidee en EDA).

    Cast egalement sex/marital_status en str : rend cette etape auto-suffisante
    pour que la pipeline reste correcte meme appliquee a des donnees brutes
    (infer.py) qui n'auraient pas transite par load_and_prepare_data().
    """
    X = X.copy()
    X["sex"] = X["sex"].astype(str)
    X["marital_status"] = X["marital_status"].astype(str).replace(MARITAL_STATUS_MAPPING)
    X["education_level"] = X["education_level"].replace(EDUCATION_LEVEL_MAPPING)
    return X


def build_pipeline(model):
    """Construit la pipeline complete : nettoyage categoriel, encodage, modele."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
            ("numeric", "passthrough", NUMERIC_COLUMNS),
        ]
    )

    return Pipeline(
        steps=[
            ("clean_categoricals", FunctionTransformer(clean_categoricals)),
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
