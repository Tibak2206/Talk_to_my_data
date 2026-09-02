import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import (
    BILL_AMT_COLUMNS,
    CATEGORICAL_COLUMNS,
    EDUCATION_LEVEL_MAPPING,
    EDUCATION_LEVEL_ORDER,
    MARITAL_STATUS_MAPPING,
    NUMERIC_COLUMNS,
    PAY_AMT_COLUMNS,
    PAY_STATUS_COLUMNS,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
    TYPE_CAST_COLUMNS,
)


def load_and_prepare_data():
    """Charge le CSV brut, type les colonnes categorielles et split train/test."""
    df = pd.read_csv(RAW_DATA_PATH)
    # pay_0 -> pay_1 : uniformise la numerotation avec bill_amt_*/pay_amt_* (suffixe 1
    # = mois le plus recent pour les 3 familles, cf. commentaire dans config.py).
    df = df.rename(columns={"pay_0": "pay_1"})
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


def engineer_features(X):
    """Ajoute les features derivees decidees pour la modelisation avancee.

    - reste_du_moyen : ecart moyen bill_amt - pay_amt sur les 6 mois (montant
      non couvert par les paiements). bill_amt_i et pay_amt_i ne sont pas
      parfaitement alignes dans le temps (pay_amt_i rembourse plutot
      bill_amt_(i-1)), mais bill_amt_i et bill_amt_(i-1) sont tres correles
      (0.80-0.95 en EDA), donc l'approximation reste raisonnable.
    - taux_utilisation_moyen : bill_amt_i / limit_balance, moyenne sur 6 mois.
      limit_balance >= 10000 sur ce dataset (cf. describe() en EDA), pas de
      risque de division par zero.
    - bill_amt_moyen / bill_amt_tendance : remplacent les 6 colonnes brutes
      bill_amt_1..6, tres colineaires entre elles (0.80-0.95), par un niveau
      d'exposition moyen et une tendance. bill_amt_1 = mois le plus recent,
      bill_amt_6 = mois le plus ancien (cf. config.py) : tendance = bill_amt_1
      - bill_amt_6, positive si le solde a augmente du mois le plus ancien
      vers le plus recent (endettement croissant).
    - retard_max / nb_mois_en_retard : agregats de severite/persistance du
      retard, en complement (pas en remplacement) des pay_1..pay_6 bruts,
      individuellement les plus discriminants et moins redondants entre eux
      (correlation max ~0.82 contre 0.77-0.95 pour bill_amt_*).
    """
    X = X.copy()

    reste_du = X[BILL_AMT_COLUMNS].to_numpy() - X[PAY_AMT_COLUMNS].to_numpy()
    X["reste_du_moyen"] = reste_du.mean(axis=1)

    X["taux_utilisation_moyen"] = X[BILL_AMT_COLUMNS].div(X["limit_balance"], axis=0).mean(axis=1)

    X["bill_amt_moyen"] = X[BILL_AMT_COLUMNS].mean(axis=1)
    X["bill_amt_tendance"] = X["bill_amt_1"] - X["bill_amt_6"]
    X = X.drop(columns=BILL_AMT_COLUMNS)

    X["retard_max"] = X[PAY_STATUS_COLUMNS].max(axis=1)
    X["nb_mois_en_retard"] = (X[PAY_STATUS_COLUMNS] > 0).sum(axis=1)

    return X


def encode_education_level(X):
    """Encode education_level en ordinal + flag plutot qu'en one-hot.

    Les modalites 1/2/3 (etudes sup/universite/lycee) suivent un ordre de
    risque croissant confirme en EDA (8%/18%/24%/24% de taux de defaut) : un
    encodage ordinal leur donne une seule variable exploitable directement par
    les modeles lineaires, sans perte pour les arbres.

    La modalite 0 (non documentee) n'a pas de position logique sur cette
    echelle : plutot que de la forcer arbitrairement dans l'ordre, elle est
    isolee dans un flag binaire is_education_undocumented, et la colonne
    ordinale recoit une valeur neutre (mediane des categories documentees)
    pour ces lignes.
    """
    X = X.copy()
    X["is_education_undocumented"] = (X["education_level"] == 0).astype(int)
    X["education_level_ordinal"] = X["education_level"].map(EDUCATION_LEVEL_ORDER).fillna(1)
    X = X.drop(columns=["education_level"])
    return X


def build_pipeline(model):
    """Construit la pipeline complete : nettoyage, feature engineering, encodage, modele."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
            ("numeric", StandardScaler(), NUMERIC_COLUMNS),
        ]
    )

    return Pipeline(
        steps=[
            ("clean_categoricals", FunctionTransformer(clean_categoricals)),
            ("engineer_features", FunctionTransformer(engineer_features)),
            ("encode_education_level", FunctionTransformer(encode_education_level)),
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
