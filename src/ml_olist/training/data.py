import pandas as pd
from pathlib import Path

FEATURE_COLS = [
    "product_weight_g",
    "product_volume_cm3",
    "price",
    "freight_value",
    "distance_km"
]
TARGET_COL = "delivery_time_days"

def load_olist_data():
    """Charge le jeu de données Olist depuis les fichiers CSV locaux."""
    data_path = Path("data/processed/olist_cleaned.csv")

    if not data_path.exists():
        print(f"⚠️ Fichier {data_path} introuvable, création d'un dataset de simulation enrichi...")
        data_path.parent.mkdir(parents=True, exist_ok=True)

        # Génération d'un dataset fictif compatible avec le feature engineering métier
        df_mock = pd.DataFrame({col: [100.0] * 10 for col in FEATURE_COLS})
        df_mock[TARGET_COL] = [10.5] * 10

        # Ajout des colonnes requises par add_olist_features
        df_mock["order_item_id"] = [1] * 10
        df_mock["order_purchase_timestamp"] = pd.date_range(start="2026-01-01", periods=10, freq="D")

        df_mock.to_csv(data_path, index=False)

    df = pd.read_csv(data_path)
    return df

def validate_schema(df):
    """Valide les colonnes de base nécessaires."""
    # On vérifie les features de base et les colonnes intermédiaires du feature engineering
    required_cols = FEATURE_COLS + [TARGET_COL, "order_item_id", "order_purchase_timestamp"]
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans le dataset : {missing}")

def split_data(df):
    """Sépare les données en train et test sets."""
    # Attention : Ici on passe tout le dataframe à l'exclusion de la cible,
    # car le Pipeline Scikit-Learn a besoin de voir toutes les colonnes brutes
    # pour faire son add_olist_features !
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=0.2, random_state=42)
