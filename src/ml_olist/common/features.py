import numpy as np
import pandas as pd


def add_olist_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering métier Olist.

    Appelé par FunctionTransformer dans le pipeline sklearn.
    DOIT être importable depuis common/ en training ET en production.

    Features créées :
    freight_ratio : part du fret dans le coût total de la commande
    log_price : log(1+price) — réduit l'asymétrie de distribution
    is_multi_item : commande multi-articles (0/1)
    purchase_dow : jour de la semaine de la commande (0=lundi, 6=dimanche)
    purchase_month : mois de la commande (1-12)
    """
    df = df.copy()

    # Sécurité pour les dates si elles arrivent sous forme de chaînes de caractères
    if not pd.api.types.is_datetime64_any_dtype(df["order_purchase_timestamp"]):
        df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])

    total = df["price"] + df["freight_value"] + 1e-8
    df["freight_ratio"] = df["freight_value"] / total
    df["log_price"] = np.log1p(df["price"])
    df["is_multi_item"] = (df["order_item_id"] > 1).astype(int)
    df["purchase_dow"] = df["order_purchase_timestamp"].dt.dayofweek
    df["purchase_month"] = df["order_purchase_timestamp"].dt.month
    return df
