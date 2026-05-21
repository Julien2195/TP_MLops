from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
 
DATA_DIR = Path("data/olist")
def load_olist_data(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """
    Charge et joint les CSV Olist nécessaires.
    Target : delivery_time_days = jours entre achat et livraison.
 
    CSVs utilisés :
      olist_orders_dataset.csv         → dates, statut commande
      olist_order_items_dataset.csv    → prix, frais de port, nb articles
      olist_order_payments_dataset.csv → nb versements, montant payé
      olist_sellers_dataset.csv        → état du vendeur
      olist_customers_dataset.csv      → état du client
    """
    orders = pd.read_csv(
        data_dir / "olist_orders_dataset.csv",
        parse_dates=[
            "order_purchase_timestamp",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ]
    )
    items     = pd.read_csv(data_dir / "olist_order_items_dataset.csv")
    payments  = pd.read_csv(data_dir / "olist_order_payments_dataset.csv")
    sellers   = pd.read_csv(data_dir / "olist_sellers_dataset.csv")
    customers = pd.read_csv(data_dir / "olist_customers_dataset.csv")
 
    # Agréger items par commande
    items_agg = items.groupby("order_id").agg(
        price          = ("price",         "sum"),
        freight_value  = ("freight_value",  "sum"),
        order_item_id  = ("order_item_id",  "max"),
        seller_id      = ("seller_id",      "first"),
    ).reset_index()
 
    # Agréger paiements par commande
    pay_agg = payments.groupby("order_id").agg(
        payment_installments = ("payment_installments", "max"),
        payment_value        = ("payment_value",        "sum"),
    ).reset_index()
 
    # Jointure principale
    df = (
        orders
        .merge(items_agg, on="order_id",  how="inner")
        .merge(pay_agg,   on="order_id",  how="left")
        .merge(sellers[["seller_id","seller_state"]], on="seller_id",   how="left")
        .merge(customers[["customer_id","customer_state"]], on="customer_id", how="left")
    )
 
    # Garder uniquement les commandes effectivement livrées
    df = df[df["order_status"] == "delivered"].copy()
    df = df.dropna(subset=[
        "order_purchase_timestamp",
        "order_delivered_customer_date",
    ])
 
    # Calculer la variable cible
    df["delivery_time_days"] = (
        (df["order_delivered_customer_date"] - df["order_purchase_timestamp"])
        .dt.total_seconds() / 86400
    )
 
    # Supprimer les anomalies : livraisons < 0 jour ou > 60 jours
    df = df[(df["delivery_time_days"] > 0) & (df["delivery_time_days"] < 60)]
    return df
 
 
FEATURE_COLS = [
    "price", "freight_value", "payment_installments",
    "payment_value", "order_item_id",
    "order_purchase_timestamp",
    "seller_state", "customer_state",
]
 
TARGET_COL = "delivery_time_days"
 
 
def validate_schema(df: pd.DataFrame) -> None:
    missing = set(FEATURE_COLS + [TARGET_COL]) - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")
    if (df[TARGET_COL] <= 0).any():
        raise ValueError("Des valeurs négatives ou nulles dans delivery_time_days")
 
 
def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
