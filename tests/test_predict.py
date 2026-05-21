from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

MODEL_PATH = Path("artifacts/model_latest.joblib")


# Commande type alignée sur FEATURE_COLS de data.py
SAMPLE = pd.DataFrame([{
    "product_weight_g": 500.0,
    "product_volume_cm3": 1000.0,
    "price": 89.90,
    "freight_value": 12.50,
    "distance_km": 250.0,
    "order_item_id": 1,
    "order_purchase_timestamp": datetime(2024, 6, 10, 9, 0),
}])


@pytest.fixture(scope="module")
def model():
    if not MODEL_PATH.exists():
        pytest.skip("Modèle absent — lancer train.py d'abord")
    import joblib
    return joblib.load(MODEL_PATH)


def test_prediction_is_positive(model):
    pred = model.predict(SAMPLE)[0]
    assert pred > 0, f"Temps prédit négatif : {pred}"


def test_prediction_under_60_days(model):
    pred = model.predict(SAMPLE)[0]
    assert pred < 60, f"Temps prédit irréaliste : {pred:.1f} jours"


def test_model_r2_threshold(model):
    """Gate CI : R2 doit dépasser 0.70 sur le holdout set."""
    from sklearn.metrics import r2_score

    from src.ml_olist.training.data import load_olist_data, split_data
    df = load_olist_data()
    _, X_te, _, y_te = split_data(df)
    r2 = r2_score(y_te, model.predict(X_te))
    assert r2 >= 0.70, f"R² trop faible : {r2:.4f}"
