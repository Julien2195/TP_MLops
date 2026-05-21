import numpy as np
import pytest

from src.ml_olist.common.features import add_olist_features
from src.ml_olist.training.data import load_olist_data, split_data
from src.ml_olist.training.features import build_preprocessing_pipeline


@pytest.fixture(scope="module")
def splits():
    df = load_olist_data()
    return split_data(df)


def test_derived_features_present(splits):
    X_tr, _, _, _ = splits
    out = add_olist_features(X_tr)
    for col in ["freight_ratio", "log_price", "is_multi_item",
                "purchase_dow", "purchase_month"]:
        assert col in out.columns, f"Feature manquante : {col}"


def test_no_nan_after_preprocessing(splits):
    X_tr, X_te, _, _ = splits
    pipe = build_preprocessing_pipeline()
    assert not np.isnan(pipe.fit_transform(X_tr)).any()
    assert not np.isnan(pipe.transform(X_te)).any()


def test_shape_preserved(splits):
    X_tr, _, _, _ = splits
    pipe = build_preprocessing_pipeline()
    out = pipe.fit_transform(X_tr)
    assert out.shape[0] == X_tr.shape[0]


def test_log_price_positive(splits):
    X_tr, _, _, _ = splits
    out = add_olist_features(X_tr)
    assert (out["log_price"] >= 0).all()
