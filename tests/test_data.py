import pandas as pd
import pytest

from src.ml_olist.training.data import load_olist_data, split_data


@pytest.fixture(scope="module")
def df():
    return load_olist_data()


def test_load_returns_non_empty(df):
    assert isinstance(df, pd.DataFrame) and len(df) > 1000


def test_target_positive(df):
    assert (df["delivery_time_days"] > 0).all()


def test_target_under_60_days(df):
    assert (df["delivery_time_days"] < 60).all()


def test_split_sizes(df):
    X_tr, X_te, _, _ = split_data(df)
    assert abs(len(X_te) / len(df) - 0.2) < 0.02
