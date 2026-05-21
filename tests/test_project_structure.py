from pathlib import Path


def test_artifacts_in_gitignore():
    gi = Path(".gitignore").read_text()
    assert "artifacts/" in gi or "*.joblib" in gi


def test_data_in_gitignore():
    """Les 9 CSV Olist (120 Mo) ne doivent pas aller dans Git."""
    gi = Path(".gitignore").read_text()
    assert "data/" in gi or "*.csv" in gi


def test_no_training_code_in_prediction():
    for f in Path("src/ml_olist/prediction").rglob("*.py"):
        c = f.read_text()
        assert "train_test_split" not in c, f"{f} contient du code training"


def test_env_file_not_committed():
    gi = Path(".gitignore").read_text()
    assert ".env" in gi
