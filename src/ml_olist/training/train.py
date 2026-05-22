import os
from pathlib import Path

import joblib
import mlflow
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src.ml_olist.training.data import load_olist_data, split_data, validate_schema
from src.ml_olist.training.features import build_preprocessing_pipeline

ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "model_latest.joblib"
MIN_R2 = 0.70


def train(n_estimators: int = 150, max_depth: int = 15, random_state: int = 42) -> dict:
    """Entraîne un RandomForestRegressor pour prédire delivery_time_days."""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "olist-delivery-time"))

    with mlflow.start_run() as run:
        print(f"MLflow run_id : {run.info.run_id}")

        # Chargement et validation
        df = load_olist_data()
        validate_schema(df)
        X_train, X_test, y_train, y_test = split_data(df)

        print(f"Train : {len(X_train)} lignes | Test : {len(X_test)} lignes")
        print(f"Target : {y_train.mean():.1f} jours en moyenne")

        # Logging des paramètres
        mlflow.log_params(
            {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "random_state": random_state,
                "train_rows": len(X_train),
                "mean_delivery": round(float(y_train.mean()), 2),
            }
        )

        # Pipeline d'entraînement
        model = Pipeline(
            [
                ("preprocessing", build_preprocessing_pipeline()),
                (
                    "regressor",
                    RandomForestRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        model.fit(X_train, y_train)

        # Évaluation
        y_pred = model.predict(X_test)
        metrics = {
            "r2": float(r2_score(y_test, y_pred)),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        }
        mlflow.log_metrics(metrics)
        print(f"R²={metrics['r2']:.4f} | MAE={metrics['mae']:.1f}j | RMSE={metrics['rmse']:.1f}j")

        # Gate de qualité / Seuil de performance
        if metrics["r2"] < MIN_R2:
            raise ValueError(
                f"R² insuffisant : {metrics['r2']:.4f} < seuil {MIN_R2}. Run marqué FAILED."
            )

        # Sauvegarde locale (Garantit que ton modèle est bien présent pour la suite du TP !)
        ARTIFACTS_DIR.mkdir(exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        print(f"✅ Modèle sauvegardé avec succès localement : {MODEL_PATH}")

        # Essai de téléversement vers MLflow, encapsulé pour éviter le crash S3
        try:
            mlflow.log_artifact(local_path=str(MODEL_PATH), artifact_path="model")
            print("🚀 Modèle téléversé avec succès sur le serveur MLflow.")
        except Exception:
            print(
                "⚠️ Alerte stockage : Le modèle n'a pas pu être poussé sur le serveur MLflow "
                "(Manque d'identifiants S3/MinIO)."
            )
            print(
                "   Pas de panique, le run MLflow contient bien tes paramètres et métriques, "
                "et le fichier est dispo localement !"
            )

        return metrics


if __name__ == "__main__":
    train()
