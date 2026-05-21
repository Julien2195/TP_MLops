from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

# Import de ta fonction d'origine corrigée
from src.ml_olist.common.features import add_olist_features


def build_preprocessing_pipeline() -> Pipeline:
    """Crée le pipeline global combinant le feature engineering Olist et le scaling."""

    # 1. Étape de Feature Engineering Personnalisé (Ta fonction)
    feature_engineering = FunctionTransformer(add_olist_features)

    # Liste de TOUTES les variables finales qui vont entrer dans le RandomForest
    all_features = [
        "product_weight_g",
        "product_volume_cm3",
        "price",
        "freight_value",
        "distance_km",
        "freight_ratio",
        "log_price",
        "is_multi_item",
        "purchase_dow",
        "purchase_month"
    ]

    # 2. Imputation et Standardisation des variables numériques
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    processor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, all_features)
        ],
        remainder="drop"
    )

    # Pipeline final : d'abord on crée les features, ensuite on applique l'imputer/scaler
    pipeline = Pipeline(steps=[
        ("feature_eng", feature_engineering),
        ("processor", processor)
    ])

    return pipeline
