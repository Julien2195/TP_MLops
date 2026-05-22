import os
from datetime import datetime

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


st.set_page_config(page_title="Olist — Prédiction livraison", page_icon="📦")
st.title("📦 Prédiction du temps de livraison")
st.caption("Olist Brazilian E-Commerce — RandomForestRegressor")


st.sidebar.header("Paramètres de la commande")

product_weight_g = st.sidebar.number_input(
    "Poids du produit (g)", min_value=1.0, value=500.0, step=50.0
)
product_volume_cm3 = st.sidebar.number_input(
    "Volume du produit (cm³)", min_value=1.0, value=1000.0, step=100.0
)
price = st.sidebar.number_input("Prix total (BRL)", min_value=1.0, value=89.90, step=10.0)
freight = st.sidebar.number_input("Frais de port (BRL)", min_value=0.0, value=12.50, step=1.0)
distance_km = st.sidebar.number_input(
    "Distance seller → customer (km)", min_value=0.0, value=250.0, step=10.0
)
n_items = st.sidebar.slider("Nombre d'articles", min_value=1, max_value=10, value=1)
purchase_dt = st.sidebar.date_input("Date d'achat", value=datetime(2024, 6, 10))


if st.button("🔮 Prédire le temps de livraison", type="primary"):
    payload = {
        "product_weight_g": product_weight_g,
        "product_volume_cm3": product_volume_cm3,
        "price": price,
        "freight_value": freight,
        "distance_km": distance_km,
        "order_item_id": n_items,
        "order_purchase_timestamp": datetime.combine(purchase_dt, datetime.min.time()).isoformat(),
    }
    try:
        resp = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=10)
        resp.raise_for_status()
        days = resp.json()["delivery_time_days"]
        st.success(f"⏱ Temps de livraison prédit : **{days:.1f} jours**")
        if days < 7:
            st.info("Livraison rapide — client probablement satisfait.")
        elif days < 15:
            st.warning("Livraison dans la moyenne Olist (médiane : ~12 jours).")
        else:
            st.error("Livraison longue — risque d'insatisfaction élevé.")
    except requests.RequestException as e:
        st.error(f"Erreur API : {e}")


with st.expander("ℹ️ À propos du modèle"):
    st.markdown(
        """
        **Algorithme** : RandomForestRegressor (scikit-learn)

        **Features brutes** : poids, volume, prix, fret, distance, nb articles, date d'achat

        **Features dérivées** : freight_ratio, log_price, is_multi_item, purchase_dow,
        purchase_month

        **Métrique** : R² > 0.70 | MAE < 5 jours

        **Artefact** : stocké dans MinIO, chargé au démarrage de l'API
        """
    )
