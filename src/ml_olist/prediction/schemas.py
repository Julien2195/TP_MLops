from datetime import datetime

from pydantic import BaseModel, Field


class OrderFeatures(BaseModel):
    """Features d'une commande Olist pour prédire le temps de livraison."""

    product_weight_g: float = Field(..., gt=0, description="Poids du produit (grammes)")
    product_volume_cm3: float = Field(..., gt=0, description="Volume du produit (cm³)")
    price: float = Field(..., gt=0, description="Prix total articles (BRL)")
    freight_value: float = Field(..., ge=0, description="Coût de livraison (BRL)")
    distance_km: float = Field(..., ge=0, description="Distance seller → customer (km)")
    order_item_id: int = Field(..., ge=1, description="Nombre d'articles")
    order_purchase_timestamp: datetime = Field(..., description="Date et heure d'achat")


class DeliveryPrediction(BaseModel):
    delivery_time_days: float = Field(..., description="Temps de livraison prédit (jours)")
    model_version: str = Field(default="latest")
