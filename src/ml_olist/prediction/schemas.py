from datetime import datetime

from pydantic import BaseModel, Field


class OrderFeatures(BaseModel):
    """Features d'une commande Olist pour prédire le temps de livraison."""

    price: float = Field(..., gt=0, description="Prix total articles (BRL)")
    freight_value: float = Field(..., ge=0, description="Coût de livraison (BRL)")
    payment_installments: int = Field(..., ge=1, description="Nombre de versements")
    payment_value: float = Field(..., gt=0, description="Montant total payé (BRL)")
    order_item_id: int = Field(..., ge=1, description="Nombre d'articles")
    order_purchase_timestamp: datetime = Field(..., description="Date et heure d'achat")
    seller_state: str = Field(..., description="État du vendeur (ex: SP)")
    customer_state: str = Field(..., description="État du client (ex: RJ)")


class DeliveryPrediction(BaseModel):
    delivery_time_days: float = Field(..., description="Temps de livraison prédit (jours)")
    model_version: str = Field(default="latest")
