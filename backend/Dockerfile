FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends gcc python3-dev \

&& rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /install /usr/local

COPY src/ml_olist/prediction/ ./src/ml_olist/prediction/

COPY src/ml_olist/common/ ./src/ml_olist/common/

COPY backend/ ./backend/

ENV PYTHONPATH=/app

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]


docker build -t olist-delivery-backend -f backend/Dockerfile .

docker run --rm -p 8000:8000 --env-file .env olist-delivery-backend


# Test rapide

curl -X POST http://localhost:8000/predict \

-H 'Content-Type: application/json' \

-d '{

"price": 89.90, "freight_value": 12.50,

"payment_installments": 1, "payment_value": 102.40,

"order_item_id": 1,

"order_purchase_timestamp": "2018-06-10T09:00:00",

"seller_state": "SP", "customer_state": "SP"

}'

# Réponse : {"delivery_time_days": 8.3, "model_version": "latest"}
