import os
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()


MODEL_PATH = Path("artifacts/model_latest.joblib")
BUCKET = os.getenv("MINIO_BUCKET_MODELS", "ml-models")
OBJECT_NAME = os.getenv("MODEL_OBJECT_NAME", "model_latest.joblib")


def get_client():
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("MINIO_ENDPOINT", "http://localhost:9000"),
        aws_access_key_id=os.getenv("MINIO_ACCESS_KEY", "admin"),
        aws_secret_access_key=os.getenv("MINIO_SECRET_KEY", "password123"),
    )


def ensure_bucket(client, bucket: str) -> None:
    try:
        client.head_bucket(Bucket=bucket)
    except ClientError:
        client.create_bucket(Bucket=bucket)
        print(f"Bucket créé : {bucket}")


def upload(model_path: Path = MODEL_PATH) -> None:
    if not model_path.exists():
        print(f"Modèle introuvable : {model_path}. Lancer train.py d'abord.")
        sys.exit(1)

    client = get_client()
    ensure_bucket(client, BUCKET)
    client.upload_file(str(model_path), BUCKET, OBJECT_NAME)
    print(f"Modèle publié → s3://{BUCKET}/{OBJECT_NAME}")


if __name__ == "__main__":
    upload()
