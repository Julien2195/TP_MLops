import os
import time

import boto3
from botocore.exceptions import ClientError


def get_s3_client():
    """Client boto3 compatible MinIO / AWS S3."""
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("MINIO_ENDPOINT"),
        aws_access_key_id=os.getenv("MINIO_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("MINIO_SECRET_KEY"),
    )


def wait_for_minio(max_retries: int = 10, delay: int = 2):
    """Attend que MinIO soit prêt avant de continuer."""
    client = get_s3_client()
    for attempt in range(1, max_retries + 1):
        try:
            client.list_buckets()
            print("MinIO disponible.")
            return client
        except Exception as e:
            print(f"MinIO indisponible ({attempt}/{max_retries}) : {e}")
            time.sleep(delay)
    raise RuntimeError("MinIO inaccessible après plusieurs tentatives.")


def download_model(local_path: str) -> str:
    """Télécharge model_latest.joblib depuis MinIO vers local_path."""
    bucket = os.getenv("MINIO_BUCKET_MODELS")
    obj = os.getenv("MODEL_OBJECT_NAME")
    client = wait_for_minio()

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    try:
        client.download_file(bucket, obj, local_path)
        print(f"Modèle téléchargé : s3://{bucket}/{obj} → {local_path}")
        return local_path
    except ClientError as e:
        raise FileNotFoundError(
            f"Modèle s3://{bucket}/{obj} introuvable dans MinIO. "
            "Vérifier que upload_model_to_minio.py a été exécuté."
        ) from e
