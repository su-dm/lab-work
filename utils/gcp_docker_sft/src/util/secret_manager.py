import os
from google.cloud import secretmanager
from dotenv import load_dotenv

def load_env_variables():
    load_dotenv()

def get_secret(secret_id, version_id="latest"):
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/sudm-ai/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")
