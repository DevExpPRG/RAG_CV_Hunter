import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

class Config:
    QDRANT_SERVER = os.getenv("QDRANT_SERVER")
    API_KEY = os.getenv("API_KEY")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME")
    EMBEDDING_SIZE = int(os.getenv("EMBEDDING_SIZE", 768))
