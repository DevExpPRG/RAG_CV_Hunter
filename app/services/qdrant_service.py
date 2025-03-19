# qdrant_service.py
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance


from app.config import Config
from .embedding_service import create_embedding_model
import logging

logging.basicConfig(level=logging.INFO)

class QdrantService:
    def __init__(self, embedding_model):
        """Inicializa la conexión con Qdrant y la integración con el modelo de embedding."""
        self.client = QdrantClient(url=Config.QDRANT_SERVER, api_key=Config.API_KEY, timeout=20)
        self._ensure_collection()

        self.vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=Config.COLLECTION_NAME,
            embedding=embedding_model,
            retrieval_mode=RetrievalMode.DENSE,
        )

    def _ensure_collection(self):
        """Crea la colección en Qdrant si no existe."""
        if not self.client.collection_exists(Config.COLLECTION_NAME):
            self.client.create_collection(
                collection_name=Config.COLLECTION_NAME,
                vectors_config=VectorParams(size=Config.EMBEDDING_SIZE, distance=Distance.COSINE),
            )
        logging.info("✅ Colección asegurada en Qdrant")

    def retrieve_docs(self, query, k=5):
        """Recupera documentos relevantes de Qdrant."""
        return self.vectorstore.similarity_search(query, k=k)

# Instanciar el servicio de Qdrant solo cuando se pase el modelo de embedding
embedding_model = create_embedding_model()
qdrant_service = QdrantService(embedding_model)
