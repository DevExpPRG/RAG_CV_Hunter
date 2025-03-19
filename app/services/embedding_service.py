# embedding_service.py
from langchain_ollama import OllamaEmbeddings

def create_embedding_model():
    """Devuelve el modelo de embedding cargado."""
    return OllamaEmbeddings(model="nomic-embed-text")
