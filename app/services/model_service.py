# model_service.py
from langchain_core.messages import HumanMessage, SystemMessage
from fastapi.responses import StreamingResponse
from langchain_ollama import OllamaLLM

from .embedding_service import create_embedding_model

# Cargar el modelo de IA
model = OllamaLLM(model="phi4")
embedding_model = create_embedding_model()

async def rag_chat(user_query):
    """Maneja la consulta del usuario con RAG y streaming de respuesta."""
    from .qdrant_service import qdrant_service  # Importar solo cuando se necesita
    retrieved_docs = qdrant_service.retrieve_docs(user_query)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    messages = [SystemMessage(content=f"Context: {context}"), HumanMessage(content=user_query)]

    async def response_generator():
        async for token in model.astream(messages):
            yield token

    return StreamingResponse(response_generator(), media_type="text/plain")
