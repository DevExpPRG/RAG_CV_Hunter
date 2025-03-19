from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.model_service import rag_chat

class ChatRequest(BaseModel):
    query: str

router = APIRouter()

@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Endpoint to ask questions to the chatbot."""
    if not request.query:
        raise HTTPException(status_code=400, detail="The query cannot be empty")
    return await rag_chat(request.query)

