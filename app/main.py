from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.chat import router as chat_router

app = FastAPI(
    title="Chatbot API",
    description="API basada en RAG para responder preguntas",
    version="1.0"
)

# Configuración de CORS
origins = [
    "http://localhost",  # Permitir solicitudes desde localhost
    "http://localhost:3000",  # Cambia esto si tu frontend está en otro puerto
    "http://127.0.0.1",  # Permitir solicitudes desde 127.0.0.1
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registrar rutas
app.include_router(chat_router)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Chatbot API"}
