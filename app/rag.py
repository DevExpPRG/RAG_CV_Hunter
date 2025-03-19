import logging
from dotenv import load_dotenv
import os

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.indexes import SQLRecordManager
from PyPDF2 import PdfReader
from langchain_core.messages import HumanMessage, SystemMessage

# Logging Settings
logging.basicConfig(level=logging.INFO)

# Model Configuration
model = OllamaLLM(model="phi4")
embedding_model = OllamaEmbeddings(model="nomic-embed-text")

load_dotenv()
#https://cloud.qdrant.io/accounts/8d635452-a636-4874-96bb-ae4f714c1ee6/overview
QDRANT_SERVER = os.getenv("QDRANT_SERVER")
API_KEY = os.getenv("API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
EMBEDDING_SIZE = int(os.getenv("EMBEDDING_SIZE", 768))

def connect_qdrant():
    qdrant_client = QdrantClient(url=QDRANT_SERVER, api_key=API_KEY, timeout=20)
    try:
        qdrant_client.get_collections()
        logging.info("✅ Successfully connected to Qdrant")
    except Exception as e:
        logging.error(f"❌ Error connecting to Qdrant: {e}")
        exit(1)
    return qdrant_client

client = connect_qdrant()

# Creating a collection if it does not exist
if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_SIZE, distance=Distance.COSINE),
    )

# Vector storage configuration
vectorstore = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embedding_model,
    retrieval_mode=RetrievalMode.DENSE,
)

# Configuring SQLRecordManager
record_manager = SQLRecordManager(
    f"qdrant/{COLLECTION_NAME}", db_url="sqlite:///record_manager_cache.sql"
)
record_manager.create_schema()

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = "".join([page.extract_text() or "" for page in reader.pages])
    return text.strip() or None

def process_pdf(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    if text:
        splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)
        docs = [Document(page_content=chunk, metadata={"source": pdf_path}) for chunk in splitter.split_text(text)]
        return docs
    else:
        logging.error("❌Could not extract text from PDF")
        return []

def retrieve_docs(query, k=5):
    """Retrieve relevant documents from Qdrant."""
    return vectorstore.similarity_search(query, k=k)

def format_context(docs):
    """Formats retrieved documents for the model."""
    return "\n".join([doc.page_content for doc in docs])

async def rag_chat(user_query):
    """Main RAG management."""
    retrieved_docs = retrieve_docs(user_query)
    context = format_context(retrieved_docs)
    messages = [SystemMessage(content=f"Context: {context}"), HumanMessage(content=user_query)]
    async for token in model.astream(messages):
        print(token, end="", flush=True)

if __name__ == "__main__":
    import asyncio
    asyncio.run(rag_chat("¿Cómo se llama el candidato?"))
