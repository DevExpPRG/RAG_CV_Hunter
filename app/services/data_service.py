from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

def extract_text_from_pdf(pdf_path):
    """Extrae texto de un archivo PDF."""
    reader = PdfReader(pdf_path)
    text = "".join([page.extract_text() or "" for page in reader.pages])
    return text.strip() or None

def process_pdf(pdf_path):
    """Divide el texto en fragmentos y los devuelve como documentos."""
    text = extract_text_from_pdf(pdf_path)
    if text:
        splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)
        return [Document(page_content=chunk, metadata={"source": pdf_path}) for chunk in splitter.split_text(text)]
    return []
