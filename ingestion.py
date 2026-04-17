import faiss
import numpy as np
from pypdf import PdfReader
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBEDDING_DIM = 1536
faiss_index = faiss.IndexFlatL2(EMBEDDING_DIM)
def extract_text_from_pdf(file_bytes: bytes) -> str:
  
    import io
    reader = PdfReader(io.BytesIO(file_bytes))
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() or ""
    return full_text
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  
    return chunks


def get_embedding(text: str) -> np.ndarray:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    vector = response.data[0].embedding
    return np.array(vector, dtype=np.float32)


def add_to_faiss(embedding: np.ndarray) -> int:
    vector = embedding.reshape(1, -1)
    idx = faiss_index.ntotal  
    faiss_index.add(vector)
    return idx


def process_pdf(file_bytes: bytes, filename: str, db) -> dict:
   
    from database import Document
    text = extract_text_from_pdf(file_bytes)
    chunks = chunk_text(text)

    stored_chunks = 0
    for i, chunk in enumerate(chunks):
        if len(chunk.strip()) < 20:
            continue

        embedding = get_embedding(chunk)
        faiss_idx = add_to_faiss(embedding)
        doc = Document(
            filename=filename,
            chunk_index=i,
            chunk_text=chunk,
            faiss_index=faiss_idx
        )
        db.add(doc)
        stored_chunks += 1

    db.commit()

    return {
        "filename": filename,
        "total_chunks": stored_chunks,
        "message": "PDF processed successfully"
    }
