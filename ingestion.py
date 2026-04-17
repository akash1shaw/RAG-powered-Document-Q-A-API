import io
import os
 
import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
 
load_dotenv()
 
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
 
EMBEDDING_DIM = 1536
faiss_index = faiss.IndexFlatL2(EMBEDDING_DIM)
 
 
def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    return "".join(page.extract_text() or "" for page in reader.pages)
 
 
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + chunk_size]))
        start += chunk_size - overlap
    return chunks
 
 
def get_embedding(text: str) -> np.ndarray:
    response = client.embeddings.create(model="text-embedding-3-small", input=text)
    return np.array(response.data[0].embedding, dtype=np.float32)
 
 
def add_to_faiss(embedding: np.ndarray) -> int:
    idx = faiss_index.ntotal
    faiss_index.add(embedding.reshape(1, -1))
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
        db.add(Document(
            filename=filename,
            chunk_index=i,
            chunk_text=chunk,
            faiss_index=faiss_idx,
        ))
        stored_chunks += 1
 
    db.commit()
    return {"filename": filename, "total_chunks": stored_chunks, "message": "PDF processed successfully"}
