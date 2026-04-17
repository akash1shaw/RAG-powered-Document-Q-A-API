import os

from openai import OpenAI
from ingestion import faiss_index, get_embedding

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def search_similar_chunks(query: str, top_k: int = 5) -> list[int]:
    query_vector = get_embedding(query).reshape(1, -1)
    _, I = faiss_index.search(query_vector, top_k)
    return I[0].tolist()


def get_chunks_from_db(faiss_indices: list[int], db) -> list[dict]:
    from database import Document

    chunks = []
    for idx in faiss_indices:
        doc = db.query(Document).filter(Document.faiss_index == idx).first()
        if doc:
            chunks.append({
                "text": doc.chunk_text,
                "filename": doc.filename,
                "chunk_index": doc.chunk_index,
            })
    return chunks


def generate_answer(question: str, context_chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join(c["text"] for c in context_chunks)
    prompt = f"""You are a helpful assistant that answers questions based ONLY on the provided document context.
If the answer is not found in the context, say "I couldn't find this in the uploaded documents."

Context from documents:
{context}

Question: {question}
Answer:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content


def answer_question(question: str, db) -> dict:
    from database import QueryHistory

    faiss_indices = search_similar_chunks(question, top_k=5)
    chunks = get_chunks_from_db(faiss_indices, db)

    if not chunks:
        return {"answer": "No documents have been uploaded yet.", "sources": []}

    answer = generate_answer(question, chunks)

    db.add(QueryHistory(question=question, answer=answer, sources_used=len(chunks)))
    db.commit()

    return {
        "answer": answer,
        "sources": [
            {"filename": c["filename"], "chunk_preview": c["text"][:150] + "..."}
            for c in chunks
        ],
    }
