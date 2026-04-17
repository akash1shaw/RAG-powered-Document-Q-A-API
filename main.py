from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
import uvicorn
from database import get_db, init_db
from ingestion import process_pdf
from retrieval import answer_question

app = FastAPI(
    title="RAG Document Q&A API",
    description="Upload PDFs and ask questions about them using AI",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    init_db()

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]

@app.get("/")
async def root():
    return {"message": "RAG Q&A API is running"}

@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")
    file_bytes = await file.read()
    result = process_pdf(file_bytes, file.filename, db)
    return result

@app.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    db: Session = Depends(get_db)
):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    result = answer_question(request.question, db)
    return result

@app.get("/history")
async def get_history(db: Session = Depends(get_db)):
    from database import QueryHistory
    history = db.query(QueryHistory).order_by(QueryHistory.created_at.desc()).all()
    return [
        {
            "question": h.question,
            "answer": h.answer,
            "sources_used": h.sources_used,
            "timestamp": h.created_at.isoformat()
        }
        for h in history
    ]

@app.get("/documents")
async def list_documents(db: Session = Depends(get_db)):
    from database import Document
    from sqlalchemy import func
    results = (
        db.query(Document.filename, func.count(Document.id).label("chunks"))
        .group_by(Document.filename)
        .all()
    )
    return [{"filename": r.filename, "chunks": r.chunks} for r in results]

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
