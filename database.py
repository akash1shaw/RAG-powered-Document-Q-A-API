from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
from dotenv import load_dotenv
load_dotenv()

# Create the database engine using your DATABASE_URL from .env
engine = create_engine(os.getenv("DATABASE_URL"))
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Document(Base):
    """
    Stores metadata about each uploaded PDF.
    We don't store the vectors here — FAISS handles that.
    We store the text chunks so we can return them to the user.
    """
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)         # original PDF name
    chunk_index = Column(Integer, nullable=False)     # which chunk (0, 1, 2...)
    chunk_text = Column(Text, nullable=False)         # the actual text
    faiss_index = Column(Integer, nullable=False)     # position in FAISS index
    created_at = Column(DateTime, default=datetime.utcnow)
    
class QueryHistory(Base):
    """
    Stores every question asked + the answer given.
    """
    __tablename__ = "query_history"

    id = Column(Integer, primary_key=True, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    sources_used = Column(Integer, nullable=False)    # how many chunks were used
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    """Call this once at startup to create tables if they don't exist."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency — gives each request its own DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
