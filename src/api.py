"""
Phase 5: FastAPI Service for RAG Pipeline
Exposes:
  - POST /v1/ask      : Ask questions with hybrid retrieval, confidence, and citations.
  - GET /v1/documents : List all indexed documents.
  - POST /v1/ingest   : Upload new text files to index dynamically.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import shutil
from pathlib import Path

app = FastAPI(
    title="Hybrid RAG Pipeline API",
    description="Production-grade RAG pipeline API with dense/sparse retrieval and citation verification.",
    version="1.0.0"
)

# Enable CORS for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request / Response Schemas
class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3

class AskResponse(BaseModel):
    query: str
    answer: str
    citations: List[int]
    confidence: Dict[str, Any]
    retrieved_chunks: List[Dict[str, Any]]
    abstained: bool = False

# Lazy load imports to avoid startup delay
_retriever = None
_generator = None
_verifier = None
_scorer = None
_idk = None

def get_rag_components():
    global _retriever, _generator, _verifier, _scorer, _idk
    if _retriever is None:
        from src.vector_store import VectorStore
        from src.bm25_search import BM25Search
        from src.embeddings_manager import EmbeddingsManager
        from src.retriever import HybridRetriever
        from src.generator import AnswerGenerator, CitationVerifier
        from src.confidence import AnswerConfidenceScorer, IDontKnowHandler
        
        vs = VectorStore()
        bm25 = BM25Search()
        bm25.load_index()
        em = EmbeddingsManager()
        
        _retriever = HybridRetriever(vs, bm25, em)
        _generator = AnswerGenerator()
        _verifier = CitationVerifier()
        _scorer = AnswerConfidenceScorer()
        _idk = IDontKnowHandler()
        
    return _retriever, _generator, _verifier, _scorer, _idk

@app.post("/v1/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    try:
        retriever, generator, verifier, scorer, idk = get_rag_components()
        
        # 1. Retrieve
        retrieval = retriever.retrieve(request.question, top_k=request.top_k)
        chunks = retrieval.get("top_results", [])
        
        # 2. Generate
        gen_out = generator.generate(request.question, chunks)
        answer = gen_out.get("answer", "")
        citations = gen_out.get("citations", [])
        
        # 3. Verify Citations
        verif = verifier.verify(answer, chunks)
        
        # 4. Score Confidence
        confidence = scorer.score(
            query=request.question,
            answer=answer,
            chunks=chunks,
            verification=verif
        )
        
        # 5. Handle "I don't know" case
        if idk.should_abstain(confidence):
            idk_resp = idk.build_response(request.question, chunks, confidence)
            return AskResponse(
                query=request.question,
                answer=idk_resp["answer"],
                citations=[],
                confidence=confidence,
                retrieved_chunks=[{"text": c["text"], "metadata": c["metadata"]} for c in chunks],
                abstained=True
            )
            
        return AskResponse(
            query=request.question,
            answer=answer,
            citations=citations,
            confidence=confidence,
            retrieved_chunks=[{"text": c["text"], "metadata": c["metadata"]} for c in chunks],
            abstained=False
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG Error: {str(e)}")

@app.get("/v1/documents")
async def list_documents():
    """List all documents currently stored in the vector database."""
    try:
        from src.vector_store import VectorStore
        vs = VectorStore()
        all_chunks = vs.get_all()
        
        # Group by source
        docs = {}
        for chunk in all_chunks:
            source = chunk.get("metadata", {}).get("source", "unknown")
            if source not in docs:
                docs[source] = {"source": source, "chunks": 0}
            docs[source]["chunks"] += 1
            
        return list(docs.values())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """Upload a text file, process it, and index it into ChromaDB + BM25."""
    if not file.filename.endswith('.txt'):
        raise HTTPException(status_code=400, detail="Only .txt files are supported currently.")
        
    try:
        # Save temporary file
        upload_dir = Path("data/documents")
        upload_dir.mkdir(parents=True, exist_ok=True)
        dest_path = upload_dir / file.filename
        
        with dest_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Re-index
        from src.ingest import DocumentLoader
        from src.chunking_strategies import ChunkingFactory
        from src.embeddings_manager import EmbeddingsManager
        from src.vector_store import VectorStore
        from src.bm25_search import BM25Search
        
        loader = DocumentLoader()
        # Reload all docs
        docs = loader.load_from_directory()
        
        all_chunks = []
        for doc in docs:
            chunks = ChunkingFactory.create_chunks(doc.content, doc.source, strategy="recursive")
            all_chunks.extend(chunks)
            
        em = EmbeddingsManager()
        vs = VectorStore()
        vs.reset()  # Clear existing index
        
        embedded = em.embed_chunks(all_chunks)
        embeddings = [r.embedding for r in embedded]
        vs.add_chunks(all_chunks[:len(embedded)], embeddings)
        
        bm25 = BM25Search()
        bm25.build_index(all_chunks[:len(embedded)])
        
        # Force reload retriever
        global _retriever
        _retriever = None
        
        return {"status": "success", "message": f"Successfully indexed {file.filename} with {len(all_chunks)} chunks"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")