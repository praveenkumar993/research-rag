import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager

from rag_engine import load_vectorstore, build_rag_chain, get_loaded_papers, query_papers

vectorstore = None
rag_chain = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vectorstore, rag_chain
    print("🚀 Loading vectorstore and building RAG chain...")
    try:
        vectorstore = load_vectorstore()
        rag_chain = build_rag_chain(vectorstore)
        print("✅ RAG system ready.")
    except Exception as e:
        print(f"❌ Failed to load vectorstore: {e}")
    yield


app = FastAPI(title="Research RAG API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    question: str


@app.get("/api/papers")
def list_papers():
    if vectorstore is None:
        raise HTTPException(status_code=503, detail="Vectorstore not loaded.")
    papers = get_loaded_papers(vectorstore)
    return {"papers": papers, "count": len(papers)}


@app.post("/api/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG chain not initialized.")
    try:
        answer = query_papers(request.question, rag_chain)
        return QueryResponse(answer=answer, question=request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "vectorstore_loaded": vectorstore is not None,
        "rag_chain_ready": rag_chain is not None,
    }


# Serve frontend
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
