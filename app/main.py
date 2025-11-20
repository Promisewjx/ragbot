from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from functools import lru_cache
import time, requests

from .settings import settings
from .core import RAGEngine

app = FastAPI(title="Private RAG Chatbot", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str

class Reference(BaseModel):
    id: int; path: str; chunk_id: int; score: float

class ChatResponse(BaseModel):
    answer: str; references: List[Reference]

@app.get("/")
def home():
    return {"msg": "RAG service is running. Try /docs or POST /chat."}

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/debug/config")
def debug_config():
    return {
        "llm_backend": settings.llm_backend,
        "ollama_base": settings.ollama_base_url,
        "ollama_model": settings.ollama_model,
        "top_k": settings.top_k,
        "max_ctx": settings.max_context_chars,
        "rerank_enable": getattr(settings, "rerank_enable", False),
    }

@app.get("/debug/ollama")
def debug_ollama():
    try:
        r = requests.get(f"{settings.ollama_base_url}/api/tags", timeout=5)
        return {"status": r.status_code, "body": r.json()}
    except Exception as e:
        raise HTTPException(502, f"Ollama not reachable: {e}")

@lru_cache(maxsize=1)
def get_engine() -> RAGEngine:
    t0 = time.time()
    print(f"[RAG] backend={settings.llm_backend} model={settings.ollama_model}")
    eng = RAGEngine()
    print(f"[RAG] engine ready in {time.time()-t0:.2f}s")
    return eng

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        eng = get_engine()
        result = eng.ask(req.query)
        refs = [Reference(**r) for r in result["references"]]
        return ChatResponse(answer=result["answer"], references=refs)
    except Exception as e:
        # 将异常打印到控制台，便于你看到 500 的具体原因
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
