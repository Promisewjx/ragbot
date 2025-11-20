# -*- coding: utf-8 -*-
import os
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv(override=True) 

class Settings(BaseModel):
    index_dir: Path = Path(os.getenv("INDEX_DIR", "./data/index"))
    emb_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    llm_backend: str = os.getenv("LLM_BACKEND", "ollama")  # mock 或 ollama
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    top_k: int = int(os.getenv("TOP_K", "5"))
    max_context_chars: int = int(os.getenv("MAX_CONTEXT_CHARS", "6000"))
    rerank_enable: bool = os.getenv("RERANK_ENABLE", "false").lower() == "true"
    reranker_model: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    rerank_candidates: int = int(os.getenv("RERANK_CANDIDATES", "20"))
    rerank_top_k: int = int(os.getenv("RERANK_TOP_K", "5"))


settings = Settings()
