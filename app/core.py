# -*- coding: utf-8 -*-
"""
核心职责：
- 载入嵌入模型 / FAISS 索引 / 元数据
- 三路检索：Dense(向量)、Sparse(BM25)、Hybrid(融合)
- 可选重排：Cross-Encoder 对候选进行 pointwise 评分
- 构造 Prompt 并调用 LLM（mock / ollama）
- 对外暴露 ask()：返回 answer + references
"""
import json
import re
from typing import Optional, List, Dict, Tuple

import numpy as np
import faiss
import requests
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder  # 用于重排

from .settings import settings
from .prompts import build_user_prompt


# ------------------------- 工具：简易分词 -------------------------
def _simple_tokens(text: str) -> List[str]:
    text = text.lower()
    # 中文单字 + 英数串
    return re.findall(r"[\u4e00-\u9fff]|[a-z0-9]+", text)


def _minmax_norm(values: List[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-9:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - lo) / (hi - lo)


# ------------------------- RAG 引擎 -------------------------
class RAGEngine:
    """
    - 索引：FAISS（向量） + 内存 BM25（稀疏）
    - 生成：mock 或 Ollama（由 .env 决定）
    - 重排：Cross-Encoder（可选）
    """

    def __init__(self):
        # ↓ 如需强制 GPU，可改 device="cuda"
        self.emb = SentenceTransformer(settings.emb_model)

        # 载入向量索引与元数据
        self.index = faiss.read_index(str(settings.index_dir / "faiss.index"))
        with open(settings.index_dir / "meta.jsonl", "r", encoding="utf-8") as f:
            self.metas: List[Dict] = [json.loads(l) for l in f]

        # 构建 BM25 语料（一次性，内存态）
        self._bm25_corpus_tokens: List[List[str]] = [_simple_tokens(m["text"]) for m in self.metas]
        self._bm25 = BM25Okapi(self._bm25_corpus_tokens)

        # Cross-Encoder（延迟加载，首次用到时再加载，避免阻塞启动）
        self._reranker: Optional[CrossEncoder] = None

    # ---------- 向量检索 ----------
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        top_k = top_k or settings.top_k
        qv = self.emb.encode(query, normalize_embeddings=True).astype("float32")
        D, I = self.index.search(np.expand_dims(qv, 0), top_k)
        hits: List[Dict] = []
        for rank, idx in enumerate(I[0]):
            if idx < 0:
                continue
            meta = dict(self.metas[idx])
            meta["_score"] = float(D[0][rank])  # dense 分数
            hits.append(meta)
        return hits

    # ---------- BM25 检索 ----------
    def bm25_retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        top_k = top_k or settings.top_k
        q_tokens = _simple_tokens(query)
        scores = self._bm25.get_scores(q_tokens)  # numpy array
        idxs = np.argsort(scores)[::-1][:top_k]
        hits: List[Dict] = []
        for idx in idxs:
            meta = dict(self.metas[int(idx)])
            meta["_bm25"] = float(scores[int(idx)])
            hits.append(meta)
        return hits

    # ---------- Hybrid 检索 ----------
    def hybrid_retrieve(self, query: str, top_k: Optional[int] = None, alpha: float = 0.6) -> List[Dict]:
        """
        final = alpha * norm(dense) + (1 - alpha) * norm(bm25)
        """
        top_k = top_k or settings.top_k

        dense_hits = self.retrieve(query, top_k=top_k)
        bm25_hits  = self.bm25_retrieve(query, top_k=top_k)

        pool: Dict[Tuple[str, int], Dict] = {}
        for h in dense_hits:
            key = (h["path"], h["chunk_id"])
            pool.setdefault(key, dict(h))
            pool[key]["_dense"] = h.get("_score", 0.0)
        for h in bm25_hits:
            key = (h["path"], h["chunk_id"])
            if key not in pool:
                pool[key] = dict(h)
            pool[key]["_sparse"] = h.get("_bm25", 0.0)

        items = list(pool.values())
        d_norm = _minmax_norm([it.get("_dense", 0.0) for it in items])
        s_norm = _minmax_norm([it.get("_sparse", 0.0) for it in items])
        for it, d, s in zip(items, d_norm, s_norm):
            it["_hybrid"] = float(alpha * d + (1.0 - alpha) * s)

        items.sort(key=lambda x: x.get("_hybrid", 0.0), reverse=True)
        return items[:top_k]

    # ---------- Cross-Encoder 重排 ----------
    def _ensure_reranker(self):
        if self._reranker is None:
            # device 自动选择；如需强制 GPU: CrossEncoder(..., device="cuda")
            self._reranker = CrossEncoder(settings.reranker_model)

    def rerank(self, query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
        """
        用 Cross-Encoder 对 (query, passage) 做点对点打分；
        返回打分最高的 top_k 片段。
        """
        if not candidates:
            return candidates

        self._ensure_reranker()

        # 组装 (query, passage) 对
        pairs = [(query, c["text"]) for c in candidates]
        scores = self._reranker.predict(pairs, batch_size=32)  # numpy array

        # 附加分数并排序
        for c, s in zip(candidates, scores):
            c["_rerank"] = float(s)
        candidates.sort(key=lambda x: x.get("_rerank", 0.0), reverse=True)
        return candidates[:top_k]

    # ---------- 上下文裁剪 ----------
    def trim_context(self, contexts: List[Dict], max_chars: int) -> List[Dict]:
        out, total = [], 0
        for c in contexts:
            t = c["text"]
            if total + len(t) > max_chars:
                break
            out.append(c)
            total += len(t)
        return out

    # ---------- 生成 ----------
    def _gen_mock(self, prompt: str) -> str:
        return "【模拟回答】依据[1]等片段可得…… 引用：[1]"

    def _gen_ollama(self, prompt: str) -> str:
        url = f"{settings.ollama_base_url}/api/generate"
        payload = {"model": settings.ollama_model, "prompt": prompt, "stream": False}
        r = requests.post(url, json=payload, timeout=600)
        r.raise_for_status()
        return r.json().get("response", "").strip()

    def generate(self, question: str, contexts: List[Dict]) -> str:
        prompt = build_user_prompt(question, contexts)
        if settings.llm_backend == "mock":
            return self._gen_mock(prompt)
        elif settings.llm_backend == "ollama":
            return self._gen_ollama(prompt)
        else:
            raise RuntimeError(f"未知 LLM_BACKEND: {settings.llm_backend}")

    # ---------- 对外接口 ----------
    def ask(self, query: str) -> Dict:
        """
        Pipeline:
        Hybrid 召回（top_k）→ 取若干候选 → (可选) Cross-Encoder 重排 → 裁剪 → 生成
        """
        # 1) Hybrid 召回：先拿到一批候选（比最终 top_k 多）
        candidates = self.hybrid_retrieve(
            query,
            top_k=max(settings.top_k, settings.rerank_candidates),
            alpha=0.6,
        )

        # 2) (可选) 重排：对候选重新打分排序
        if settings.rerank_enable:
            reranked = self.rerank(query, candidates, top_k=settings.rerank_top_k)
            hits = reranked
        else:
            # 未开启重排则直接截断
            hits = candidates[:settings.top_k]

        # 3) 上下文裁剪（按字符预算）
        ctxs = self.trim_context(hits, settings.max_context_chars)

        # 4) 生成
        answer = self.generate(query, ctxs)

        # 5) 引用（优先展示 _rerank / 次选 _hybrid / 再次选 _score）
        refs = []
        for i, c in enumerate(ctxs):
            score = float(
                c.get("_rerank", c.get("_hybrid", c.get("_score", 0.0)))
            )
            refs.append(
                {"id": i + 1, "path": c["path"], "chunk_id": c["chunk_id"], "score": score}
            )
        return {"answer": answer, "references": refs}
