from typing import List, Dict, Any, Sequence, Optional, Tuple
import torch
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    """
    sentence-transformers CrossEncoder 版本的重排器。
    典型模型：
      - 英文常用：'cross-encoder/ms-marco-MiniLM-L-6-v2'（轻量快）
      - 更准但慢：'cross-encoder/ms-marco-MiniLM-L-12-v2'
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
        batch_size: int = 32,
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = CrossEncoder(
            model_name,
            device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
            max_length=max_length,     # 控制截断，避免溢出
        )

    def score_pairs(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        # pairs: [(query, doc_text), ...]
        scores = self.model.predict(list(pairs), batch_size=self.batch_size)
        try:
            return scores.tolist()
        except Exception:
            return [float(s) for s in scores]

    def rerank(
        self,
        query: str,
        docs: Sequence[Dict[str, Any] | str],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        # docs 可为 {"text": "...", ...} 或 纯字符串
        pairs = [(query, (d["text"] if isinstance(d, dict) else str(d))) for d in docs]
        scores = self.score_pairs(pairs)
        merged = list(zip(docs, scores))
        merged.sort(key=lambda x: x[1], reverse=True)
        if top_k:
            merged = merged[:top_k]

        out: List[Dict[str, Any]] = []
        for item, s in merged:
            if isinstance(item, dict):
                item = {**item, "rerank_score": float(s)}
            else:
                item = {"text": str(item), "rerank_score": float(s)}
            out.append(item)
        return out
