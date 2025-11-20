# -*- coding: utf-8 -*-
import os, json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np, faiss
from sentence_transformers import SentenceTransformer
from ingest.chunkers import semantic_chunk
from ingest.loaders import load_docs_from_dir

INDEX_DIR = Path(os.getenv("INDEX_DIR", "./data/index"))
RAW_DIR = Path("./data/raw")
EMB_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

def build_index(docs: List[Dict], emb) -> Tuple[faiss.Index, List[Dict]]:
    """
    文档 → 分块 → 向量化（归一化）→ 组装 FAISS Index + 元数据
    """
    vectors, metas = [], []
    total_chunks = 0

    print(f"📂 加载到 {len(docs)} 篇文档，开始分块和向量化...")

    for doc_id, d in enumerate(docs, 1):
        chunks = semantic_chunk(d["content"])
        print(f"  → 文档 {doc_id}/{len(docs)}: {d['path']} 分成 {len(chunks)} 个块")

        for i, ck in enumerate(chunks):
            v = emb.encode(ck, normalize_embeddings=True)
            vectors.append(v)
            metas.append({"path": d["path"], "chunk_id": i, "text": ck})
            total_chunks += 1

            # 每 100 个块打印一次
            if total_chunks % 100 == 0:
                print(f"    已处理 {total_chunks} 个分块...")

    if not vectors:
        raise SystemExit("⚠️ 没有可供索引的文本块，请检查 data/raw/ 下是否有内容。")

    X = np.vstack(vectors).astype("float32")
    index = faiss.IndexFlatIP(X.shape[1])  # 归一化后点积≈余弦
    index.add(X)

    print(f"✅ 全部分块完成，总计 {total_chunks} 个，向量维度 {X.shape[1]}")
    return index, metas

def save_index(index: faiss.Index, metas: List[Dict], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "faiss.index"))
    with open(out_dir / "meta.jsonl", "w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"💾 索引已保存到 {out_dir} (faiss.index + meta.jsonl)")

def main():
    print(f"🚀 使用 Embedding 模型: {EMB_MODEL_NAME}")
    emb = SentenceTransformer(EMB_MODEL_NAME, device="cuda")  # 可以加 device="cuda"
    docs = load_docs_from_dir(RAW_DIR)
    index, metas = build_index(docs, emb)
    save_index(index, metas, INDEX_DIR)

if __name__ == "__main__":
    main()
