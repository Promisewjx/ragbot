"""
构建 BM25 倒排索引
----------------------------------
从 data/index/meta.jsonl 读取语义块（chunk），
并建立 BM25 索引，保存到 data/index/bm25.pkl
"""

import json
import os
import pickle
from typing import List

from rank_bm25 import BM25Okapi


DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
META_PATH = os.path.join(DATA_DIR, "index/meta.jsonl")
BM25_OUT = os.path.join(DATA_DIR, "index/bm25.pkl")


def load_documents(meta_path: str) -> List[str]:
    """加载 meta.jsonl 中的文本内容，返回文档列表"""
    docs = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            docs.append(item["text"])
    return docs


def tokenize(doc: str) -> List[str]:
    """简单分词器（可根据需求改为jieba或更复杂的tokenizer）"""
    return doc.lower().split()


def build_bm25_index(docs: List[str]) -> BM25Okapi:
    """构建 BM25 索引"""
    tokenized_docs = [tokenize(d) for d in docs]
    bm25 = BM25Okapi(tokenized_docs)
    return bm25


def save_index(index, save_path):
    """保存 BM25 索引"""
    with open(save_path, "wb") as f:
        pickle.dump(index, f)
    print(f"[BM25] 已保存 BM25 索引到：{save_path}")


def main():
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f"找不到 meta.jsonl，请先运行 ingest.py 生成语义块：{META_PATH}")

    print(f"[BM25] 从 meta.jsonl 加载 chunk 文本：{META_PATH}")
    docs = load_documents(META_PATH)
    print(f"[BM25] 加载 {len(docs)} 个 chunk")

    print("[BM25] 正在构建 BM25 索引...")
    bm25 = build_bm25_index(docs)

    print("[BM25] 保存 BM25 索引...")
    save_index(bm25, BM25_OUT)

    print("[BM25] 构建完成！")


if __name__ == "__main__":
    main()
