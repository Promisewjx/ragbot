# -*- coding: utf-8 -*-
"""
从已建索引的片段中随机抽句，生成伪标注评测集：
- query: 随机抽到的句子（中英皆可，简单正则分句）
- gold: 采用 BM25 Top-1 作为伪标注（chunk 粒度）
用法示例：
  python -m eval.build_pseudo_qa --out eval/qa.jsonl --n 50 --per_doc 5
"""
import os
import re
import json
import random
import argparse
from typing import List, Dict, Any

from app.core import RAGEngine


def split_sentences(text: str) -> List[str]:
    """中英混合的简易分句：中文按 。！？；，英文按 .!?;:，保留较长句子"""
    # 先替换一些全角符号为半角，便于统一处理
    text = text.replace("\u3002", "。")
    # 用中文标点分割
    parts = re.split(r"[。！？；\n]+", text)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # 再按英文标点进一步切
        subs = re.split(r"[.!?;:]+", p)
        for s in subs:
            s = s.strip()
            if not s:
                continue
            # 过滤掉太短/太长/噪声句子
            # 中文：长度 8~120；英文：词数 5~40（都满足其一即可）
            cn_len = len(re.findall(r"[\u4e00-\u9fff]", s))
            en_words = re.findall(r"[A-Za-z0-9]+", s)
            if (8 <= cn_len <= 120) or (5 <= len(en_words) <= 40):
                out.append(s)
    # 去重，保持顺序
    dedup, seen = [], set()
    for s in out:
        if s not in seen:
            seen.add(s)
            dedup.append(s)
    return dedup


def build_pseudo_qa(n: int, per_doc: int, seed: int) -> List[Dict[str, Any]]:
    """
    从所有文档片段中抽样，生成 n 条伪标注
    - 每个文档最多采样 per_doc 个 query
    - 伪标注 gold = BM25 Top-1 (path, chunk_id)
    """
    random.seed(seed)

    eng = RAGEngine()  # 只用到 BM25 与元数据
    metas = eng.metas
    print(f"[build] loaded metas: {len(metas)} chunks")

    # 将同一文档的所有 chunk 聚合，便于按“文档维度”抽样
    docs: Dict[str, List[Dict[str, Any]]] = {}
    for m in metas:
        docs.setdefault(m["path"], []).append(m)

    # 对每个文档，先分句，准备候选句池
    per_doc_candidates: Dict[str, List[str]] = {}
    for path, chunks in docs.items():
        sents = []
        for c in chunks:
            sents.extend(split_sentences(c["text"]))
        random.shuffle(sents)
        per_doc_candidates[path] = sents
        # print(f"[build] {path}: {len(sents)} sent candidates")

    # 轮询各文档抽样，最多 per_doc，每轮按文档遍历直到凑够 n
    out = []
    doc_paths = list(docs.keys())
    round_idx = 0
    while len(out) < n and doc_paths:
        made_progress = False
        for path in doc_paths:
            if len(out) >= n:
                break
            pool = per_doc_candidates.get(path, [])
            if not pool:
                continue
            # 取一个句子作为 query
            q = pool.pop()
            # 用 BM25 检索 top1 作为伪 gold
            bm25_hits = eng.bm25_retrieve(q, top_k=1)
            if not bm25_hits:
                continue
            h = bm25_hits[0]
            record = {
                "query": q,
                "gold": {"type": "chunk", "items": [[h["path"], int(h["chunk_id"])]]},
            }
            out.append(record)
            made_progress = True
            # 控制每文档最多 per_doc
            # 简单做法：当该文档已达配额，就清空其候选池
            used = per_doc - sum(1 for r in out if r["gold"]["items"][0][0] == path)
            if used <= 0:
                per_doc_candidates[path] = []
        if not made_progress:
            break
        round_idx += 1

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="eval/qa.jsonl", help="输出 jsonl 路径")
    ap.add_argument("--n", type=int, default=50, help="生成样本数量")
    ap.add_argument("--per_doc", type=int, default=5, help="每个文档最多样本数")
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    data = build_pseudo_qa(n=args.n, per_doc=args.per_doc, seed=args.seed)

    with open(args.out, "w", encoding="utf-8") as f:
        for r in data:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[build] wrote {len(data)} samples to {args.out}")


if __name__ == "__main__":
    main()
