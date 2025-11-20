# -*- coding: utf-8 -*-
"""
评测检索质量：Recall@K、nDCG@10、MRR@10、平均延迟等
默认调用你项目的 RAGEngine，并支持选择 dense/bm25/hybrid 三种检索。
用法示例：
  python -m eval.evaluate --data eval/qa.jsonl --metric_k 1 3 5 10 --by chunk --retriever hybrid --alpha 0.6
"""
import json
import time
import math
import argparse
from typing import List, Tuple, Dict, Any

import numpy as np

# 复用你的项目代码
from app.core import RAGEngine
from app.settings import settings


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, help="标注数据 JSONL 路径")
    p.add_argument("--metric_k", type=int, nargs="+", default=[1, 3, 5, 10], help="计算 Recall@K 的 K 列表")
    p.add_argument("--by", choices=["chunk", "path"], default="chunk", help="gold 匹配粒度")
    p.add_argument("--retriever", choices=["dense", "bm25", "hybrid"], default="hybrid", help="选择检索器")
    p.add_argument("--alpha", type=float, default=0.6, help="hybrid 融合权重（dense 权重）")
    p.add_argument("--topk", type=int, default=20, help="每条样本检索的候选数（越大越稳）")
    return p.parse_args()


# ----------------- IR 指标 -----------------

def dcg(rels: List[float], k: int) -> float:
    """DCG@k, rels 是按检索顺序的相关性（一般二值 0/1）"""
    s = 0.0
    for i, r in enumerate(rels[:k], start=1):
        s += (2**r - 1) / math.log2(i + 1)
    return s

def idcg(ground_truth_count: int, k: int) -> float:
    """最佳情况的 DCG（把所有相关项排在最前面）"""
    rels = [1.0] * min(ground_truth_count, k) + [0.0] * max(0, k - ground_truth_count)
    return dcg(rels, k)

def ndcg_at_k(rels: List[float], k: int, gt_count: int) -> float:
    ideal = idcg(gt_count, k)
    if ideal == 0:
        return 0.0
    return dcg(rels, k) / ideal

def mrr_at_k(rels: List[float], k: int) -> float:
    """MRR@k：第一个相关文档的倒数排名"""
    for i, r in enumerate(rels[:k], start=1):
        if r > 0:
            return 1.0 / i
    return 0.0

def recall_at_k(rels: List[float], k: int, gt_count: int) -> float:
    if gt_count == 0:
        return 0.0
    hit = sum(rels[:k])
    # 对于“chunk 粒度”评估，gt_count 通常是 gold chunk 的个数；path 粒度则是 gold path 个数
    return float(hit) / float(gt_count)


# ----------------- 匹配逻辑 -----------------

def as_key_by(mode: str, item: Dict[str, Any]) -> Tuple:
    """
    mode="chunk"  → 使用 (path, chunk_id)
    mode="path"   → 使用 (path,)
    """
    if mode == "chunk":
        return (item["path"], int(item["chunk_id"]))
    else:
        return (item["path"],)

def load_gold(entry: Dict[str, Any], mode: str) -> List[Tuple]:
    g = entry["gold"]
    assert g["type"] == mode, f"gold.type={g['type']} 与 --by={mode} 不一致"
    if mode == "chunk":
        return [(p, int(cid)) for p, cid in g["items"]]
    else:
        return [(p,) for p in g["items"]]


# ----------------- 主流程 -----------------

def main():
    args = parse_args()
    print(f"== eval on {args.data} ==")
    print(f"retriever={args.retriever} alpha={args.alpha} by={args.by} topk={args.topk} Ks={args.metric_k}")

    # 构造引擎（不触发重排/生成）
    eng = RAGEngine()

    # 读数据
    data = []
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    assert data, "标注文件为空"

    # 汇总容器
    Ks = sorted(list(set(args.metric_k)))
    m_recall = {k: [] for k in Ks}
    m_ndcg10 = []
    m_mrr10 = []
    lat_ms = []

    for idx, ex in enumerate(data, start=1):
        q = ex["query"]
        gold_keys = set(load_gold(ex, args.by))
        gt_count = len(gold_keys)

        # 检索
        t0 = time.time()
        if args.retriever == "dense":
            hits = eng.retrieve(q, top_k=args.topk)
        elif args.retriever == "bm25":
            hits = eng.bm25_retrieve(q, top_k=args.topk)
        else:
            hits = eng.hybrid_retrieve(q, top_k=args.topk, alpha=args.alpha)
        lat_ms.append((time.time() - t0) * 1000)

        # 构造相关性序列（按检索结果顺序）
        rels = []
        for h in hits:
            key = as_key_by(args.by, h)
            rels.append(1.0 if key in gold_keys else 0.0)
        # 如果命中少于 topk，后面的相关性视为 0
        if len(rels) < args.topk:
            rels.extend([0.0] * (args.topk - len(rels)))

        # 计算指标
        for k in Ks:
            m_recall[k].append(recall_at_k(rels, k, gt_count))
        m_ndcg10.append(ndcg_at_k(rels, 10, gt_count))
        m_mrr10.append(mrr_at_k(rels, 10))

    # 汇总
    def avg(xs: List[float]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    print("\n== Results ==")
    for k in Ks:
        print(f"Recall@{k}: {avg(m_recall[k]):.4f}")
    print(f"nDCG@10   : {avg(m_ndcg10):.4f}")
    print(f"MRR@10    : {avg(m_mrr10):.4f}")
    print(f"Latency   : avg {avg(lat_ms):.1f} ms | p50 {np.percentile(lat_ms,50):.1f} ms | p95 {np.percentile(lat_ms,95):.1f} ms | n={len(lat_ms)}")


if __name__ == "__main__":
    main()
