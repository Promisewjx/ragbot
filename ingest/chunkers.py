# -*- coding: utf-8 -*-
import re
from typing import List

# 中文句末标点 + 英文句末 .!? + 换行分句（够用且不依赖网络）
_SENT_SPLIT = re.compile(r'(?<=[。！？!?\.])\s+|(?<=\n)\s*')

def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]

def semantic_chunk(text: str, target_chars=1400, max_chars=2000) -> List[str]:
    """
    以句子为单位累加到 ~350 tokens（字符近似≈1400），上限 ~500 tokens（≈2000字符）即切
    """
    chunks, buf, size = [], [], 0
    for s in split_sentences(text):
        L = len(s)
        if size + L > max_chars and buf:
            chunks.append(" ".join(buf)); buf, size = [s], L
        else:
            buf.append(s); size += L
            if size >= target_chars:
                chunks.append(" ".join(buf)); buf, size = [], 0
    if buf: chunks.append(" ".join(buf))
    return [c.strip() for c in chunks if c.strip()]