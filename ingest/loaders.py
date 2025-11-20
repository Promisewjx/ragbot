# -*- coding: utf-8 -*-
from pathlib import Path
from typing import List, Dict

def load_text_file(fp: Path) -> str:
    return fp.read_text(encoding="utf-8", errors="ignore")

def load_docs_from_dir(raw_dir: Path, exts=(".md", ".txt")) -> List[Dict]:
    """
    从 data/raw/ 读取 .md/.txt 文本；
    返回格式统一为 {"path": str, "content": str}
    """
    items = []
    for p in raw_dir.rglob("*"):
        if p.suffix.lower() in exts and p.is_file():
            items.append({"path": str(p), "content": load_text_file(p)})
    return items
