print("== START test_ingest.py ==")  # 必然输出

import sys, os
print("cwd:", os.getcwd())
print("sys.executable:", sys.executable)
print("sys.path[0]:", sys.path[0])

# 确保 ingest 被识别为包（建议存在空 __init__.py）
try:
    import ingest
    print("import ingest: OK (package)")
except Exception as e:
    print("import ingest: FAILED ->", repr(e))

from ingest.loaders import load_docs_from_dir
from ingest.chunkers import semantic_chunk
from pathlib import Path

raw_dir = Path("data/raw").resolve()
print("raw_dir exists:", raw_dir.exists(), "path:", str(raw_dir))

docs = load_docs_from_dir(raw_dir)
print("docs count:", len(docs))
if docs:
    print("first doc path:", docs[0]["path"])
    chunks = semantic_chunk(docs[0]["content"])
    print("first doc chunks:", len(chunks), "sample:", chunks[:1])
else:
    print("No docs found under data/raw. Put a .txt/.md file there.")
