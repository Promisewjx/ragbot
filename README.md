ragbot/
├─ app/
│  ├─ main.py              # FastAPI 服务
│  ├─ core.py              # 检索、重排、构造prompt、生成
│  ├─ models.py            # Pydantic 模型
│  ├─ settings.py          # 配置加载
│  └─ prompts.py           # 系统/回答模板
├─ ingest/
│  ├─ ingest.py            # 文档→分块→嵌入→入库
│  ├─ loaders.py           # 各类文档解析
│  └─ chunkers.py          # 分块逻辑
├─ data/
│  ├─ raw/                 # 原始文档
│  └─ index/               # 向量库与元数据
├─ eval/
│  └─ evaluate.py        # 简易评估脚本
├─ requirements.txt
├─ .env.example
└─ README.md
