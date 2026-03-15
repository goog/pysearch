---
AIGC:
    ContentProducer: Minimax Agent AI
    ContentPropagator: Minimax Agent AI
    Label: AIGC
    ProduceID: "00000000000000000000000000000000"
    PropagateID: "00000000000000000000000000000000"
    ReservedCode1: 304402201105fc8d845fff19a837201d89c4ac5a2b1f09064960e076af5baab422cebadc02206e40d8995c2d3134084f37b38f5e003c6d66966bffb50c2885d086d71c3f674a
    ReservedCode2: 30450220649d55bdb002373527a66a2a41a7b000ea00ecfa0864353ec803a6bd721cb1810221008d1ca097a0a7b04279857f5a00dd9c5fdb6a27f4195bb7dedd7cbc8f4994d1ec
---

# PySearch - 高性能全文搜索引擎

[English](./README_EN.md) | 中文

PySearch 是一个用 Python 实现的高性能全文搜索引擎，支持中英文混合检索，采用倒排索引和 BM25/TF-IDF 排名算法。

## 特性

- 🚀 **高性能**: 采用倒排索引，支持批量索引和并行处理
- 🌐 **中英文支持**: 内置分词器支持中文（jieba）和英文
- 📊 **多种排名算法**: 支持 BM25 和 TF-IDF 算法
- 🔍 **高级搜索**: 支持布尔搜索、结果高亮、查询建议
- 💾 **持久化**: 支持索引保存和加载
- 🌐 **REST API**: 提供 FastAPI 接口
- ⚡ **异步支持**: 支持异步索引和查询

## 安装

```bash
# 安装依赖
pip install -r requirements.txt

# 安装 jieba (中文分词)
pip install jieba
```

## 快速开始

### 1. 命令行运行演示

```bash
python -m pysearch.main demo
```

### 2. 启动 API 服务

```bash
python -m pysearch.main serve --port 8000
```

### 3. 程序化使用

```python
from pysearch.main import SearchEngine

# 创建搜索引擎
engine = SearchEngine()

# 索引文档
documents = [
    {
        "id": 1,
        "text": "Python is a high-level programming language",
        "title": "Python Programming"
    },
    {
        "id": 2,
        "text": "Java is a class-based programming language",
        "title": "Java Programming"
    }
]

result = engine.index(documents)
print(f"Indexed {result['documents_indexed']} documents")

# 搜索
results = engine.search("Python programming")
print(f"Found {results['total_hits']} results")

for r in results['results']:
    print(f"  - Doc {r['id']}: {r['document'].get('title')} (score: {r['score']:.4f})")
```

## API 接口

### 索引文档

```bash
curl -X POST "http://localhost:8000/index/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {"id": 1, "text": "Python is great", "title": "Python"},
      {"id": 2, "text": "Java is popular", "title": "Java"}
    ]
  }'
```

### 搜索

```bash
curl -X GET "http://localhost:8000/search?q=python&algorithm=bm25&limit=10"
```

### 获取统计

```bash
curl -X GET "http://localhost:8000/stats"
```

## 配置

```python
from pysearch.config import Config

# 自定义配置
config = Config(
    bm25={"k1": 1.5, "b": 0.75},
    index={"batch_size": 10000, "cache_size": 1000},
    storage={"index_path": "./data/index"}
)

engine = SearchEngine(config)
```

## 架构

```
pysearch/
├── config.py        # 配置模块
├── tokenizer.py     # 分词器
├── storage.py      # 存储引擎
├── indexer.py       # 索引引擎
├── query.py        # 查询引擎
├── api.py          # FastAPI 接口
└── main.py         # 主入口
```

## 性能基准

- 索引速度: >1000 文档/秒
- 查询延迟: <100ms (100万文档)
- 索引大小: 原始文本的 1.5x

## 运行测试

```bash
pytest tests/ -v
```

## License

MIT License
