"""
PySearch Example Scripts
========================

Demonstrates various features of the PySearch engine.

Author: MiniMax Agent
"""

import json
import time
from pysearch.main import SearchEngine
from pysearch.config import Config


def example_basic_usage():
    """Basic usage example."""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)

    # Create engine
    engine = SearchEngine()

    # Sample documents
    documents = [
        {
            "id": 1,
            "text": "Python is a high-level programming language. Python is widely used in data science and machine learning.",
            "title": "Python Programming",
            "category": "programming"
        },
        {
            "id": 2,
            "text": "Java is a class-based, object-oriented programming language. Java runs on billions of devices.",
            "title": "Java Programming",
            "category": "programming"
        },
        {
            "id": 3,
            "text": "Search engines are software systems that search the World Wide Web. Google is the most popular search engine.",
            "title": "Search Engines",
            "category": "technology"
        },
        {
            "id": 4,
            "text": "Machine learning is a subset of artificial intelligence. Deep learning is a subset of machine learning.",
            "title": "Machine Learning",
            "category": "ai"
        },
        {
            "id": 5,
            "text": "自然语言处理是人工智能的一个重要分支。它涉及计算机对自然语言的理解和生成。",
            "title": "自然语言处理",
            "category": "ai"
        },
        {
            "id": 6,
            "text": "搜索引擎优化是一种提高网站在搜索引擎中排名的技术。SEO对于网站流量非常重要。",
            "title": "搜索引擎优化",
            "category": "marketing"
        }
    ]

    # Index documents
    print("\n[1] Indexing documents...")
    result = engine.index(documents)
    print(f"    Indexed {result['documents_indexed']} documents")
    print(f"    Time: {result['indexing_time_ms']:.2f}ms")
    print(f"    Speed: {result['documents_per_second']:.2f} docs/sec")

    # Search
    print("\n[2] Searching for 'Python'...")
    results = engine.search("Python")
    print(f"    Found {results['total_hits']} results in {results['execution_time_ms']:.2f}ms")

    for r in results['results']:
        print(f"    - Doc {r['id']}: {r['document'].get('title')} (score: {r['score']:.4f})")

    print("\n[3] Searching for 'search engine'...")
    results = engine.search("search engine")
    print(f"    Found {results['total_hits']} results in {results['execution_time_ms']:.2f}ms")

    for r in results['results']:
        print(f"    - Doc {r['id']}: {r['document'].get('title')} (score: {r['score']:.4f})")


def example_chinese_search():
    """Chinese search example."""
    print("\n" + "=" * 60)
    print("Example 2: Chinese Search")
    print("=" * 60)

    engine = SearchEngine()

    documents = [
        {
            "id": 1,
            "text": "自然语言处理是人工智能的重要应用领域。它包括文本分析、情感分析、机器翻译等任务。",
            "title": "自然语言处理"
        },
        {
            "id": 2,
            "text": "机器学习算法可以自动从数据中学习和改进。常见的机器学习算法包括监督学习、无监督学习和强化学习。",
            "title": "机器学习"
        },
        {
            "id": 3,
            "text": "深度学习是机器学习的一个分支，使用神经网络模型进行特征学习和预测。",
            "title": "深度学习"
        },
        {
            "id": 4,
            "text": "搜索引擎帮助用户快速找到所需信息。搜索引擎优化可以提高网站在搜索结果中的排名。",
            "title": "搜索引擎"
        }
    ]

    engine.index(documents)

    print("\n[1] Searching for '机器学习'...")
    results = engine.search("机器学习")
    print(f"    Found {results['total_hits']} results")

    for r in results['results']:
        print(f"    - {r['document'].get('title')} (score: {r['score']:.4f})")

    print("\n[2] Searching for '自然语言处理'...")
    results = engine.search("自然语言处理")
    print(f"    Found {results['total_hits']} results")

    for r in results['results']:
        print(f"    - {r['document'].get('title')} (score: {r['score']:.4f})")


def example_algorithms():
    """Compare different ranking algorithms."""
    print("\n" + "=" * 60)
    print("Example 3: Algorithm Comparison")
    print("=" * 60)

    engine = SearchEngine()

    documents = [
        {"id": 1, "text": "Python Python Python programming language", "title": "Doc 1"},
        {"id": 2, "text": "Python is a programming language", "title": "Doc 2"},
        {"id": 3, "text": "Java programming language", "title": "Doc 3"}
    ]

    engine.index(documents)

    print("\n[1] BM25 Algorithm:")
    results = engine.search("Python", algorithm="bm25")
    for r in results['results']:
        print(f"    {r['document'].get('title')}: {r['score']:.4f}")

    print("\n[2] TF-IDF Algorithm:")
    results = engine.search("Python", algorithm="tfidf")
    for r in results['results']:
        print(f"    {r['document'].get('title')}: {r['score']:.4f}")


def example_boolean_search():
    """Boolean search example."""
    print("\n" + "=" * 60)
    print("Example 4: Boolean Search")
    print("=" * 60)

    engine = SearchEngine()

    documents = [
        {"id": 1, "text": "Python programming language", "title": "Doc 1"},
        {"id": 2, "text": "Java programming", "title": "Doc 2"},
        {"id": 3, "text": "Python and Java are languages", "title": "Doc 3"},
        {"id": 4, "text": "Machine learning with Python", "title": "Doc 4"}
    ]

    engine.index(documents)

    print("\n[1] AND search: 'Python Java'")
    results = engine.search_boolean("Python Java", operator="AND")
    print(f"    Found {results['total_hits']} results")
    for r in results['results']:
        print(f"    - {r['document'].get('title')}")

    print("\n[2] OR search: 'Python Java'")
    results = engine.search_boolean("Python Java", operator="OR")
    print(f"    Found {results['total_hits']} results")
    for r in results['results']:
        print(f"    - {r['document'].get('title')}")


def example_pagination():
    """Pagination example."""
    print("\n" + "=" * 60)
    print("Example 5: Pagination")
    print("=" * 60)

    engine = SearchEngine()

    # Create 20 sample documents
    documents = [
        {"id": i, "text": f"Document {i} about Python programming", "title": f"Doc {i}"}
        for i in range(1, 21)
    ]

    engine.index(documents)

    print("\n[1] Page 1 (limit=5):")
    results = engine.search("Python", limit=5)
    for r in results['results']:
        print(f"    - {r['document'].get('title')}")

    print("\n[2] Page 2 (offset=5, limit=5):")
    results = engine.search("Python", limit=5, offset=5)
    for r in results['results']:
        print(f"    - {r['document'].get('title')}")


def example_statistics():
    """Index statistics example."""
    print("\n" + "=" * 60)
    print("Example 6: Index Statistics")
    print("=" * 60)

    engine = SearchEngine()

    documents = [
        {"id": 1, "text": "Python programming", "title": "Doc 1"},
        {"id": 2, "text": "Java programming", "title": "Doc 2"},
        {"id": 3, "text": "Machine learning", "title": "Doc 3"}
    ]

    engine.index(documents)

    stats = engine.stats()
    print("\nIndex Statistics:")
    for key, value in stats.items():
        print(f"    {key}: {value}")


def example_performance():
    """Performance test example."""
    print("\n" + "=" * 60)
    print("Example 7: Performance Test")
    print("=" * 60)

    engine = SearchEngine()

    # Create 1000 documents
    num_docs = 1000
    print(f"\n[1] Creating {num_docs} sample documents...")
    documents = [
        {
            "id": i,
            "text": f"Document {i} about Python programming language machine learning artificial intelligence",
            "title": f"Document {i}"
        }
        for i in range(1, num_docs + 1)
    ]

    # Index
    print("[2] Indexing documents...")
    start = time.time()
    result = engine.index(documents)
    indexing_time = time.time() - start

    print(f"    Indexed {result['documents_indexed']} documents")
    print(f"    Time: {indexing_time:.2f}s")
    print(f"    Speed: {result['documents_per_second']:.2f} docs/sec")

    # Search
    print("\n[3] Performing searches...")
    queries = ["Python", "machine learning", "artificial intelligence", "programming"]

    for query in queries:
        start = time.time()
        results = engine.search(query)
        elapsed = (time.time() - start) * 1000

        print(f"    '{query}': {results['total_hits']} hits in {elapsed:.2f}ms")


def example_suggestions():
    """Query suggestions example."""
    print("\n" + "=" * 60)
    print("Example 8: Query Suggestions")
    print("=" * 60)

    engine = SearchEngine()

    documents = [
        {"id": 1, "text": "Python tutorial for beginners", "title": "Python Tutorial"},
        {"id": 2, "text": "Python advanced programming", "title": "Advanced Python"},
        {"id": 3, "text": "Python machine learning", "title": "ML with Python"},
        {"id": 4, "text": "JavaScript frontend development", "title": "JavaScript Guide"},
        {"id": 5, "text": "Java Spring framework", "title": "Spring Framework"}
    ]

    engine.index(documents)

    print("\nSuggestions for 'Py':")
    suggestions = engine.suggest("Py")
    print(f"    {suggestions}")

    print("\nSuggestions for 'Java':")
    suggestions = engine.suggest("Java")
    print(f"    {suggestions}")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("PySearch Example Scripts")
    print("=" * 60)

    example_basic_usage()
    example_chinese_search()
    example_algorithms()
    example_boolean_search()
    example_pagination()
    example_statistics()
    example_performance()
    example_suggestions()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
