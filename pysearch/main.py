"""
Main Entry Point for PySearch
=============================

Provides both CLI interface and programmatic API for the search engine.

Author: MiniMax Agent
"""

import sys
import argparse
import json
from typing import List, Dict, Any, Optional, TYPE_CHECKING

from .config import Config, BM25Config, IndexConfig, TokenizerConfig, APIConfig, StorageConfig
from .storage import Storage
from .indexer import Indexer
from .query import QueryEngine

# Lazy import for optional dependencies
if TYPE_CHECKING:
    from .api import create_app, run_server


def _get_api_functions():
    """Lazy import for API functions."""
    try:
        from .api import create_app, run_server
        return create_app, run_server
    except ImportError:
        return None, None


class SearchEngine:
    """
    Main search engine class providing a simple programmatic interface.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.storage = Storage(self.config.storage)
        self.indexer = Indexer(self.storage, self.config.index)
        self.query_engine = QueryEngine(
            self.storage,
            self.config.bm25,
            self.config.index
        )

    def index(self, documents: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        Index documents.

        Args:
            documents: List of dicts with 'id', 'text' fields
            **kwargs: Additional arguments for indexer

        Returns:
            Indexing statistics
        """
        stats = self.indexer.index_documents(documents, **kwargs)
        return {
            'documents_indexed': stats.documents_indexed,
            'terms_indexed': stats.terms_indexed,
            'indexing_time_ms': stats.indexing_time_ms,
            'documents_per_second': stats.documents_per_second
        }

    def add(self, doc_id: int, text: str, **kwargs) -> bool:
        """
        Add a single document.

        Args:
            doc_id: Document ID
            text: Document text
            **kwargs: Additional metadata

        Returns:
            Success status
        """
        return self.indexer.index_document(doc_id, text, kwargs)

    def search(
        self,
        query: str,
        algorithm: str = "bm25",
        limit: int = 10,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Search documents.

        Args:
            query: Search query
            algorithm: Ranking algorithm
            limit: Max results
            offset: Result offset

        Returns:
            Search results
        """
        response = self.query_engine.search(
            query=query,
            algorithm=algorithm,
            limit=limit,
            offset=offset
        )

        return {
            'results': [
                {
                    'id': r.doc_id,
                    'score': r.score,
                    'document': r.document,
                    'highlights': r.highlights
                }
                for r in response.results
            ],
            'total_hits': response.total_hits,
            'execution_time_ms': response.execution_time_ms,
            'query': response.query,
            'algorithm': response.algorithm
        }

    def suggest(self, prefix: str, limit: int = 10) -> List[str]:
        """Get query suggestions."""
        return self.query_engine.get_suggestions(prefix, limit)

    def search_boolean(
        self,
        query: str,
        operator: str = "AND",
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Execute boolean search.

        Args:
            query: Search query
            operator: Boolean operator ('AND', 'OR', 'NOT')
            limit: Max results

        Returns:
            Search results
        """
        response = self.query_engine.search_boolean(
            query=query,
            operator=operator,
            limit=limit
        )

        return {
            'results': [
                {
                    'id': r.doc_id,
                    'score': r.score,
                    'document': r.document,
                    'highlights': r.highlights
                }
                for r in response.results
            ],
            'total_hits': response.total_hits,
            'execution_time_ms': response.execution_time_ms,
            'query': response.query,
            'algorithm': response.algorithm
        }

    def stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return self.storage.get_stats()

    def clear(self) -> None:
        """Clear the index."""
        self.storage.clear()
        self.query_engine.clear_cache()

    def save(self) -> bool:
        """Save index to disk."""
        return self.storage.save()

    def load(self) -> bool:
        """Load index from disk."""
        return self.storage.load()


def create_engine(config: Optional[Config] = None) -> SearchEngine:
    """
    Factory function to create a search engine instance.

    Args:
        config: Optional configuration

    Returns:
        SearchEngine instance
    """
    return SearchEngine(config)


def demo():
    """Run a simple demonstration."""
    print("PySearch Demo")
    print("=" * 50)

    # Create engine
    engine = SearchEngine()

    # Sample documents
    documents = [
        {
            'id': 1,
            'text': 'Python is a high-level programming language. Python is widely used in data science and machine learning.',
            'title': 'Python Programming'
        },
        {
            'id': 2,
            'text': 'Java is a class-based, object-oriented programming language. Java runs on billions of devices.',
            'title': 'Java Programming'
        },
        {
            'id': 3,
            'text': 'Search engines are software systems that search the World Wide Web. Google is the most popular search engine.',
            'title': 'Search Engines'
        },
        {
            'id': 4,
            'text': 'Machine learning is a subset of artificial intelligence. Deep learning is a subset of machine learning.',
            'title': 'Machine Learning'
        },
        {
            'id': 5,
            'text': '自然语言处理是人工智能的一个重要分支。它涉及计算机对自然语言的理解和生成。',
            'title': '自然语言处理'
        },
        {
            'id': 6,
            'text': '搜索引擎优化是一种提高网站在搜索引擎中排名的技术。SEO对于网站流量非常重要。',
            'title': '搜索引擎优化'
        }
    ]

    # Index documents
    print("\nIndexing documents...")
    result = engine.index(documents)
    print(f"Indexed {result['documents_indexed']} documents")
    print(f"Time: {result['indexing_time_ms']:.2f}ms")
    print(f"Speed: {result['documents_per_second']:.2f} docs/sec")

    # Search
    print("\n" + "=" * 50)
    print("Search: 'Python'")
    results = engine.search('Python')
    print(f"Found {results['total_hits']} results in {results['execution_time_ms']:.2f}ms")

    for r in results['results']:
        print(f"  - Doc {r['id']}: {r['document'].get('title', 'N/A')} (score: {r['score']:.4f})")

    print("\n" + "=" * 50)
    print("Search: 'search engine'")
    results = engine.search('search engine')
    print(f"Found {results['total_hits']} results in {results['execution_time_ms']:.2f}ms")

    for r in results['results']:
        print(f"  - Doc {r['id']}: {r['document'].get('title', 'N/A')} (score: {r['score']:.4f})")

    print("\n" + "=" * 50)
    print("Search: '机器学习' (Chinese)")
    results = engine.search('机器学习')
    print(f"Found {results['total_hits']} results in {results['execution_time_ms']:.2f}ms")

    for r in results['results']:
        print(f"  - Doc {r['id']}: {r['document'].get('title', 'N/A')} (score: {r['score']:.4f})")

    # Statistics
    print("\n" + "=" * 50)
    print("Index Statistics:")
    stats = engine.stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="PySearch - High-Performance Full-Text Search Engine"
    )

    parser.add_argument(
        'command',
        choices=['serve', 'demo', 'index', 'search'],
        help='Command to execute'
    )

    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='Server host (default: 0.0.0.0)'
    )

    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Server port (default: 8000)'
    )

    parser.add_argument(
        '--config',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--query',
        help='Search query (for search command)'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=10,
        help='Number of results (default: 10)'
    )

    args = parser.parse_args()

    if args.command == 'serve':
        # Run server (lazy import)
        create_app_fn, run_server_fn = _get_api_functions()
        if run_server_fn is None:
            print("Error: FastAPI is required for server mode.")
            print("Install it with: pip install fastapi uvicorn")
            sys.exit(1)
        print(f"Starting PySearch server on {args.host}:{args.port}")
        run_server_fn(args.host, args.port)

    elif args.command == 'demo':
        # Run demo
        demo()

    elif args.command == 'index':
        # Index documents from stdin
        print("Reading documents from stdin...")
        try:
            data = sys.stdin.read()
            documents = json.loads(data)

            engine = SearchEngine()
            result = engine.index(documents)

            print(json.dumps(result, indent=2))

        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            sys.exit(1)

    elif args.command == 'search':
        # Search
        if not args.query:
            print("Error: --query is required for search command")
            sys.exit(1)

        engine = SearchEngine()
        results = engine.search(args.query, limit=args.limit)

        print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
