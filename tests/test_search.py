"""
PySearch Unit Tests
===================

Tests for core search engine functionality including:
- Tokenization
- Indexing
- Search algorithms (BM25, TF-IDF)
- Query engine

Author: MiniMax Agent
"""

import pytest
import time
from typing import List, Dict, Any

# Import pysearch modules
import sys
sys.path.insert(0, '/workspace')

from pysearch.config import Config, BM25Config, TokenizerConfig
from pysearch.storage import Storage, InvertedIndex, DocumentStore
from pysearch.tokenizer import Tokenizer, EnglishTokenizer, ChineseTokenizer, MixedTokenizer
from pysearch.indexer import Indexer
from pysearch.query import QueryEngine, BM25Scorer, TFIDFScorer, SearchResult


class TestTokenizers:
    """Test tokenizer functionality."""

    def test_english_tokenizer(self):
        """Test English tokenization."""
        config = TokenizerConfig(enable_stemming=False, remove_stopwords=False)
        tokenizer = EnglishTokenizer(config)

        text = "Python is a high-level programming language"
        tokens = tokenizer.tokenize(text)

        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert "python" in tokens

    def test_english_tokenizer_lowercase(self):
        """Test English tokenizer with lowercase conversion."""
        config = TokenizerConfig(lowercase=True)
        tokenizer = EnglishTokenizer(config)

        text = "PYTHON PROGRAMMING"
        tokens = tokenizer.tokenize(text)

        assert all(t.islower() for t in tokens)

    def test_english_tokenizer_stopwords(self):
        """Test English tokenizer with stopword removal."""
        config = TokenizerConfig(remove_stopwords=True)
        tokenizer = EnglishTokenizer(config)

        text = "the quick brown fox"
        tokens = tokenizer.tokenize(text)

        assert "the" not in tokens
        assert "quick" in tokens
        assert "brown" in tokens
        assert "fox" in tokens

    def test_chinese_tokenizer(self):
        """Test Chinese tokenization."""
        config = TokenizerConfig()
        tokenizer = ChineseTokenizer(config)

        text = "自然语言处理是人工智能的重要分支"
        tokens = tokenizer.tokenize(text)

        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_mixed_tokenizer_english(self):
        """Test mixed tokenizer with English text."""
        config = TokenizerConfig()
        tokenizer = MixedTokenizer(config)

        text = "Python programming language"
        tokens = tokenizer.tokenize(text)

        assert "python" in tokens
        assert "programming" in tokens
        assert "language" in tokens

    def test_mixed_tokenizer_chinese(self):
        """Test mixed tokenizer with Chinese text."""
        config = TokenizerConfig()
        tokenizer = MixedTokenizer(config)

        text = "搜索引擎优化技术"
        tokens = tokenizer.tokenize(text)

        assert len(tokens) > 0

    def test_mixed_tokenizer_both(self):
        """Test mixed tokenizer with both Chinese and English."""
        config = TokenizerConfig()
        tokenizer = MixedTokenizer(config)

        text = "Python是很好的编程语言，机器学习是人工智能的分支"
        tokens = tokenizer.tokenize(text)

        assert "python" in tokens
        assert "机器学习" in tokens or "机器" in tokens


class TestInvertedIndex:
    """Test inverted index functionality."""

    def test_add_document(self):
        """Test adding documents to index."""
        index = InvertedIndex()

        terms = ["python", "programming", "language"]
        index.add_document(1, terms)

        assert index.doc_count == 1
        assert index.get_document_frequency("python") == 1

    def test_multiple_documents(self):
        """Test adding multiple documents."""
        index = InvertedIndex()

        index.add_document(1, ["python", "programming"])
        index.add_document(2, ["python", "language"])
        index.add_document(3, ["java", "programming"])

        assert index.doc_count == 3
        assert index.get_document_frequency("python") == 2
        assert index.get_document_frequency("programming") == 2
        assert index.get_document_frequency("java") == 1

    def test_term_frequency(self):
        """Test term frequency calculation."""
        index = InvertedIndex()

        terms = ["python", "python", "python", "programming"]
        index.add_document(1, terms)

        assert index.get_term_frequency("python", 1) == 3
        assert index.get_term_frequency("programming", 1) == 1

    def test_get_postings(self):
        """Test retrieving posting lists."""
        index = InvertedIndex()

        index.add_document(1, ["python", "programming"])
        index.add_document(2, ["python", "java"])

        postings = index.get_postings("python")
        assert 1 in postings
        assert 2 in postings


class TestStorage:
    """Test storage functionality."""

    def test_add_documents(self):
        """Test adding documents to storage."""
        storage = Storage()

        documents = [
            {"id": 1, "text": "Python is a programming language"},
            {"id": 2, "text": "Java is another programming language"}
        ]

        storage.add_documents(documents)

        assert storage.index.doc_count == 2

    def test_get_document(self):
        """Test retrieving a document."""
        storage = Storage()

        documents = [
            {"id": 1, "text": "Test document", "title": "Test Title"}
        ]

        storage.add_documents(documents)

        doc = storage.get_document(1)
        assert doc is not None
        assert doc["title"] == "Test Title"

    def test_get_stats(self):
        """Test getting index statistics."""
        storage = Storage()

        documents = [
            {"id": 1, "text": "Python programming"},
            {"id": 2, "text": "Java programming"}
        ]

        storage.add_documents(documents)

        stats = storage.get_stats()
        assert stats["document_count"] == 2


class TestBM25Scorer:
    """Test BM25 scoring algorithm."""

    def test_bm25_basic(self):
        """Test basic BM25 scoring."""
        # Create index
        index = InvertedIndex()
        index.add_document(1, ["python", "programming", "language"])
        index.add_document(2, ["python", "tutorial", "beginners"])
        index.add_document(3, ["java", "programming", "tutorial"])

        # Create scorer
        config = BM25Config()
        scorer = BM25Scorer(index, config)

        # Score query
        score = scorer.score(["python"], 1)

        assert score > 0

    def test_bm25_idf(self):
        """Test IDF calculation."""
        index = InvertedIndex()
        index.add_document(1, ["python"])
        index.add_document(2, ["python"])
        index.add_document(3, ["rare"])

        config = BM25Config()
        scorer = BM25Scorer(index, config)

        idf_python = scorer.get_idf("python")
        idf_rare = scorer.get_idf("rare")

        # Common term should have lower IDF than rare term
        assert idf_python < idf_rare


class TestTFIDFScorer:
    """Test TF-IDF scoring algorithm."""

    def test_tfidf_basic(self):
        """Test basic TF-IDF scoring."""
        index = InvertedIndex()
        index.add_document(1, ["python", "programming"])
        index.add_document(2, ["python", "python"])

        scorer = TFIDFScorer(index)

        # Document 2 has higher term frequency for "python"
        score1 = scorer.score(["python"], 1)
        score2 = scorer.score(["python"], 2)

        assert score2 > score1


class TestQueryEngine:
    """Test query engine functionality."""

    @pytest.fixture
    def engine(self):
        """Create query engine with sample data."""
        config = Config()
        storage = Storage()
        indexer = Indexer(storage, config.index)
        query_engine = QueryEngine(storage, config.bm25, config.index)

        # Add sample documents
        documents = [
            {
                "id": 1,
                "text": "Python is a high-level programming language. Python is widely used.",
                "title": "Python Programming"
            },
            {
                "id": 2,
                "text": "Java is a class-based programming language. Java runs on many platforms.",
                "title": "Java Programming"
            },
            {
                "id": 3,
                "text": "Search engines help find information on the web. Google is a popular search engine.",
                "title": "Search Engines"
            },
            {
                "id": 4,
                "text": "Machine learning is a subset of artificial intelligence. Deep learning uses neural networks.",
                "title": "Machine Learning"
            },
            {
                "id": 5,
                "text": "自然语言处理是人工智能的重要应用。NLP可以用于文本分析。",
                "title": "自然语言处理"
            }
        ]

        indexer.index_documents(documents)

        return query_engine

    def test_basic_search(self, engine):
        """Test basic search functionality."""
        results = engine.search("Python")

        assert results.total_hits >= 1
        assert len(results.results) > 0
        assert results.results[0].doc_id == 1

    def test_search_with_limit(self, engine):
        """Test search with result limit."""
        results = engine.search("programming", limit=2)

        assert len(results.results) <= 2

    def test_search_with_offset(self, engine):
        """Test search with pagination."""
        results1 = engine.search("programming", limit=1, offset=0)
        results2 = engine.search("programming", limit=1, offset=1)

        # Results should be different
        if results1.total_hits > 1:
            assert results1.results[0].doc_id != results2.results[0].doc_id

    def test_tfidf_algorithm(self, engine):
        """Test TF-IDF algorithm."""
        results = engine.search("Python", algorithm="tfidf")

        assert results.algorithm == "tfidf"
        assert len(results.results) > 0

    def test_chinese_search(self, engine):
        """Test Chinese text search."""
        results = engine.search("自然语言处理")

        assert results.total_hits >= 1

    def test_boolean_and(self, engine):
        """Test boolean AND search."""
        results = engine.search_boolean("Python programming", operator="AND")

        # Should only return documents with both terms
        for r in results.results:
            text = r.document.get("text", "").lower()
            assert "python" in text or "编程" in text

    def test_boolean_or(self, engine):
        """Test boolean OR search."""
        results = engine.search_boolean("Python Java", operator="OR")

        assert results.total_hits >= 2

    def test_empty_query(self, engine):
        """Test empty query handling."""
        results = engine.search("")

        assert results.total_hits == 0
        assert len(results.results) == 0


class TestIndexer:
    """Test indexer functionality."""

    def test_index_documents(self):
        """Test document indexing."""
        config = Config()
        storage = Storage()
        indexer = Indexer(storage, config.index)

        documents = [
            {"id": i, "text": f"Document {i} with some content"}
            for i in range(10)
        ]

        stats = indexer.index_documents(documents)

        assert stats.documents_indexed == 10

    def test_index_single_document(self):
        """Test indexing a single document."""
        config = Config()
        storage = Storage()
        indexer = Indexer(storage, config.index)

        success = indexer.index_document(1, "Test document content")

        assert success is True
        assert storage.index.doc_count == 1


class TestSearchEngine:
    """Test integrated search engine."""

    def test_end_to_end(self):
        """Test complete search workflow."""
        from pysearch.main import SearchEngine

        engine = SearchEngine()

        # Index documents
        documents = [
            {"id": 1, "text": "Python is a great programming language"},
            {"id": 2, "text": "Java is also popular"},
            {"id": 3, "text": "Search engines find information"}
        ]

        result = engine.index(documents)
        assert result["documents_indexed"] == 3

        # Search
        results = engine.search("Python")
        assert results["total_hits"] >= 1

    def test_stats(self):
        """Test index statistics."""
        from pysearch.main import SearchEngine

        engine = SearchEngine()

        documents = [
            {"id": 1, "text": "Test document one"},
            {"id": 2, "text": "Test document two"}
        ]

        engine.index(documents)
        stats = engine.stats()

        assert stats["document_count"] == 2


# Performance tests
class TestPerformance:
    """Performance tests for the search engine."""

    @pytest.mark.slow
    def test_indexing_performance(self):
        """Test indexing performance."""
        config = Config()
        storage = Storage()
        indexer = Indexer(storage, config.index)

        # Create large document set
        num_docs = 1000
        documents = [
            {
                "id": i,
                "text": f"Document {i} with some content about programming and technology"
            }
            for i in range(num_docs)
        ]

        start = time.time()
        stats = indexer.index_documents(documents)
        elapsed = time.time() - start

        print(f"\nIndexed {num_docs} documents in {elapsed:.2f}s")
        print(f"Speed: {stats.documents_per_second:.2f} docs/sec")

        assert stats.documents_indexed == num_docs

    @pytest.mark.slow
    def test_search_performance(self):
        """Test search performance."""
        config = Config()
        storage = Storage()
        indexer = Indexer(storage, config.index)
        query_engine = QueryEngine(storage, config.bm25, config.index)

        # Index documents
        num_docs = 1000
        documents = [
            {
                "id": i,
                "text": f"Document {i} content about Python programming and machine learning"
            }
            for i in range(num_docs)
        ]

        indexer.index_documents(documents)

        # Search
        start = time.time()
        results = query_engine.search("Python machine learning")
        elapsed = time.time() - start

        print(f"\nSearch completed in {elapsed * 1000:.2f}ms")
        print(f"Found {results.total_hits} results")

        # Performance assertion (should be under 100ms for 1000 docs)
        assert elapsed < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
