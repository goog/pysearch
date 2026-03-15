"""
Query Engine Module for PySearch
================================

Provides search functionality with multiple ranking algorithms:
- BM25 (Okapi Best Matching 25)
- TF-IDF (Term Frequency-Inverse Document Frequency)

Author: MiniMax Agent
"""

import math
import time
import threading
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
from functools import lru_cache
import heapq

from .config import BM25Config, IndexConfig
from .tokenizer import Tokenizer
from .storage import Storage, InvertedIndex


@dataclass
class SearchResult:
    """Represents a single search result."""
    doc_id: int
    score: float
    document: Dict[str, Any] = field(default_factory=dict)
    highlights: List[str] = field(default_factory=list)
    terms: List[str] = field(default_factory=list)


@dataclass
class SearchResponse:
    """Response object for search queries."""
    results: List[SearchResult]
    total_hits: int
    execution_time_ms: float
    query: str
    algorithm: str
    facets: Dict[str, Any] = field(default_factory=dict)


class BM25Scorer:
    """
    BM25 (Okapi Best Matching 25) ranking algorithm.

    BM25 is a probabilistic ranking function used for information retrieval.
    It ranks documents based on the query terms appearing in each document,
    regardless of their proximity within the document.
    """

    def __init__(self, index: InvertedIndex, config: Optional[BM25Config] = None):
        self.index = index
        self.config = config or BM25Config()

        # Pre-compute IDF for all terms
        self._idf_cache: Dict[str, float] = {}
        self._compute_idf()

    def _compute_idf(self) -> None:
        """Pre-compute IDF values for all indexed terms."""
        N = self.index.doc_count
        if N == 0:
            return

        for term in self.index.get_all_terms():
            df = self.index.get_document_frequency(term)
            if df > 0:
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
                self._idf_cache[term] = idf

    def refresh_idf(self) -> None:
        """Refresh IDF values after index updates."""
        self._idf_cache.clear()
        self._compute_idf()

    def get_idf(self, term: str) -> float:
        """Get IDF value for a term."""
        return self._idf_cache.get(term, 0)

    def score(self, query_terms: List[str], doc_id: int) -> float:
        """
        Calculate BM25 score for a document given query terms.

        Formula:
        score(Q, D) = sum over q in Q of:
            IDF(q) * (f(q, D) * (k1 + 1)) /
            (f(q, D) + k1 * (1 - b + b * |D| / avgdl))

        Where:
        - f(q, D) = term frequency of q in document D
        - |D| = document length
        - avgdl = average document length
        - k1 and b = parameters
        """
        score = 0.0
        doc_length = self.index.get_document_length(doc_id)
        avgdl = self.index.get_average_document_length()

        if avgdl == 0:
            avgdl = 1.0

        for term in query_terms:
            tf = self.index.get_term_frequency(term, doc_id)
            if tf == 0:
                continue

            idf = self.get_idf(term)

            # BM25 formula
            numerator = tf * (self.config.k1 + 1)
            denominator = tf + self.config.k1 * (
                1 - self.config.b + self.config.b * doc_length / avgdl
            )

            score += idf * numerator / denominator

        return score

    def score_batch(
        self,
        query_terms: List[str],
        doc_ids: Optional[List[int]] = None
    ) -> Dict[int, float]:
        """
        Calculate BM25 scores for multiple documents.

        Args:
            query_terms: List of query terms
            doc_ids: Optional list of document IDs to score (if None, score all)

        Returns:
            Dictionary mapping doc_id to score
        """
        if not query_terms:
            return {}

        # If no doc_ids provided, get all documents containing any query term
        if doc_ids is None:
            doc_ids = set()
            for term in query_terms:
                postings = self.index.get_postings(term)
                doc_ids.update(postings.keys())
            doc_ids = list(doc_ids)

        # Score each document
        scores = {}
        for doc_id in doc_ids:
            score = self.score(query_terms, doc_id)
            if score > 0:
                scores[doc_id] = score

        return scores


class TFIDFScorer:
    """
    TF-IDF (Term Frequency-Inverse Document Frequency) ranking algorithm.

    Classic vector space model scoring approach.
    """

    def __init__(self, index: InvertedIndex):
        self.index = index

        # Pre-compute IDF values
        self._idf_cache: Dict[str, float] = {}
        self._compute_idf()

    def _compute_idf(self) -> None:
        """Pre-compute IDF values for all indexed terms."""
        N = self.index.doc_count
        if N == 0:
            return

        for term in self.index.get_all_terms():
            df = self.index.get_document_frequency(term)
            if df > 0:
                # Standard IDF with smoothing
                idf = math.log(N / df) + 1
                self._idf_cache[term] = idf

    def refresh_idf(self) -> None:
        """Refresh IDF values after index updates."""
        self._idf_cache.clear()
        self._compute_idf()

    def get_idf(self, term: str) -> float:
        """Get IDF value for a term."""
        return self._idf_cache.get(term, 0)

    def score(self, query_terms: List[str], doc_id: int) -> float:
        """
        Calculate TF-IDF score for a document.

        Formula:
        score(Q, D) = sum over q in Q of:
            (1 + log(tf(q, D))) * IDF(q)
        """
        score = 0.0

        for term in query_terms:
            tf = self.index.get_term_frequency(term, doc_id)
            if tf == 0:
                continue

            idf = self.get_idf(term)

            # TF with logarithmic scaling
            tf_score = 1 + math.log(tf)

            score += tf_score * idf

        return score

    def score_batch(
        self,
        query_terms: List[str],
        doc_ids: Optional[List[int]] = None
    ) -> Dict[int, float]:
        """Calculate TF-IDF scores for multiple documents."""
        if not query_terms:
            return {}

        # Get documents containing any query term
        if doc_ids is None:
            doc_ids = set()
            for term in query_terms:
                postings = self.index.get_postings(term)
                doc_ids.update(postings.keys())
            doc_ids = list(doc_ids)

        scores = {}
        for doc_id in doc_ids:
            score = self.score(query_terms, doc_id)
            if score > 0:
                scores[doc_id] = score

        return scores


class QueryEngine:
    """
    Main query engine supporting multiple ranking algorithms
    and advanced search features.
    """

    def __init__(
        self,
        storage: Storage,
        bm25_config: Optional[BM25Config] = None,
        index_config: Optional[IndexConfig] = None
    ):
        self.storage = storage
        self.index = storage.index
        self.bm25_config = bm25_config or BM25Config()
        self.index_config = index_config or IndexConfig()

        # Initialize scorers
        self.bm25_scorer = BM25Scorer(self.index, self.bm25_config)
        self.tfidf_scorer = TFIDFScorer(self.index)

        # Tokenizer
        self.tokenizer = Tokenizer()

        # Query cache
        self._cache: Dict[str, SearchResponse] = {}
        self._cache_lock = threading.RLock()
        self._max_cache_size = self.index_config.cache_size

    def search(
        self,
        query: str,
        algorithm: str = "bm25",
        limit: int = 10,
        offset: int = 0,
        use_cache: bool = True,
        highlight: bool = True
    ) -> SearchResponse:
        """
        Execute a search query.

        Args:
            query: Search query string
            algorithm: 'bm25' or 'tfidf'
            limit: Maximum number of results
            offset: Result offset for pagination
            use_cache: Whether to use query cache
            highlight: Whether to highlight matched terms

        Returns:
            SearchResponse with results and metadata
        """
        start_time = time.time()

        # Refresh IDF in case index was updated
        self.bm25_scorer.refresh_idf()
        self.tfidf_scorer.refresh_idf()

        # Check cache
        cache_key = f"{query}:{algorithm}:{limit}:{offset}"
        if use_cache:
            with self._cache_lock:
                if cache_key in self._cache:
                    return self._cache[cache_key]

        # Parse query
        query_terms = self.tokenizer.tokenize(query)
        if not query_terms:
            return SearchResponse(
                results=[],
                total_hits=0,
                execution_time_ms=0,
                query=query,
                algorithm=algorithm
            )

        # Get scores based on algorithm
        if algorithm.lower() == "bm25":
            scores = self.bm25_scorer.score_batch(query_terms)
        else:
            scores = self.tfidf_scorer.score_batch(query_terms)

        # Sort by score (descending)
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Apply pagination
        total_hits = len(sorted_docs)
        paginated_docs = sorted_docs[offset:offset + limit]

        # Build results
        results = []
        doc_ids = [doc_id for doc_id, _ in paginated_docs]
        documents = self.storage.get_documents(doc_ids)

        for doc_id, score in paginated_docs:
            doc = documents.get(doc_id, {})
            highlights = []

            if highlight:
                # Generate simple highlights
                text = doc.get('text', doc.get('content', ''))
                highlights = self._generate_highlights(text, query_terms)

            results.append(SearchResult(
                doc_id=doc_id,
                score=score,
                document=doc,
                highlights=highlights,
                terms=query_terms
            ))

        # Build response
        execution_time = (time.time() - start_time) * 1000

        response = SearchResponse(
            results=results,
            total_hits=total_hits,
            execution_time_ms=execution_time,
            query=query,
            algorithm=algorithm,
            facets=self._compute_facets(results)
        )

        # Cache result
        if use_cache:
            with self._cache_lock:
                if len(self._cache) >= self._max_cache_size:
                    # Simple cache eviction (remove first item)
                    first_key = next(iter(self._cache))
                    del self._cache[first_key]
                self._cache[cache_key] = response

        return response

    def search_boolean(
        self,
        query: str,
        operator: str = "AND",
        limit: int = 10
    ) -> SearchResponse:
        """
        Execute boolean search (AND, OR, NOT).

        Args:
            query: Search query
            operator: Boolean operator ('AND', 'OR', 'NOT')
            limit: Maximum results

        Returns:
            SearchResponse
        """
        start_time = time.time()

        # Refresh IDF in case index was updated
        self.bm25_scorer.refresh_idf()
        self.tfidf_scorer.refresh_idf()

        query_terms = self.tokenizer.tokenize(query)

        if not query_terms:
            return SearchResponse(
                results=[],
                total_hits=0,
                execution_time_ms=0,
                query=query,
                algorithm="boolean"
            )

        # Get posting lists for each term
        posting_lists = []
        for term in query_terms:
            postings = self.index.get_postings(term)
            posting_lists.append(set(postings.keys()))

        # Apply boolean operator
        if operator.upper() == "AND":
            # Intersection
            if not posting_lists:
                result_docs = set()
            else:
                result_docs = posting_lists[0]
                for postings in posting_lists[1:]:
                    result_docs &= postings

        elif operator.upper() == "OR":
            # Union
            result_docs = set()
            for postings in posting_lists:
                result_docs |= postings

        elif operator.upper() == "NOT":
            # Difference - documents NOT containing any query terms
            all_docs = set(self.index.doc_lengths.keys())
            if posting_lists:
                result_docs = all_docs - posting_lists[0]
            else:
                result_docs = all_docs

        else:
            result_docs = set()

        # Score using BM25
        scores = self.bm25_scorer.score_batch(query_terms, list(result_docs))

        # Sort by score
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        sorted_docs = sorted_docs[:limit]

        # Build results
        results = []
        doc_ids = [doc_id for doc_id, _ in sorted_docs]
        documents = self.storage.get_documents(doc_ids)

        for doc_id, score in sorted_docs:
            doc = documents.get(doc_id, {})
            results.append(SearchResult(
                doc_id=doc_id,
                score=score,
                document=doc,
                terms=query_terms
            ))

        execution_time = (time.time() - start_time) * 1000

        return SearchResponse(
            results=results,
            total_hits=len(result_docs),
            execution_time_ms=execution_time,
            query=query,
            algorithm=f"boolean_{operator}"
        )

    def _generate_highlights(self, text: str, terms: List[str]) -> List[str]:
        """Generate highlighted snippets from text."""
        if not text or not terms:
            return []

        highlights = []
        text_lower = text.lower()

        for term in terms:
            term_lower = term.lower()
            pos = text_lower.find(term_lower)

            if pos >= 0:
                # Extract snippet around the term
                start = max(0, pos - 30)
                end = min(len(text), pos + len(term) + 30)
                snippet = text[start:end]

                if start > 0:
                    snippet = "..." + snippet
                if end < len(text):
                    snippet = snippet + "..."

                # Highlight the term
                snippet = snippet.replace(
                    term,
                    f"<em>{term}</em>"
                )
                snippet = snippet.replace(
                    term.lower(),
                    f"<em>{term.lower()}</em>"
                )

                highlights.append(snippet)

        return highlights[:3]  # Limit to 3 highlights

    def _compute_facets(self, results: List[SearchResult]) -> Dict[str, Any]:
        """Compute facets from search results."""
        # Simple facet computation based on result fields
        facets = {
            'result_count': len(results)
        }

        return facets

    def get_suggestions(self, prefix: str, limit: int = 10) -> List[str]:
        """Get query suggestions based on prefix."""
        prefix_lower = prefix.lower()
        terms = self.index.get_all_terms()

        suggestions = [
            term for term in terms
            if term.lower().startswith(prefix_lower)
        ]

        return suggestions[:limit]

    def clear_cache(self) -> None:
        """Clear query cache."""
        with self._cache_lock:
            self._cache.clear()
