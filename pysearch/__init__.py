"""
PySearch - High-Performance Full-Text Search Engine
====================================================

A lightweight but scalable full-text search engine built with Python,
supporting both Chinese and English text, featuring BM25 ranking algorithm
and inverted index for efficient document retrieval.

Author: MiniMax Agent
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "MiniMax Agent"

from .tokenizer import Tokenizer, ChineseTokenizer, EnglishTokenizer
from .indexer import Indexer
from .query import QueryEngine, SearchResult
from .storage import Storage

__all__ = [
    "Tokenizer",
    "ChineseTokenizer",
    "EnglishTokenizer",
    "Indexer",
    "QueryEngine",
    "SearchResult",
    "Storage"
]
