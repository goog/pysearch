"""
Storage Module for PySearch
==========================

Provides persistent storage for inverted index and document store
using memory-mapped files and efficient serialization.

Author: MiniMax Agent
"""

import os
import json
import pickle
import struct
import mmap
import threading
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict
import numpy as np

from .config import StorageConfig


class InvertedIndex:
    """
    In-memory inverted index with optional disk persistence.

    Data Structure:
    - term -> {doc_id -> {'tf': term_frequency, 'positions': [pos1, pos2, ...]}}
    """

    def __init__(self):
        self.index: Dict[str, Dict[int, Dict[str, Any]]] = defaultdict(dict)
        self.doc_count = 0
        self.total_terms = 0
        self.doc_lengths: Dict[int, int] = {}
        self._lock = threading.RLock()

    def add_document(self, doc_id: int, terms: List[str]) -> None:
        """Add a document to the index."""
        with self._lock:
            # Track term positions
            term_positions: Dict[str, List[int]] = defaultdict(list)
            for pos, term in enumerate(terms):
                term_positions[term].append(pos)

            # Calculate document length (number of terms)
            self.doc_lengths[doc_id] = len(terms)

            # Update inverted index
            for term, positions in term_positions.items():
                if doc_id not in self.index[term]:
                    self.index[term][doc_id] = {'tf': 0, 'positions': []}

                self.index[term][doc_id]['tf'] += len(positions)
                self.index[term][doc_id]['positions'].extend(positions)

            self.doc_count += 1
            self.total_terms += len(terms)

    def get_postings(self, term: str) -> Dict[int, Dict[str, Any]]:
        """Get posting list for a term."""
        return self.index.get(term, {})

    def get_term_frequency(self, term: str, doc_id: int) -> int:
        """Get term frequency in a specific document."""
        return self.index.get(term, {}).get(doc_id, {}).get('tf', 0)

    def get_document_frequency(self, term: str) -> int:
        """Get document frequency (number of documents containing the term)."""
        return len(self.index.get(term, {}))

    def get_average_document_length(self) -> float:
        """Calculate average document length."""
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def get_document_length(self, doc_id: int) -> int:
        """Get length of a specific document."""
        return self.doc_lengths.get(doc_id, 0)

    def get_all_terms(self) -> List[str]:
        """Get all indexed terms."""
        return list(self.index.keys())

    def merge(self, other: 'InvertedIndex') -> None:
        """Merge another inverted index into this one."""
        with self._lock:
            for term, postings in other.index.items():
                for doc_id, data in postings.items():
                    if doc_id not in self.index[term]:
                        self.index[term][doc_id] = {'tf': 0, 'positions': []}
                    self.index[term][doc_id]['tf'] += data['tf']
                    self.index[term][doc_id]['positions'].extend(data['positions'])

            self.doc_count += other.doc_count
            self.total_terms += other.total_terms
            self.doc_lengths.update(other.doc_lengths)

    def clear(self) -> None:
        """Clear the index."""
        with self._lock:
            self.index.clear()
            self.doc_count = 0
            self.total_terms = 0
            self.doc_lengths.clear()


class DocumentStore:
    """
    Store for original documents with efficient retrieval.
    """

    def __init__(self):
        self.documents: Dict[int, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    def add_document(self, doc_id: int, document: Dict[str, Any]) -> None:
        """Add a document to the store."""
        with self._lock:
            self.documents[doc_id] = document

    def get_document(self, doc_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a document by ID."""
        return self.documents.get(doc_id)

    def get_documents(self, doc_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """Retrieve multiple documents by IDs."""
        return {doc_id: self.documents[doc_id] for doc_id in doc_ids if doc_id in self.documents}

    def delete_document(self, doc_id: int) -> bool:
        """Delete a document by ID."""
        with self._lock:
            if doc_id in self.documents:
                del self.documents[doc_id]
                return True
            return False

    def count(self) -> int:
        """Get total number of documents."""
        return len(self.documents)

    def clear(self) -> None:
        """Clear all documents."""
        with self._lock:
            self.documents.clear()


class Storage:
    """
    Main storage manager handling both inverted index and document store
    with persistence capabilities.
    """

    def __init__(self, config: Optional[StorageConfig] = None):
        self.config = config or StorageConfig()
        self.index = InvertedIndex()
        self.doc_store = DocumentStore()

        # Create directories if they don't exist
        self._ensure_directories()

        # Metadata
        self._metadata = {
            'doc_count': 0,
            'term_count': 0,
            'created_at': None,
            'updated_at': None
        }

    def _ensure_directories(self) -> None:
        """Create storage directories if they don't exist."""
        for path_attr in ['index_path', 'doc_store_path', 'temp_path']:
            path = getattr(self.config, path_attr)
            if path:
                Path(path).mkdir(parents=True, exist_ok=True)

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add multiple documents to the index and store.

        Args:
            documents: List of dicts with 'id', 'text', and optional 'title', 'content' fields
        """
        from .tokenizer import Tokenizer

        tokenizer = Tokenizer()

        for doc in documents:
            doc_id = doc.get('id', doc.get('_id'))
            if doc_id is None:
                continue

            # Get text content
            text = doc.get('text', doc.get('content', ''))
            if not text:
                continue

            # Tokenize
            terms = tokenizer.tokenize(text)

            # Add to index
            self.index.add_document(doc_id, terms)

            # Add to document store
            self.doc_store.add_document(doc_id, doc)

        # Update metadata
        self._metadata['doc_count'] = self.index.doc_count
        self._metadata['term_count'] = len(self.index.get_all_terms())
        self._metadata['updated_at'] = self._get_timestamp()

    def get_document(self, doc_id: int) -> Optional[Dict[str, Any]]:
        """Get a document by ID."""
        return self.doc_store.get_document(doc_id)

    def get_documents(self, doc_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """Get multiple documents by IDs."""
        return self.doc_store.get_documents(doc_ids)

    def search_by_id(self, doc_id: int) -> Optional[Dict[str, Any]]:
        """Search document by ID (exact match)."""
        return self.doc_store.get_document(doc_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            'document_count': self.index.doc_count,
            'term_count': len(self.index.get_all_terms()),
            'average_document_length': self.index.get_average_document_length(),
            'total_terms': self.index.total_terms,
            'index_size_mb': self._estimate_index_size()
        }

    def _estimate_index_size(self) -> float:
        """Estimate index size in MB."""
        import sys
        size = sys.getsizeof(pickle.dumps(self.index.index))
        size += sys.getsizeof(pickle.dumps(self.doc_store.documents))
        return size / (1024 * 1024)

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()

    def save(self) -> bool:
        """Persist index to disk."""
        try:
            # Save inverted index
            index_file = os.path.join(self.config.index_path, 'index.pkl')
            with open(index_file, 'wb') as f:
                pickle.dump({
                    'index': dict(self.index.index),
                    'doc_lengths': dict(self.index.doc_lengths),
                    'doc_count': self.index.doc_count,
                    'total_terms': self.index.total_terms
                }, f)

            # Save document store
            docs_file = os.path.join(self.config.doc_store_path, 'docs.pkl')
            with open(docs_file, 'wb') as f:
                pickle.dump(dict(self.doc_store.documents), f)

            # Save metadata
            meta_file = os.path.join(self.config.index_path, 'metadata.json')
            with open(meta_file, 'w') as f:
                json.dump(self._metadata, f, indent=2)

            return True
        except Exception as e:
            print(f"Error saving index: {e}")
            return False

    def load(self) -> bool:
        """Load index from disk."""
        try:
            # Load inverted index
            index_file = os.path.join(self.config.index_path, 'index.pkl')
            if os.path.exists(index_file):
                with open(index_file, 'rb') as f:
                    data = pickle.load(f)
                    self.index.index = defaultdict(dict, data['index'])
                    self.index.doc_lengths = data['doc_lengths']
                    self.index.doc_count = data['doc_count']
                    self.index.total_terms = data['total_terms']

            # Load document store
            docs_file = os.path.join(self.config.doc_store_path, 'docs.pkl')
            if os.path.exists(docs_file):
                with open(docs_file, 'rb') as f:
                    self.doc_store.documents = pickle.load(f)

            # Load metadata
            meta_file = os.path.join(self.config.index_path, 'metadata.json')
            if os.path.exists(meta_file):
                with open(meta_file, 'r') as f:
                    self._metadata = json.load(f)

            return True
        except Exception as e:
            print(f"Error loading index: {e}")
            return False

    def clear(self) -> None:
        """Clear all data."""
        self.index.clear()
        self.doc_store.clear()
        self._metadata = {
            'doc_count': 0,
            'term_count': 0,
            'created_at': None,
            'updated_at': None
        }
