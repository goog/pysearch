"""
Indexer Module for PySearch
===========================

Handles document indexing with batch processing, parallel execution,
and real-time updates.

Author: MiniMax Agent
"""

import time
import threading
import multiprocessing
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from queue import Queue
import math

from .config import IndexConfig
from .storage import Storage
from .tokenizer import Tokenizer


@dataclass
class IndexStats:
    """Statistics about the indexing process."""
    documents_indexed: int = 0
    terms_indexed: int = 0
    indexing_time_ms: float = 0
    documents_per_second: float = 0
    errors: int = 0


class Indexer:
    """
    High-performance document indexer with batch processing
    and parallel execution support.
    """

    def __init__(
        self,
        storage: Storage,
        config: Optional[IndexConfig] = None,
        tokenizer: Optional[Tokenizer] = None
    ):
        self.storage = storage
        self.config = config or IndexConfig()
        self.tokenizer = tokenizer or Tokenizer()

        # Indexing state
        self._is_indexing = False
        self._indexing_lock = threading.RLock()

        # Batch queue
        self._batch_queue: Queue = Queue()
        self._batch_size = self.config.batch_size

        # Statistics
        self.stats = IndexStats()

        # Document-to-terms mapping for efficient deletion
        self._doc_terms: Dict[int, set] = {}

        # Track changes for auto-persistence
        self._changes_since_persist = 0

    def index_documents(
        self,
        documents: List[Dict[str, Any]],
        parallel: bool = False,
        num_workers: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> IndexStats:
        """
        Index multiple documents.

        Args:
            documents: List of documents with 'id', 'text' fields
            parallel: Enable parallel indexing
            num_workers: Number of worker processes (default: CPU count)
            progress_callback: Optional callback for progress updates

        Returns:
            IndexStats with indexing statistics
        """
        with self._indexing_lock:
            self._is_indexing = True

        start_time = time.time()

        try:
            if parallel and len(documents) > self._batch_size:
                # Use parallel processing for large batches
                self._index_parallel(
                    documents,
                    num_workers=num_workers,
                    progress_callback=progress_callback
                )
            else:
                # Sequential indexing
                self._index_sequential(
                    documents,
                    progress_callback=progress_callback
                )

        finally:
            with self._indexing_lock:
                self._is_indexing = False

        # Calculate statistics
        elapsed = time.time() - start_time
        self.stats.indexing_time_ms = elapsed * 1000

        if elapsed > 0:
            self.stats.documents_per_second = self.stats.documents_indexed / elapsed

        # Auto-persist if configured
        self._auto_persist()

        return self.stats

    def _index_sequential(
        self,
        documents: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> None:
        """Index documents sequentially."""

        # Process in batches
        for i in range(0, len(documents), self._batch_size):
            batch = documents[i:i + self._batch_size]

            # Index batch
            self._index_batch(batch)

            # Update statistics
            self.stats.documents_indexed += len(batch)

            # Progress callback
            if progress_callback:
                progress_callback(
                    self.stats.documents_indexed,
                    len(documents)
                )

    def _index_parallel(
        self,
        documents: List[Dict[str, Any]],
        num_workers: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> None:
        """
        Index documents in parallel using multiple processes.

        Note: This creates temporary index segments that need to be merged.
        """
        num_workers = num_workers or multiprocessing.cpu_count()

        # Split documents into chunks
        chunk_size = max(1, len(documents) // num_workers)
        chunks = [
            documents[i:i + chunk_size]
            for i in range(0, len(documents), chunk_size)
        ]

        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []

            for chunk in chunks:
                future = executor.submit(self._index_batch, chunk)
                futures.append(future)

            # Collect results
            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                self.stats.documents_indexed += result

                if progress_callback:
                    progress_callback(
                        self.stats.documents_indexed,
                        len(documents)
                    )

    def _index_batch(self, documents: List[Dict[str, Any]]) -> int:
        """
        Index a batch of documents.

        Returns:
            Number of documents successfully indexed
        """
        indexed = 0

        for doc in documents:
            try:
                doc_id = doc.get('id', doc.get('_id'))
                if doc_id is None:
                    continue

                # Get text content
                text = doc.get('text', doc.get('content', ''))
                if not text:
                    continue

                # Tokenize
                terms = self.tokenizer.tokenize(text)

                # Add to index
                self.storage.index.add_document(doc_id, terms)

                # Add to document store
                self.storage.doc_store.add_document(doc_id, doc)

                # Track document terms for deletion
                self._doc_terms[doc_id] = set(terms)

                indexed += 1
                self.stats.terms_indexed += len(terms)
                self._changes_since_persist += 1

            except Exception as e:
                self.stats.errors += 1
                print(f"Error indexing document: {e}")

        return indexed

    def index_document(
        self,
        doc_id: int,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Index a single document in real-time.

        Args:
            doc_id: Document ID
            text: Document text content
            metadata: Optional metadata

        Returns:
            True if successful
        """
        try:
            # Tokenize
            terms = self.tokenizer.tokenize(text)

            # Create document
            document = {
                'id': doc_id,
                'text': text,
                **(metadata or {})
            }

            # Add to index
            self.storage.index.add_document(doc_id, terms)

            # Add to document store
            self.storage.doc_store.add_document(doc_id, document)

            # Track document terms for deletion
            self._doc_terms[doc_id] = set(terms)

            # Update statistics
            self.stats.documents_indexed += 1
            self.stats.terms_indexed += len(terms)
            self._changes_since_persist += 1

            # Auto-persist if threshold reached
            self._auto_persist()

            return True

        except Exception as e:
            print(f"Error indexing document: {e}")
            return False

    def remove_document(self, doc_id: int) -> bool:
        """
        Remove a document from the index completely.

        Args:
            doc_id: Document ID to remove

        Returns:
            True if successful
        """
        try:
            # Get the terms associated with this document
            if doc_id not in self._doc_terms:
                # Document not tracked, might need to scan or skip
                print(f"Warning: Document {doc_id} not in term tracking. Removing from store only.")
                self.storage.doc_store.delete_document(doc_id)
                return True

            terms = self._doc_terms[doc_id]

            # Remove document from inverted index for each term
            with self.storage.index._lock:
                for term in terms:
                    if term in self.storage.index.index:
                        postings = self.storage.index.index[term]
                        if doc_id in postings:
                            # Remove document from this term's posting list
                            del postings[doc_id]

                            # If no documents left for this term, remove term entirely
                            if not postings:
                                del self.storage.index.index[term]

                # Update document count and length tracking
                if doc_id in self.storage.index.doc_lengths:
                    doc_length = self.storage.index.doc_lengths[doc_id]
                    self.storage.index.total_terms -= doc_length
                    del self.storage.index.doc_lengths[doc_id]

                self.storage.index.doc_count = max(0, self.storage.index.doc_count - 1)

            # Remove from document store
            self.storage.doc_store.delete_document(doc_id)

            # Remove from term tracking
            del self._doc_terms[doc_id]

            # Track change
            self._changes_since_persist += 1

            # Auto-persist if threshold reached
            self._auto_persist()

            return True

        except Exception as e:
            print(f"Error removing document: {e}")
            return False

    def rebuild_index(self) -> IndexStats:
        """
        Rebuild the entire index from document store.

        Returns:
            IndexStats
        """
        # Clear current index
        self.storage.index.clear()

        # Clear document-term mapping
        self._doc_terms.clear()

        # Get all documents
        documents = []
        for doc_id, doc in self.storage.doc_store.documents.items():
            documents.append({
                'id': doc_id,
                'text': doc.get('text', doc.get('content', ''))
            })

        # Re-index
        self.stats = IndexStats()
        self._changes_since_persist = 0
        return self.index_documents(documents)

    def get_stats(self) -> Dict[str, Any]:
        """Get current indexing statistics."""
        return {
            'documents_indexed': self.stats.documents_indexed,
            'terms_indexed': self.stats.terms_indexed,
            'indexing_time_ms': self.stats.indexing_time_ms,
            'documents_per_second': self.stats.documents_per_second,
            'errors': self.stats.errors,
            'is_indexing': self._is_indexing,
            'changes_since_persist': self._changes_since_persist
        }

    def _auto_persist(self) -> bool:
        """
        Automatically persist index if configured and threshold reached.

        Returns:
            True if persisted, False otherwise
        """
        if not self.config.auto_persist:
            return False

        if self._changes_since_persist >= self.config.persist_threshold:
            return self.persist()

        return False

    def persist(self) -> bool:
        """
        Manually persist the index to disk.

        Returns:
            True if successful
        """
        try:
            # Save index and document store
            success = self.storage.save()

            if success:
                # Save document-term mapping
                import pickle
                import os
                mapping_file = os.path.join(
                    self.storage.config.index_path,
                    'doc_terms.pkl'
                )
                with open(mapping_file, 'wb') as f:
                    pickle.dump(self._doc_terms, f)

                # Reset change counter
                self._changes_since_persist = 0

            return success

        except Exception as e:
            print(f"Error persisting index: {e}")
            return False

    def load_from_disk(self) -> bool:
        """
        Load index from disk including document-term mappings.

        Returns:
            True if successful
        """
        try:
            # Load index and document store
            success = self.storage.load()

            if success:
                # Load document-term mapping
                import pickle
                import os
                mapping_file = os.path.join(
                    self.storage.config.index_path,
                    'doc_terms.pkl'
                )
                if os.path.exists(mapping_file):
                    with open(mapping_file, 'rb') as f:
                        self._doc_terms = pickle.load(f)
                else:
                    # Rebuild mapping from existing index
                    self._rebuild_doc_terms_mapping()

                # Reset change counter
                self._changes_since_persist = 0

            return success

        except Exception as e:
            print(f"Error loading index: {e}")
            return False

    def _rebuild_doc_terms_mapping(self) -> None:
        """
        Rebuild document-term mapping from the inverted index.
        This is useful when loading old indexes that don't have the mapping saved.
        """
        self._doc_terms.clear()

        # Iterate through all terms and their postings
        for term, postings in self.storage.index.index.items():
            for doc_id in postings.keys():
                if doc_id not in self._doc_terms:
                    self._doc_terms[doc_id] = set()
                self._doc_terms[doc_id].add(term)


class RealTimeIndexer(Indexer):
    """
    Real-time indexer that immediately indexes documents
    as they are added, with callback support for updates and deletions.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._update_callbacks: List[Callable[[int, str], None]] = []

    def add_update_callback(
        self,
        callback: Callable[[int, str], None]
    ) -> None:
        """
        Add callback for document updates.

        Args:
            callback: Function(doc_id, action) where action is 'added', 'updated', or 'removed'
        """
        self._update_callbacks.append(callback)

    def index_document(
        self,
        doc_id: int,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Index document and trigger callbacks."""
        # Check if document already exists
        is_update = doc_id in self._doc_terms

        result = super().index_document(doc_id, text, metadata)

        # Trigger callbacks
        if result:
            action = 'updated' if is_update else 'added'
            for callback in self._update_callbacks:
                try:
                    callback(doc_id, action)
                except Exception as e:
                    print(f"Error in update callback: {e}")

        return result

    def remove_document(self, doc_id: int) -> bool:
        """Remove document and trigger callbacks."""
        result = super().remove_document(doc_id)

        # Trigger callbacks
        if result:
            for callback in self._update_callbacks:
                try:
                    callback(doc_id, 'removed')
                except Exception as e:
                    print(f"Error in update callback: {e}")

        return result
