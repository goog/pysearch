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

        # Save to disk if persistence is enabled
        if self.config.enable_compression:
            self.storage.save()

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

                indexed += 1
                self.stats.terms_indexed += len(terms)

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

            # Update statistics
            self.stats.documents_indexed += 1
            self.stats.terms_indexed += len(terms)

            return True

        except Exception as e:
            print(f"Error indexing document: {e}")
            return False

    def remove_document(self, doc_id: int) -> bool:
        """
        Remove a document from the index.

        Args:
            doc_id: Document ID to remove

        Returns:
            True if successful
        """
        try:
            # This is a simplified implementation
            # In a production system, you'd want to track which terms
            # are associated with each document for efficient removal

            # For now, we just remove from document store
            self.storage.doc_store.delete_document(doc_id)

            # Note: Full removal from inverted index requires
            # rebuilding the index or tracking document-term relationships

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

        # Get all documents
        documents = []
        for doc_id, doc in self.storage.doc_store.documents.items():
            documents.append({
                'id': doc_id,
                'text': doc.get('text', doc.get('content', ''))
            })

        # Re-index
        self.stats = IndexStats()
        return self.index_documents(documents)

    def get_stats(self) -> Dict[str, Any]:
        """Get current indexing statistics."""
        return {
            'documents_indexed': self.stats.documents_indexed,
            'terms_indexed': self.stats.terms_indexed,
            'indexing_time_ms': self.stats.indexing_time_ms,
            'documents_per_second': self.stats.documents_per_second,
            'errors': self.stats.errors,
            'is_indexing': self._is_indexing
        }


class RealTimeIndexer(Indexer):
    """
    Real-time indexer that immediately indexes documents
    as they are added.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._update_callbacks: List[Callable[[int, str], None]] = []

    def add_update_callback(
        self,
        callback: Callable[[int, str], None]
    ) -> None:
        """Add callback for document updates."""
        self._update_callbacks.append(callback)

    def index_document(
        self,
        doc_id: int,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Index document and trigger callbacks."""
        result = super().index_document(doc_id, text, metadata)

        # Trigger callbacks
        if result:
            for callback in self._update_callbacks:
                try:
                    callback(doc_id, 'added')
                except Exception as e:
                    print(f"Error in update callback: {e}")

        return result
