"""
FastAPI Application for PySearch
================================

REST API for the high-performance full-text search engine.

Author: MiniMax Agent
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from .config import Config, APIConfig, BM25Config, IndexConfig, StorageConfig
from .storage import Storage
from .indexer import Indexer
from .query import QueryEngine, SearchResult


# Pydantic models for API
class Document(BaseModel):
    """Document model for indexing."""
    id: int = Field(..., description="Document unique identifier")
    text: str = Field(..., description="Document text content")
    title: Optional[str] = Field(None, description="Document title")
    content: Optional[str] = Field(None, description="Alternative content field")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class DocumentBatch(BaseModel):
    """Batch of documents for indexing."""
    documents: List[Document] = Field(..., description="List of documents to index")
    parallel: bool = Field(False, description="Enable parallel indexing")


class SearchRequest(BaseModel):
    """Search request model."""
    q: str = Field(..., description="Search query")
    algorithm: str = Field("bm25", description="Ranking algorithm: 'bm25' or 'tfidf'")
    limit: int = Field(10, ge=1, le=1000, description="Maximum results")
    offset: int = Field(0, ge=0, description="Result offset for pagination")
    highlight: bool = Field(True, description="Enable result highlighting")


class BooleanSearchRequest(BaseModel):
    """Boolean search request model."""
    q: str = Field(..., description="Search query")
    operator: str = Field("AND", description="Boolean operator: 'AND', 'OR', 'NOT'")
    limit: int = Field(10, ge=1, le=1000, description="Maximum results")


class IndexDocumentRequest(BaseModel):
    """Single document indexing request."""
    id: int = Field(..., description="Document unique identifier")
    text: str = Field(..., description="Document text")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class SearchResultItem(BaseModel):
    """Individual search result."""
    doc_id: int
    score: float
    document: Dict[str, Any]
    highlights: List[str]


class SearchResponse(BaseModel):
    """Search response model."""
    results: List[SearchResultItem]
    total_hits: int
    execution_time_ms: float
    query: str
    algorithm: str


class StatsResponse(BaseModel):
    """Index statistics response."""
    document_count: int
    term_count: int
    average_document_length: float
    total_terms: int
    index_size_mb: float


class SuggestResponse(BaseModel):
    """Query suggestions response."""
    suggestions: List[str]


# Global instances
_app: Optional[FastAPI] = None
_storage: Optional[Storage] = None
_indexer: Optional[Indexer] = None
_query_engine: Optional[QueryEngine] = None


def create_app(config: Optional[Config] = None) -> FastAPI:
    """
    Create and configure FastAPI application.

    Args:
        config: Optional configuration

    Returns:
        Configured FastAPI app
    """
    global _app, _storage, _indexer, _query_engine

    config = config or Config()

    # Initialize storage
    _storage = Storage(config.storage)

    # Initialize indexer
    _indexer = Indexer(_storage, config.index)

    # Initialize query engine
    _query_engine = QueryEngine(_storage, config.bm25, config.index)

    # Create FastAPI app
    _app = FastAPI(
        title="PySearch API",
        description="High-Performance Full-Text Search Engine API",
        version="1.0.0"
    )

    # CORS middleware
    _app.add_middleware(
        CORSMiddleware,
        allow_origins=config.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    _register_routes()

    return _app


def _register_routes() -> None:
    """Register API routes."""

    @_app.get("/")
    def root():
        """Root endpoint."""
        return {
            "name": "PySearch API",
            "version": "1.0.0",
            "description": "High-Performance Full-Text Search Engine"
        }

    @_app.get("/health")
    def health():
        """Health check endpoint."""
        return {"status": "healthy"}

    @_app.post("/index/batch", response_model=Dict[str, Any])
    async def index_batch(request: DocumentBatch = Body(...)):
        """Index a batch of documents."""
        try:
            # Convert to dict format
            documents = [doc.dict() for doc in request.documents]

            # Index documents
            stats = _indexer.index_documents(
                documents,
                parallel=request.parallel
            )

            return {
                "success": True,
                "documents_indexed": stats.documents_indexed,
                "terms_indexed": stats.terms_indexed,
                "indexing_time_ms": stats.indexing_time_ms,
                "documents_per_second": stats.documents_per_second
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @_app.post("/index/document")
    async def index_document(request: IndexDocumentRequest = Body(...)):
        """Index a single document."""
        try:
            success = _indexer.index_document(
                request.id,
                request.text,
                request.metadata
            )

            if success:
                return {"success": True, "doc_id": request.id}
            else:
                raise HTTPException(status_code=500, detail="Failed to index document")

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @_app.delete("/index/document/{doc_id}")
    async def delete_document(doc_id: int):
        """Delete a document from the index."""
        try:
            success = _indexer.remove_document(doc_id)
            if success:
                return {"success": True, "doc_id": doc_id}
            else:
                raise HTTPException(status_code=404, detail="Document not found")

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @_app.post("/search", response_model=SearchResponse)
    async def search(request: SearchRequest = Body(...)):
        """Execute a search query."""
        try:
            response = _query_engine.search(
                query=request.q,
                algorithm=request.algorithm,
                limit=request.limit,
                offset=request.offset,
                highlight=request.highlight
            )

            # Convert to response model
            results = [
                SearchResultItem(
                    doc_id=r.doc_id,
                    score=r.score,
                    document=r.document,
                    highlights=r.highlights
                )
                for r in response.results
            ]

            return SearchResponse(
                results=results,
                total_hits=response.total_hits,
                execution_time_ms=response.execution_time_ms,
                query=response.query,
                algorithm=response.algorithm
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @_app.get("/search", response_model=SearchResponse)
    async def search_get(
        q: str = Query(..., description="Search query"),
        algorithm: str = Query("bm25", description="Ranking algorithm"),
        limit: int = Query(10, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        highlight: bool = Query(True)
    ):
        """Execute a search query (GET method)."""
        try:
            response = _query_engine.search(
                query=q,
                algorithm=algorithm,
                limit=limit,
                offset=offset,
                highlight=highlight
            )

            results = [
                SearchResultItem(
                    doc_id=r.doc_id,
                    score=r.score,
                    document=r.document,
                    highlights=r.highlights
                )
                for r in response.results
            ]

            return SearchResponse(
                results=results,
                total_hits=response.total_hits,
                execution_time_ms=response.execution_time_ms,
                query=response.query,
                algorithm=response.algorithm
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @_app.post("/search/boolean", response_model=SearchResponse)
    async def search_boolean(request: BooleanSearchRequest = Body(...)):
        """Execute a boolean search query."""
        try:
            response = _query_engine.search_boolean(
                query=request.q,
                operator=request.operator,
                limit=request.limit
            )

            results = [
                SearchResultItem(
                    doc_id=r.doc_id,
                    score=r.score,
                    document=r.document,
                    highlights=r.highlights
                )
                for r in response.results
            ]

            return SearchResponse(
                results=results,
                total_hits=response.total_hits,
                execution_time_ms=response.execution_time_ms,
                query=response.query,
                algorithm=response.algorithm
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @_app.get("/suggest", response_model=SuggestResponse)
    async def suggest(
        q: str = Query(..., description="Query prefix"),
        limit: int = Query(10, ge=1, le=100)
    ):
        """Get query suggestions."""
        try:
            suggestions = _query_engine.get_suggestions(q, limit)
            return SuggestResponse(suggestions=suggestions)

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @_app.get("/stats", response_model=StatsResponse)
    async def get_stats():
        """Get index statistics."""
        try:
            storage_stats = _storage.get_stats()
            indexer_stats = _indexer.get_stats()

            return StatsResponse(
                document_count=storage_stats['document_count'],
                term_count=storage_stats['term_count'],
                average_document_length=storage_stats['average_document_length'],
                total_terms=storage_stats['total_terms'],
                index_size_mb=storage_stats['index_size_mb']
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @_app.delete("/index")
    async def clear_index():
        """Clear the entire index."""
        try:
            _storage.clear()
            _query_engine.clear_cache()

            return {"success": True, "message": "Index cleared"}

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @_app.post("/index/rebuild")
    async def rebuild_index():
        """Rebuild the entire index."""
        try:
            stats = _indexer.rebuild_index()

            return {
                "success": True,
                "documents_indexed": stats.documents_indexed,
                "terms_indexed": stats.terms_indexed,
                "indexing_time_ms": stats.indexing_time_ms
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    config: Optional[Config] = None,
    reload: bool = False
) -> None:
    """
    Run the FastAPI server.

    Args:
        host: Server host
        port: Server port
        config: Optional configuration
        reload: Enable auto-reload
    """
    config = config or Config()

    app = create_app(config)

    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
        log_level=config.api.log_level
    )
