"""
Configuration Module for PySearch
=================================

Contains all configuration settings for the search engine,
including BM25 parameters, index settings, and API configurations.

Author: MiniMax Agent
"""

from typing import Dict, Any
from dataclasses import dataclass, field
import os


@dataclass
class BM25Config:
    """BM25 algorithm configuration parameters."""
    k1: float = 1.5  # Term frequency saturation parameter
    b: float = 0.75  # Document length normalization parameter
    epsilon: float = 0.25  # IDF smoothing parameter


@dataclass
class IndexConfig:
    """Index storage and performance configuration."""
    batch_size: int = 10000  # Documents per batch before flushing
    memory_limit_mb: int = 512  # Maximum memory for in-memory index
    enable_compression: bool = True  # Enable index compression
    cache_size: int = 1000  # LRU cache size for queries
    merge_threshold: int = 10  # Number of segments before merging
    auto_persist: bool = True  # Automatically persist after indexing operations
    persist_threshold: int = 1000  # Persist after N document changes


@dataclass
class TokenizerConfig:
    """Tokenizer configuration."""
    enable_chinese: bool = True  # Enable Chinese tokenization
    enable_english: bool = True  # Enable English tokenization
    lowercase: bool = True  # Convert to lowercase
    remove_stopwords: bool = True  # Remove stopwords
    enable_stemming: bool = False  # Enable English stemming
    min_word_length: int = 2  # Minimum word length
    max_word_length: int = 50  # Maximum word length


@dataclass
class APIConfig:
    """API server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = False
    log_level: str = "info"
    cors_origins: list = field(default_factory=lambda: ["*"])
    max_results: int = 1000  # Maximum search results
    default_results: int = 10  # Default number of results


@dataclass
class StorageConfig:
    """Storage configuration."""
    index_path: str = "./data/index"
    doc_store_path: str = "./data/docs"
    temp_path: str = "./data/temp"
    enable_persistence: bool = True


class Config:
    """Main configuration class aggregating all settings."""

    def __init__(self, **kwargs):
        self.bm25 = BM25Config()
        self.index = IndexConfig()
        self.tokenizer = TokenizerConfig()
        self.api = APIConfig()
        self.storage = StorageConfig()

        # Override with custom settings
        for key, value in kwargs.items():
            if hasattr(self, key):
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if hasattr(getattr(self, key), sub_key):
                            setattr(getattr(self, key), sub_key, sub_value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "bm25": self.bm25.__dict__,
            "index": self.index.__dict__,
            "tokenizer": self.tokenizer.__dict__,
            "api": self.api.__dict__,
            "storage": self.storage.__dict__
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create configuration from dictionary."""
        return cls(**config_dict)


# Global default configuration
default_config = Config()


# Stopwords lists
ENGLISH_STOPWORDS = frozenset([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
    'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
    'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
    'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're',
    've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't",
    'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't",
    'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn',
    "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren',
    "weren't", 'won', "won't", 'wouldn', "wouldn't"
])

CHINESE_STOPWORDS = frozenset([
    '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
    '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
    '自己', '这', '那', '里', '为', '什么', '可以', '他', '她', '它', '们', '但',
    '还', '这个', '那个', '这样', '那样', '如何', '怎么', '为什么', '哪', '哪个',
    '哪里', '谁', '多少', '几', '什么', '怎样', '怎么样', '因为', '所以', '但是',
    '然而', '如果', '虽然', '只是', '或者', '并且', '而且', '然后', '接着', '于是',
    '因此', '从而', '已经', '曾经', '正在', '将要', '可能', '应该', '必须', '需要',
    '能够', '可以', '会', '能', '要', '想', '愿', '肯', '敢', '务必', '一定'
])
