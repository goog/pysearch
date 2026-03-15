"""
Tokenizer Module for PySearch
=============================

Provides text tokenization support for both Chinese and English,
with stopword removal and stemming capabilities.

Author: MiniMax Agent
"""

import re
import math
from abc import ABC, abstractmethod
from typing import List, Set, Optional, Dict, Any
from dataclasses import dataclass

# Try to import optional dependencies
try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False

try:
    from nltk.stem import PorterStemmer
    from nltk.tokenize import word_tokenize
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

from .config import TokenizerConfig, ENGLISH_STOPWORDS, CHINESE_STOPWORDS


@dataclass
class Token:
    """Represents a token with its metadata."""
    text: str
    position: int
    is_chinese: bool = False
    is_english: bool = False
    original: str = ""


class BaseTokenizer(ABC):
    """Abstract base class for tokenizers."""

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Tokenize input text into list of tokens."""
        pass

    @abstractmethod
    def is_supported(self, text: str) -> bool:
        """Check if this tokenizer can handle the given text."""
        pass


class EnglishTokenizer(BaseTokenizer):
    """English text tokenizer with stopword removal and optional stemming."""

    def __init__(self, config: Optional[TokenizerConfig] = None):
        self.config = config or TokenizerConfig()
        self.stopwords: Set[str] = ENGLISH_STOPWORDS if self.config.remove_stopwords else set()
        self.stemmer = None

        if self.config.enable_stemming and NLTK_AVAILABLE:
            try:
                self.stemmer = PorterStemmer()
            except:
                pass

        # Regex patterns
        self.word_pattern = re.compile(r'[a-zA-Z]+')
        self.normalize_pattern = re.compile(r'[^a-zA-Z0-9\s]')

    def is_supported(self, text: str) -> bool:
        """Check if text contains English characters."""
        return bool(self.word_pattern.search(text))

    def tokenize(self, text: str) -> List[str]:
        """Tokenize English text."""
        if not text:
            return []

        # Extract English words
        words = self.word_pattern.findall(text)

        # Normalize case
        if self.config.lowercase:
            words = [w.lower() for w in words]

        # Filter by length
        words = [
            w for w in words
            if self.config.min_word_length <= len(w) <= self.config.max_word_length
        ]

        # Remove stopwords
        if self.stopwords:
            words = [w for w in words if w not in self.stopwords]

        # Apply stemming
        if self.stemmer:
            words = [self.stemmer.stem(w) for w in words]

        return words


class ChineseTokenizer(BaseTokenizer):
    """Chinese text tokenizer using jieba library."""

    def __init__(self, config: Optional[TokenizerConfig] = None):
        self.config = config or TokenizerConfig()
        self.stopwords: Set[str] = CHINESE_STOPWORDS if self.config.remove_stopwords else set()

        if not JIEBA_AVAILABLE:
            raise ImportError(
                "jieba is required for Chinese tokenization. "
                "Install it with: pip install jieba"
            )

        # Initialize jieba
        jieba.initialize()

    def is_supported(self, text: str) -> bool:
        """Check if text contains Chinese characters."""
        return bool(re.search(r'[\u4e00-\u9fff]', text))

    def tokenize(self, text: str) -> List[str]:
        """Tokenize Chinese text using jieba."""
        if not text or not self.is_supported(text):
            return []

        # Use jieba for segmentation
        words = jieba.cut_for_search(text)

        # Filter
        result = []
        for word in words:
            # Skip stopwords
            if word in self.stopwords:
                continue

            # Skip single characters (usually not meaningful)
            if len(word) < self.config.min_word_length:
                continue

            # Skip too long words
            if len(word) > self.config.max_word_length:
                continue

            # Skip pure numbers and punctuation
            if word.isdigit() or word.isalpha():
                if word.isdigit():
                    continue

            result.append(word)

        return result


class MixedTokenizer:
    """
    Mixed tokenizer that handles both Chinese and English text.
    Automatically detects language and applies appropriate tokenization.
    """

    def __init__(self, config: Optional[TokenizerConfig] = None):
        self.config = config or TokenizerConfig()

        self.english_tokenizer = EnglishTokenizer(config) if self.config.enable_english else None
        self.chinese_tokenizer = ChineseTokenizer(config) if self.config.enable_chinese else None

        # Regex patterns
        self.chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
        self.english_pattern = re.compile(r'[a-zA-Z]+')

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize mixed Chinese and English text.

        Strategy:
        1. First extract all English words and tokenize them
        2. Then extract Chinese segments and tokenize them
        3. Combine and deduplicate results
        """
        if not text:
            return []

        all_tokens = []

        # Extract and tokenize English
        if self.english_tokenizer:
            english_words = self.english_tokenizer.tokenize(text)
            all_tokens.extend(english_words)

        # Extract and tokenize Chinese
        if self.chinese_tokenizer:
            chinese_words = self.chinese_tokenizer.tokenize(text)
            all_tokens.extend(chinese_words)

        # Remove duplicates while preserving order
        seen = set()
        unique_tokens = []
        for token in all_tokens:
            if token not in seen:
                seen.add(token)
                unique_tokens.append(token)

        return unique_tokens

    def tokenize_with_positions(self, text: str) -> List[Token]:
        """
        Tokenize text and return tokens with position information.
        Useful for phrase searching and highlighting.
        """
        if not text:
            return []

        tokens = []
        position = 0

        # Split text into Chinese and non-Chinese segments
        # This is a simplified approach
        segments = self._split_segments(text)

        for segment in segments:
            if not segment:
                continue

            if self.chinese_pattern.search(segment) and self.chinese_tokenizer:
                # Chinese segment
                chinese_tokens = self.chinese_tokenizer.tokenize(segment)
                for token in chinese_tokens:
                    tokens.append(Token(
                        text=token,
                        position=position,
                        is_chinese=True,
                        original=segment
                    ))
                    position += 1
            else:
                # English segment
                if self.english_tokenizer:
                    english_tokens = self.english_tokenizer.tokenize(segment)
                    for token in english_tokens:
                        tokens.append(Token(
                            text=token,
                            position=position,
                            is_english=True,
                            original=segment
                        ))
                        position += 1

        return tokens

    def _split_segments(self, text: str) -> List[str]:
        """Split text into Chinese and non-Chinese segments."""
        segments = []
        current = ""

        for char in text:
            if self.chinese_pattern.search(char):
                if current:
                    segments.append(current)
                    current = ""
                segments.append(char)
            else:
                current += char

        if current:
            segments.append(current)

        return segments


class Tokenizer:
    """
    Main tokenizer class that provides a unified interface
    for tokenization operations.
    """

    def __init__(self, config: Optional[TokenizerConfig] = None):
        self.config = config or TokenizerConfig()
        self.mixed_tokenizer = MixedTokenizer(config)

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into list of terms."""
        return self.mixed_tokenizer.tokenize(text)

    def tokenize_with_positions(self, text: str) -> List[Token]:
        """Tokenize text with position information."""
        return self.mixed_tokenizer.tokenize_with_positions(text)

    def get_document_terms(self, documents: List[Dict[str, Any]]) -> Dict[int, List[str]]:
        """
        Tokenize multiple documents and return doc_id to terms mapping.

        Args:
            documents: List of dicts with 'id' and 'text' fields

        Returns:
            Dictionary mapping document ID to list of tokens
        """
        result = {}
        for doc in documents:
            doc_id = doc.get('id', doc.get('_id', 0))
            text = doc.get('text', doc.get('content', ''))
            result[doc_id] = self.tokenize(text)
        return result


# Factory function
def create_tokenizer(
    language: str = "mixed",
    config: Optional[TokenizerConfig] = None
) -> BaseTokenizer:
    """
    Factory function to create appropriate tokenizer.

    Args:
        language: 'english', 'chinese', or 'mixed'
        config: Tokenizer configuration

    Returns:
        Tokenizer instance
    """
    if language == "english":
        return EnglishTokenizer(config)
    elif language == "chinese":
        return ChineseTokenizer(config)
    else:
        return MixedTokenizer(config)
