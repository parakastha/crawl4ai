"""
Text chunking utilities for splitting large documents into manageable chunks.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer
from bs4 import BeautifulSoup

# Configure logging
logger = logging.getLogger(__name__)

try:
    # Try to import NLTK resources
    import nltk
    nltk.download('punkt', quiet=True)
except ImportError:
    logger.warning("NLTK not available. Some chunking methods may not work correctly.")
except Exception as e:
    logger.warning(f"Error initializing NLTK resources: {e}")


class BaseChunker:
    """Base class for text chunking strategies."""
    
    def __init__(self):
        """Initialize the chunker."""
        pass
    
    def chunk_text(self, text: str, **kwargs) -> List[str]:
        """Split text into chunks.
        
        Args:
            text: The text to chunk
            **kwargs: Additional parameters
            
        Returns:
            List of text chunks
        """
        raise NotImplementedError("Subclasses must implement the chunk_text method")
    
    def chunk_html(self, html: str, **kwargs) -> List[str]:
        """Split HTML content into chunks.
        
        Args:
            html: The HTML content to chunk
            **kwargs: Additional parameters
            
        Returns:
            List of HTML/text chunks
        """
        # Default implementation extracts text and chunks it
        try:
            soup = BeautifulSoup(html, 'html.parser')
            text = soup.get_text(separator=" ", strip=True)
            return self.chunk_text(text, **kwargs)
        except Exception as e:
            logger.error(f"Error chunking HTML: {e}")
            return [html]


class FixedSizeChunker(BaseChunker):
    """Chunker that splits text into fixed-size chunks."""
    
    def __init__(self, chunk_size: int = 4000, chunk_overlap: int = 200):
        """Initialize the fixed-size chunker.
        
        Args:
            chunk_size: Maximum number of characters per chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        super().__init__()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str, **kwargs) -> List[str]:
        """Split text into fixed-size chunks.
        
        Args:
            text: The text to chunk
            **kwargs: Additional parameters (ignored)
            
        Returns:
            List of text chunks
        """
        # Handle empty or None text
        if not text:
            return []
        
        # Get parameters with possible overrides
        chunk_size = kwargs.get("chunk_size", self.chunk_size)
        chunk_overlap = kwargs.get("chunk_overlap", self.chunk_overlap)
        
        # Ensure overlap is smaller than chunk size
        if chunk_overlap >= chunk_size:
            chunk_overlap = chunk_size // 2
        
        # Create chunks
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            # Calculate end index
            end = start + chunk_size
            
            # Adjust end to avoid cutting words
            if end < text_len:
                # Try to find a space to break at
                while end > start and not text[end].isspace():
                    end -= 1
                
                # If no space found, just cut at chunk_size
                if end <= start:
                    end = start + chunk_size
            else:
                end = text_len
            
            # Add chunk
            chunks.append(text[start:end])
            
            # Move start position for next chunk, considering overlap
            start = end - chunk_overlap
            
            # Ensure we progress if overlap would put us at same position
            if start <= 0 or start >= text_len:
                break
        
        return chunks


class SentenceChunker(BaseChunker):
    """Chunker that splits text by sentences while respecting max chunk size."""
    
    def __init__(self, max_chunk_size: int = 4000, min_chunk_size: int = 100):
        """Initialize the sentence chunker.
        
        Args:
            max_chunk_size: Maximum number of characters per chunk
            min_chunk_size: Minimum number of characters per chunk
        """
        super().__init__()
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        
        # Initialize tokenizer
        try:
            self.tokenizer = PunktSentenceTokenizer()
        except Exception as e:
            logger.warning(f"Error initializing sentence tokenizer: {e}")
            self.tokenizer = None
    
    def _get_sentences(self, text: str) -> List[str]:
        """Split text into sentences.
        
        Args:
            text: The text to split
            
        Returns:
            List of sentences
        """
        try:
            if self.tokenizer:
                return self.tokenizer.tokenize(text)
            else:
                return sent_tokenize(text)
        except Exception as e:
            logger.error(f"Error tokenizing sentences: {e}")
            # Fallback: split on periods with space
            return re.split(r'\.(?=\s)', text)
    
    def chunk_text(self, text: str, **kwargs) -> List[str]:
        """Split text into chunks based on sentences.
        
        Args:
            text: The text to chunk
            **kwargs: Additional parameters
            
        Returns:
            List of text chunks
        """
        # Handle empty or None text
        if not text:
            return []
        
        # Get parameters with possible overrides
        max_chunk_size = kwargs.get("max_chunk_size", self.max_chunk_size)
        min_chunk_size = kwargs.get("min_chunk_size", self.min_chunk_size)
        
        # Split text into sentences
        sentences = self._get_sentences(text)
        
        # Handle case of no sentences
        if not sentences:
            return [text] if text else []
        
        # Create chunks by grouping sentences
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed max_chunk_size
            if len(current_chunk) + len(sentence) > max_chunk_size and len(current_chunk) >= min_chunk_size:
                # Add current chunk to chunks and start a new one
                chunks.append(current_chunk)
                current_chunk = sentence
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks


class SemanticChunker(BaseChunker):
    """Chunker that tries to maintain semantic coherence in chunks."""
    
    def __init__(self, max_chunk_size: int = 4000, min_chunk_size: int = 100):
        """Initialize the semantic chunker.
        
        Args:
            max_chunk_size: Maximum number of characters per chunk
            min_chunk_size: Minimum number of characters per chunk
        """
        super().__init__()
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.sentence_chunker = SentenceChunker(max_chunk_size, min_chunk_size)
    
    def _find_section_boundaries(self, text: str) -> List[int]:
        """Find indices of section boundaries.
        
        Args:
            text: The text to analyze
            
        Returns:
            List of boundary indices
        """
        # Look for markdown-style headers
        header_matches = list(re.finditer(r'^#{1,6}\s+.+$', text, re.MULTILINE))
        
        # Look for repeated line breaks which might indicate section breaks
        newline_matches = list(re.finditer(r'\n\s*\n', text))
        
        # Combine and sort all potential boundary indices
        boundaries = []
        
        for match in header_matches:
            boundaries.append(match.start())
        
        for match in newline_matches:
            boundaries.append(match.start())
        
        # Sort and remove duplicates
        boundaries = sorted(set(boundaries))
        
        return boundaries
    
    def chunk_text(self, text: str, **kwargs) -> List[str]:
        """Split text into semantically coherent chunks.
        
        Args:
            text: The text to chunk
            **kwargs: Additional parameters
            
        Returns:
            List of text chunks
        """
        # Handle empty or None text
        if not text:
            return []
        
        # Get parameters with possible overrides
        max_chunk_size = kwargs.get("max_chunk_size", self.max_chunk_size)
        min_chunk_size = kwargs.get("min_chunk_size", self.min_chunk_size)
        
        # Find section boundaries
        boundaries = self._find_section_boundaries(text)
        
        # If no boundaries found, fall back to sentence chunking
        if not boundaries:
            return self.sentence_chunker.chunk_text(text, 
                                               max_chunk_size=max_chunk_size, 
                                               min_chunk_size=min_chunk_size)
        
        # Create chunks based on sections
        chunks = []
        start_idx = 0
        
        for idx in boundaries:
            # If current section is too big, chunk it further
            if idx - start_idx > max_chunk_size:
                section_text = text[start_idx:idx]
                section_chunks = self.sentence_chunker.chunk_text(section_text,
                                                            max_chunk_size=max_chunk_size,
                                                            min_chunk_size=min_chunk_size)
                chunks.extend(section_chunks)
            else:
                # Add section as a chunk if it's big enough
                section_text = text[start_idx:idx]
                if len(section_text) >= min_chunk_size:
                    chunks.append(section_text)
                elif chunks:  # If too small, append to previous chunk if possible
                    chunks[-1] += section_text
                else:  # If this is the first chunk, keep it despite size
                    chunks.append(section_text)
            
            # Update start index
            start_idx = idx
        
        # Handle the last section
        if start_idx < len(text):
            last_section = text[start_idx:]
            if len(last_section) > max_chunk_size:
                last_chunks = self.sentence_chunker.chunk_text(last_section,
                                                         max_chunk_size=max_chunk_size,
                                                         min_chunk_size=min_chunk_size)
                chunks.extend(last_chunks)
            elif len(last_section) >= min_chunk_size or not chunks:
                chunks.append(last_section)
            elif chunks:
                chunks[-1] += last_section
        
        return chunks
    
    def chunk_html(self, html: str, **kwargs) -> List[str]:
        """Split HTML content into chunks, trying to maintain HTML structure.
        
        Args:
            html: The HTML content to chunk
            **kwargs: Additional parameters
            
        Returns:
            List of HTML chunks
        """
        try:
            # Parse HTML
            soup = BeautifulSoup(html, 'html.parser')
            
            # Get parameters with possible overrides
            max_chunk_size = kwargs.get("max_chunk_size", self.max_chunk_size)
            min_chunk_size = kwargs.get("min_chunk_size", self.min_chunk_size)
            
            # Find natural chunk boundaries based on HTML structure
            chunks = []
            current_chunk = ""
            
            # Consider headers, divs, sections as chunk boundaries
            for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'div', 'section', 'article', 'p']):
                # Get HTML string for this element
                element_html = str(element)
                
                # If adding this element would exceed max_chunk_size
                if len(current_chunk) + len(element_html) > max_chunk_size and len(current_chunk) >= min_chunk_size:
                    # Add current chunk to chunks and start a new one
                    chunks.append(current_chunk)
                    current_chunk = element_html
                else:
                    # Add element to current chunk
                    current_chunk += element_html
            
            # Add the last chunk if it's not empty
            if current_chunk:
                chunks.append(current_chunk)
            
            # If no chunks were created, fall back to text chunking
            if not chunks:
                text = soup.get_text(separator=" ", strip=True)
                return self.chunk_text(text, **kwargs)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking HTML: {e}")
            # Fall back to text chunking
            return super().chunk_html(html, **kwargs)


def get_chunker(chunker_type: str = "semantic", **kwargs) -> BaseChunker:
    """Get a chunker by type.
    
    Args:
        chunker_type: Type of chunker ('fixed', 'sentence', or 'semantic')
        **kwargs: Additional parameters for the chunker
        
    Returns:
        A chunker instance
    """
    chunkers = {
        "fixed": FixedSizeChunker(**kwargs),
        "sentence": SentenceChunker(**kwargs),
        "semantic": SemanticChunker(**kwargs)
    }
    
    return chunkers.get(chunker_type.lower(), SemanticChunker(**kwargs)) 