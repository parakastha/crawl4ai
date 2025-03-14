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
        
        # Look for HTML-style headers
        html_header_matches = list(re.finditer(r'<h[1-6][^>]*>.*?</h[1-6]>', text, re.DOTALL | re.IGNORECASE))
        
        # Look for other boundary markers
        boundary_matches = list(re.finditer(r'^[A-Z][^.!?]+:$', text, re.MULTILINE))  # Title with colon
        boundary_matches.extend(re.finditer(r'\n\s*\n', text))  # Double newline
        
        # Combine all matches
        all_matches = header_matches + html_header_matches + boundary_matches
        
        # Extract start indices, removing duplicates
        boundaries = sorted(set(m.start() for m in all_matches))
        
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
        
        # Find semantic boundaries
        boundaries = self._find_section_boundaries(text)
        
        # If no boundaries found, fall back to sentence chunking
        if not boundaries:
            return self.sentence_chunker.chunk_text(text, 
                                                    max_chunk_size=max_chunk_size,
                                                    min_chunk_size=min_chunk_size)
        
        # Add text start and end to boundaries
        if 0 not in boundaries:
            boundaries.insert(0, 0)
        if len(text) not in boundaries:
            boundaries.append(len(text))
        
        # Create initial chunks based on semantic boundaries
        raw_chunks = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            
            # Skip empty sections
            if end > start:
                raw_chunks.append(text[start:end])
        
        # Further process chunks that are too large
        final_chunks = []
        for raw_chunk in raw_chunks:
            if len(raw_chunk) <= max_chunk_size:
                final_chunks.append(raw_chunk)
            else:
                # If chunk is too large, use sentence chunking
                sub_chunks = self.sentence_chunker.chunk_text(raw_chunk,
                                                            max_chunk_size=max_chunk_size,
                                                            min_chunk_size=min_chunk_size)
                final_chunks.extend(sub_chunks)
        
        return final_chunks
    
    def chunk_html(self, html: str, **kwargs) -> List[str]:
        """Split HTML content into semantically coherent chunks.
        
        Args:
            html: The HTML content to chunk
            **kwargs: Additional parameters
            
        Returns:
            List of text chunks
        """
        # Try to find semantic boundaries in HTML
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find all header elements
            headers = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            
            # If we have headers, try to chunk by sections
            if headers:
                chunks = []
                current_section = ""
                
                # Function to get all content up to next header
                def get_next_elements(elem):
                    content = []
                    current = elem.next_sibling
                    while current and not (current.name and current.name.startswith('h') and len(current.name) == 2):
                        if hasattr(current, 'text'):
                            content.append(current.text.strip())
                        elif isinstance(current, str):
                            content.append(current.strip())
                        current = current.next_sibling
                    return ' '.join(filter(None, content))
                
                # Process each header and its content
                for header in headers:
                    # Extract header text
                    header_text = header.text.strip()
                    
                    # Get content following this header
                    section_content = get_next_elements(header)
                    
                    # Create a new section
                    current_section = f"{header_text}\n\n{section_content}".strip()
                    
                    # Add to chunks if not empty
                    if current_section:
                        chunks.append(current_section)
                
                # If we created chunks, return them - otherwise fall through to default method
                if chunks:
                    max_chunk_size = kwargs.get("max_chunk_size", self.max_chunk_size)
                    min_chunk_size = kwargs.get("min_chunk_size", self.min_chunk_size)
                    
                    # Further process chunks that are too large
                    final_chunks = []
                    for chunk in chunks:
                        if len(chunk) <= max_chunk_size:
                            final_chunks.append(chunk)
                        else:
                            # If chunk is too large, use sentence chunking
                            sub_chunks = self.sentence_chunker.chunk_text(chunk,
                                                                        max_chunk_size=max_chunk_size,
                                                                        min_chunk_size=min_chunk_size)
                            final_chunks.extend(sub_chunks)
                    
                    return final_chunks
        
        except Exception as e:
            logger.error(f"Error performing semantic HTML chunking: {e}")
        
        # Fall back to default method
        return super().chunk_html(html, **kwargs)


class RegexChunking(BaseChunker):
    """Chunker that splits text based on regex patterns."""
    
    def __init__(self, patterns: List[str] = None):
        """Initialize the regex chunker.
        
        Args:
            patterns: List of regex patterns for splitting text
                     Default: [r'\n\n'] (split on double newline)
        """
        super().__init__()
        self.patterns = patterns or [r'\n\n']  # Default pattern for paragraphs
    
    def chunk_text(self, text: str, **kwargs) -> List[str]:
        """Split text based on regex patterns.
        
        Args:
            text: The text to chunk
            **kwargs: Additional parameters
            
        Returns:
            List of text chunks
        """
        # Handle empty or None text
        if not text:
            return []
        
        # Get patterns from kwargs if provided
        patterns = kwargs.get("patterns", self.patterns)
        
        # Apply each pattern in sequence
        chunks = [text]
        for pattern in patterns:
            # Create a new list of chunks by splitting existing chunks
            new_chunks = []
            for chunk in chunks:
                # Split the chunk using the current pattern
                split_parts = re.split(pattern, chunk)
                # Add non-empty parts to new chunks
                new_chunks.extend([part.strip() for part in split_parts if part.strip()])
            # Update chunks for next iteration
            chunks = new_chunks
        
        return chunks


class SlidingWindowChunking(BaseChunker):
    """Chunker that creates chunks with a sliding window approach."""
    
    def __init__(self, window_size: int = 100, step: int = 50):
        """Initialize the sliding window chunker.
        
        Args:
            window_size: Number of words per window
            step: Number of words to slide the window
        """
        super().__init__()
        self.window_size = window_size
        self.step = step
    
    def chunk_text(self, text: str, **kwargs) -> List[str]:
        """Split text into chunks using a sliding window.
        
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
        window_size = kwargs.get("window_size", self.window_size)
        step = kwargs.get("step", self.step)
        
        # Tokenize the text into words
        try:
            words = word_tokenize(text)
        except Exception as e:
            logger.error(f"Error tokenizing words: {e}")
            # Simple fallback tokenization by splitting on whitespace
            words = text.split()
        
        # Handle case with fewer words than window size
        if len(words) <= window_size:
            return [text]
        
        # Create chunks using sliding window
        chunks = []
        for i in range(0, len(words) - window_size + 1, step):
            # Extract words for this window
            window_words = words[i:i + window_size]
            # Join words back into text
            chunk = ' '.join(window_words)
            chunks.append(chunk)
        
        return chunks


class OverlappingWindowChunking(BaseChunker):
    """Chunker that creates chunks with specified overlap."""
    
    def __init__(self, window_size: int = 500, overlap: int = 50):
        """Initialize the overlapping window chunker.
        
        Args:
            window_size: Number of words per chunk
            overlap: Number of words to overlap between chunks
        """
        super().__init__()
        self.window_size = window_size
        self.overlap = min(overlap, window_size - 1)  # Ensure overlap is less than window_size
    
    def chunk_text(self, text: str, **kwargs) -> List[str]:
        """Split text into chunks with specified overlap.
        
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
        window_size = kwargs.get("window_size", self.window_size)
        overlap = kwargs.get("overlap", self.overlap)
        overlap = min(overlap, window_size - 1)  # Ensure overlap is less than window_size
        
        # Calculate step size based on window size and overlap
        step = window_size - overlap
        
        # Tokenize the text into words
        try:
            words = word_tokenize(text)
        except Exception as e:
            logger.error(f"Error tokenizing words: {e}")
            # Simple fallback tokenization by splitting on whitespace
            words = text.split()
        
        # Handle case with fewer words than window size
        if len(words) <= window_size:
            return [text]
        
        # Create chunks with overlap
        chunks = []
        for i in range(0, len(words), step):
            # Extract words for this chunk
            chunk_words = words[i:i + window_size]
            # Skip small final chunks
            if len(chunk_words) < window_size * 0.5 and chunks:
                # Instead of creating a small final chunk, extend the last chunk
                last_chunk_words = words[i - step:i + len(chunk_words)]
                chunks[-1] = ' '.join(last_chunk_words)
                break
            # Join words back into text
            chunk = ' '.join(chunk_words)
            chunks.append(chunk)
        
        return chunks


def get_chunker(chunker_type: str = "semantic", **kwargs) -> BaseChunker:
    """Get a chunker by type.
    
    Args:
        chunker_type: Type of chunker ('fixed', 'sentence', 'semantic', 'regex', 'sliding_window', 'overlapping_window')
        **kwargs: Additional parameters for the chunker
        
    Returns:
        A chunker instance
    """
    chunker_type = chunker_type.lower()
    
    if chunker_type == "fixed":
        return FixedSizeChunker(**kwargs)
    elif chunker_type == "sentence":
        return SentenceChunker(**kwargs)
    elif chunker_type == "semantic":
        return SemanticChunker(**kwargs)
    elif chunker_type == "regex":
        return RegexChunking(**kwargs)
    elif chunker_type in ["sliding_window", "sliding"]:
        return SlidingWindowChunking(**kwargs)
    elif chunker_type in ["overlapping_window", "overlapping"]:
        return OverlappingWindowChunking(**kwargs)
    else:
        logger.warning(f"Unknown chunker type: {chunker_type}, using semantic chunker")
        return SemanticChunker(**kwargs) 