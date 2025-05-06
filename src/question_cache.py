"""
Question Caching Module for Lucky Train AI Assistant

This module provides specialized caching for questions asked to the AI system:
- Semantic similarity matching for questions
- Vector storage for efficient retrieval
- TTL-based cache expiration
- Persistence options
"""

import logging
import os
import json
import time
import hashlib
from typing import Dict, List, Optional, Tuple, Union, Any
# Try to import numpy, use a fallback if not available
try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False
    logger = logging.getLogger(__name__)
    logger.warning(
        "numpy not installed. Using fallback method for vector operations. "
        "Install with: pip install numpy"
    )
    # Simple array class to use as fallback
    class MockNumpy:
        class ndarray(list):
            def tobytes(self):
                return str(self).encode()
        
        @staticmethod
        def array(data, dtype=None):
            return MockNumpy.ndarray(data)
        
        @staticmethod
        def vstack(arrays):
            return MockNumpy.ndarray([item for sublist in arrays for item in sublist])
        
        @staticmethod
        def dot(a, b):
            # Simple dot product implementation
            if isinstance(a, list) and isinstance(b, list):
                return sum(x*y for x,y in zip(a,b))
            return 0.5  # Default similarity
        
        @staticmethod
        def argmax(arr):
            if not arr:
                return 0
            return arr.index(max(arr))
        
        @staticmethod
        def linalg():
            class Norm:
                @staticmethod
                def norm(v, axis=None, keepdims=False):
                    if isinstance(v, list):
                        return sum(x*x for x in v) ** 0.5
                    return 1.0
            return Norm()
    
    # Use the mock as a fallback
    np = MockNumpy()

from datetime import datetime
from collections import OrderedDict
import pickle
import threading

from caching import CacheManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import sentence-transformers for vector embeddings
try:
    from sentence_transformers import SentenceTransformer
    HAVE_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAVE_SENTENCE_TRANSFORMERS = False
    logger.warning(
        "sentence-transformers not installed. Using fallback hashing method for question matching. "
        "Install with: pip install sentence-transformers"
    )

class VectorCache:
    """Cache that uses vector similarity for matching similar questions."""
    
    def __init__(self, model_name: str = 'paraphrase-MiniLM-L6-v2', threshold: float = 0.85, max_size: int = 10000, ttl_seconds: int = 86400):
        """Initialize the vector cache.
        
        Args:
            model_name: Name of the sentence transformer model to use
            threshold: Similarity threshold for considering questions the same
            max_size: Maximum number of entries in the cache
            ttl_seconds: Time to live in seconds
        """
        self.model_name = model_name
        self.threshold = threshold
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        
        # Initialize model for encoding
        if HAVE_SENTENCE_TRANSFORMERS:
            try:
                self.model = SentenceTransformer(model_name)
                logger.info(f"Initialized sentence transformer model: {model_name}")
            except Exception as e:
                logger.error(f"Error loading sentence transformer model: {e}")
                HAVE_SENTENCE_TRANSFORMERS = False
        
        # Cache data
        self.cache = OrderedDict()  # {vector_key: {"vector": vector, "value": value, "expiry": expiry}}
        self.vector_matrix = None  # Will be initialized on first add
        self.vector_keys = []  # Maps row index to vector key
        
        # Lock for thread safety
        self.lock = threading.RLock()
    
    def _compute_vector(self, question: str) -> np.ndarray:
        """Compute the vector embedding for a question.
        
        Args:
            question: The question to encode
            
        Returns:
            Vector embedding as numpy array
        """
        if HAVE_SENTENCE_TRANSFORMERS:
            try:
                vector = self.model.encode(question)
                return vector
            except Exception as e:
                logger.error(f"Error encoding question: {e}")
        
        # Fallback to simple hash-based "vector"
        hash_val = hashlib.md5(question.encode()).hexdigest()
        return np.array([int(hash_val[i:i+2], 16) for i in range(0, 32, 2)], dtype=np.float32)
    
    def _compute_key(self, vector: np.ndarray) -> str:
        """Compute a key for a vector.
        
        Args:
            vector: The vector to compute a key for
            
        Returns:
            String key
        """
        # Convert vector to bytes and hash
        vector_bytes = vector.tobytes()
        return hashlib.md5(vector_bytes).hexdigest()
    
    def _rebuild_vector_matrix(self) -> None:
        """Rebuild the vector matrix for similarity comparisons."""
        with self.lock:
            if not self.cache:
                self.vector_matrix = None
                self.vector_keys = []
                return
            
            # Extract vectors and keys
            vectors = []
            self.vector_keys = []
            
            for key, entry in self.cache.items():
                vectors.append(entry["vector"])
                self.vector_keys.append(key)
            
            # Build matrix
            self.vector_matrix = np.vstack(vectors)
    
    def _find_similar(self, vector: np.ndarray) -> Tuple[Optional[str], Optional[float]]:
        """Find the most similar vector in the cache.
        
        Args:
            vector: The vector to find matches for
            
        Returns:
            Tuple of (key, similarity) or (None, None) if no match
        """
        if self.vector_matrix is None or len(self.vector_keys) == 0:
            return None, None
        
        # Compute similarity scores
        vector_normalized = vector / np.linalg.norm(vector)
        matrix_normalized = self.vector_matrix / np.linalg.norm(self.vector_matrix, axis=1, keepdims=True)
        
        similarities = np.dot(matrix_normalized, vector_normalized)
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        
        if best_score >= self.threshold:
            return self.vector_keys[best_idx], best_score
        
        return None, None
    
    def add(self, question: str, value: Any) -> None:
        """Add a question and its value to the cache.
        
        Args:
            question: The question to cache
            value: The value to cache
        """
        with self.lock:
            # Compute vector and key
            vector = self._compute_vector(question)
            key = self._compute_key(vector)
            
            # Set expiry time
            expiry = time.time() + self.ttl_seconds
            
            # Add to cache
            self.cache[key] = {
                "question": question,
                "vector": vector,
                "value": value,
                "expiry": expiry,
                "access_count": 0,
                "created": time.time()
            }
            
            # Handle max size - remove oldest entries
            while len(self.cache) > self.max_size:
                self.cache.popitem(last=False)
            
            # Rebuild vector matrix
            self._rebuild_vector_matrix()
    
    def get(self, question: str) -> Tuple[Optional[Any], Optional[float], Optional[str]]:
        """Get a cached value for a similar question.
        
        Args:
            question: The question to look up
            
        Returns:
            Tuple of (value, similarity, original_question) or (None, None, None) if not found
        """
        with self.lock:
            # Clean expired entries
            self._clean_expired()
            
            if not self.cache:
                return None, None, None
            
            # Compute vector
            vector = self._compute_vector(question)
            
            # Find similar
            key, similarity = self._find_similar(vector)
            
            if key is None:
                return None, None, None
            
            # Update access count and move to end (most recently used)
            entry = self.cache.pop(key)
            entry["access_count"] += 1
            self.cache[key] = entry
            
            return entry["value"], similarity, entry["question"]
    
    def _clean_expired(self) -> int:
        """Remove expired entries from the cache.
        
        Returns:
            Number of entries removed
        """
        now = time.time()
        expired_keys = [k for k, v in self.cache.items() if v["expiry"] < now]
        
        for key in expired_keys:
            del self.cache[key]
        
        # Rebuild vector matrix if necessary
        if expired_keys:
            self._rebuild_vector_matrix()
        
        return len(expired_keys)
    
    def clear(self) -> None:
        """Clear the cache."""
        with self.lock:
            self.cache.clear()
            self.vector_matrix = None
            self.vector_keys = []
    
    def save(self, file_path: str) -> bool:
        """Save the cache to a file.
        
        Args:
            file_path: Path to save the cache to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(file_path, 'wb') as f:
                pickle.dump({
                    "cache": self.cache,
                    "threshold": self.threshold,
                    "max_size": self.max_size,
                    "ttl_seconds": self.ttl_seconds,
                    "model_name": self.model_name
                }, f)
            return True
        except Exception as e:
            logger.error(f"Error saving vector cache: {e}")
            return False
    
    @classmethod
    def load(cls, file_path: str) -> 'VectorCache':
        """Load a cache from a file.
        
        Args:
            file_path: Path to load the cache from
            
        Returns:
            Loaded cache, or a new cache if loading fails
        """
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            cache = cls(
                model_name=data["model_name"],
                threshold=data["threshold"],
                max_size=data["max_size"],
                ttl_seconds=data["ttl_seconds"]
            )
            
            cache.cache = data["cache"]
            cache._rebuild_vector_matrix()
            
            return cache
        except Exception as e:
            logger.error(f"Error loading vector cache: {e}")
            return cls()
    
    def stats(self) -> Dict:
        """Get cache statistics.
        
        Returns:
            Dictionary of cache statistics
        """
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "threshold": self.threshold,
            "ttl_seconds": self.ttl_seconds,
            "model_name": self.model_name,
            "have_sentence_transformers": HAVE_SENTENCE_TRANSFORMERS
        }

class QuestionCache:
    """Cache for AI assistant questions and responses."""
    
    def __init__(self, config: Dict = None):
        """Initialize the question cache.
        
        Args:
            config: Cache configuration dictionary
        """
        self.config = config or {}
        
        # Get cache settings
        self.enabled = self.config.get("enabled", True)
        self.ttl_seconds = self.config.get("ttl_seconds", 24 * 60 * 60)  # 24 hours default
        self.similarity_threshold = self.config.get("similarity_threshold", 0.85)
        self.max_size = self.config.get("max_size", 10000)
        self.model_name = self.config.get("model_name", "paraphrase-MiniLM-L6-v2")
        self.persist_file = self.config.get("persist_file")
        self.persist_on_shutdown = self.config.get("persist_on_shutdown", True)
        self.load_on_startup = self.config.get("load_on_startup", True)
        
        # Initialize vector cache
        if self.enabled:
            self.cache = VectorCache(
                model_name=self.model_name,
                threshold=self.similarity_threshold,
                max_size=self.max_size,
                ttl_seconds=self.ttl_seconds
            )
            
            # Load from file if configured
            if self.persist_file and self.load_on_startup and os.path.exists(self.persist_file):
                try:
                    self.cache = VectorCache.load(self.persist_file)
                    logger.info(f"Loaded question cache from {self.persist_file}")
                except Exception as e:
                    logger.error(f"Error loading question cache: {e}")
            
            logger.info(f"Question cache initialized with similarity threshold {self.similarity_threshold}")
        else:
            self.cache = None
            logger.info("Question cache is disabled")
        
        # Register shutdown handler for persistence
        if self.enabled and self.persist_on_shutdown and self.persist_file:
            import atexit
            atexit.register(self._save_on_exit)
    
    def _save_on_exit(self) -> None:
        """Save the cache when the program exits."""
        if self.cache and self.persist_file:
            logger.info(f"Saving question cache to {self.persist_file}")
            try:
                self.cache.save(self.persist_file)
            except Exception as e:
                logger.error(f"Error saving question cache: {e}")
    
    def get(self, question: str, user_id: str = None, context: Dict = None) -> Tuple[Optional[Any], Optional[float], Optional[str]]:
        """Get a cached response for a question.
        
        Args:
            question: The question to look up
            user_id: Optional user ID for user-specific caching
            context: Optional context information for the question
            
        Returns:
            Tuple of (response, similarity, original_question) or (None, None, None) if not found
        """
        if not self.enabled or not self.cache:
            return None, None, None
        
        # Normalize the question
        normalized_question = self._normalize_question(question)
        
        # Add user and context to the key if provided
        if user_id and self.config.get("user_specific", False):
            normalized_question = f"USER:{user_id}:{normalized_question}"
        
        if context and self.config.get("context_aware", True):
            context_str = self._context_to_string(context)
            if context_str:
                normalized_question = f"CTX:{context_str}:{normalized_question}"
        
        # Get from cache
        value, similarity, original = self.cache.get(normalized_question)
        
        if value is not None:
            logger.debug(f"Question cache hit: {similarity:.2f} similarity to '{original}'")
        
        return value, similarity, original
    
    def set(self, question: str, response: Any, user_id: str = None, context: Dict = None) -> None:
        """Store a response for a question.
        
        Args:
            question: The question to cache
            response: The response to cache
            user_id: Optional user ID for user-specific caching
            context: Optional context information for the question
        """
        if not self.enabled or not self.cache:
            return
        
        # Normalize the question
        normalized_question = self._normalize_question(question)
        
        # Add user and context to the key if provided
        if user_id and self.config.get("user_specific", False):
            normalized_question = f"USER:{user_id}:{normalized_question}"
        
        if context and self.config.get("context_aware", True):
            context_str = self._context_to_string(context)
            if context_str:
                normalized_question = f"CTX:{context_str}:{normalized_question}"
        
        # Store in cache
        self.cache.add(normalized_question, {
            "response": response,
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "context": context
        })
    
    def _normalize_question(self, question: str) -> str:
        """Normalize a question for caching.
        
        Args:
            question: The question to normalize
            
        Returns:
            Normalized question
        """
        # Convert to lowercase
        normalized = question.lower()
        
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        
        # Remove punctuation if configured
        if self.config.get("remove_punctuation", True):
            normalized = ''.join(c for c in normalized if c.isalnum() or c.isspace())
        
        return normalized
    
    def _context_to_string(self, context: Dict) -> str:
        """Convert context dictionary to a string for caching.
        
        Args:
            context: Context dictionary
            
        Returns:
            String representation of the context
        """
        if not context:
            return ""
        
        # Extract relevant context fields for similarity
        relevant_fields = self.config.get("context_fields", ["language", "topic", "mode"])
        
        context_parts = []
        for field in relevant_fields:
            if field in context and context[field]:
                context_parts.append(f"{field}:{context[field]}")
        
        return "|".join(context_parts)
    
    def clear(self) -> None:
        """Clear the cache."""
        if self.enabled and self.cache:
            self.cache.clear()
    
    def stats(self) -> Dict:
        """Get cache statistics.
        
        Returns:
            Dictionary of cache statistics
        """
        if not self.enabled or not self.cache:
            return {"enabled": False}
        
        stats = self.cache.stats()
        stats["enabled"] = True
        return stats
    
    def save(self) -> bool:
        """Manually save the cache to disk.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not self.cache or not self.persist_file:
            return False
        
        return self.cache.save(self.persist_file)

# Function to get a question cache instance
def get_question_cache(config: Dict = None) -> QuestionCache:
    """Get a question cache instance.
    
    Args:
        config: Cache configuration
        
    Returns:
        Configured question cache
    """
    return QuestionCache(config) 