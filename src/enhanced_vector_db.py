"""
Enhanced Vector Database Module for Lucky Train AI

Features:
- Support for multiple vector database backends (Qdrant, Weaviate, Milvus)
- Memory-efficient caching with TTL
- Sharding support for large collections
- Automatic embeddings refresh on knowledge base updates
- Performance monitoring and logging
"""

import os
import time
import logging
import json
import numpy as np
import threading
import hashlib
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmbeddingCache:
    """Memory-efficient cache for vector embeddings."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """Initialize the embedding cache.
        
        Args:
            max_size: Maximum number of embeddings to store
            ttl_seconds: Time-to-live in seconds
        """
        self.cache = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.lock = threading.RLock()
        
        # Stats
        self.hits = 0
        self.misses = 0
        self.total_saved_time = 0
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """Get an embedding from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Embedding vector or None if not found
        """
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None
            
            # Check if expired
            entry = self.cache[key]
            if time.time() - entry["timestamp"] > self.ttl_seconds:
                del self.cache[key]
                self.misses += 1
                return None
            
            self.hits += 1
            self.total_saved_time += entry["generation_time"]
            return entry["embedding"]
    
    def set(self, key: str, embedding: np.ndarray, generation_time: float) -> None:
        """Set an embedding in the cache.
        
        Args:
            key: Cache key
            embedding: Embedding vector
            generation_time: Time taken to generate the embedding
        """
        with self.lock:
            # If cache is full, remove oldest entries
            if len(self.cache) >= self.max_size:
                # Sort by timestamp and keep only max_size - 1 newest entries
                sorted_entries = sorted(
                    [(k, v["timestamp"]) for k, v in self.cache.items()],
                    key=lambda x: x[1]
                )
                for k, _ in sorted_entries[:len(self.cache) - self.max_size + 1]:
                    del self.cache[k]
            
            self.cache[key] = {
                "embedding": embedding,
                "timestamp": time.time(),
                "generation_time": generation_time
            }
    
    def clear(self) -> None:
        """Clear the cache."""
        with self.lock:
            self.cache.clear()
    
    def get_stats(self) -> Dict:
        """Get cache statistics.
        
        Returns:
            Dictionary of cache statistics
        """
        with self.lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0
            
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hit_rate": hit_rate,
                "hits": self.hits,
                "misses": self.misses,
                "ttl_seconds": self.ttl_seconds,
                "time_saved": self.total_saved_time
            }

class VectorDBManager:
    """Manager for vector database operations."""
    
    def __init__(self, config: Dict = None):
        """Initialize the vector database manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.db_type = self.config.get("db_type", "qdrant")
        self.collection_name = self.config.get("collection_name", "lucky_train_kb")
        self.sharding_enabled = self.config.get("sharding_enabled", False)
        self.num_shards = self.config.get("num_shards", 3)
        self.embedding_dim = self.config.get("embedding_dim", 1536)  # Default for OpenAI embeddings
        
        # Caching settings
        self.enable_cache = self.config.get("enable_cache", True)
        self.cache_size = self.config.get("cache_size", 1000)
        self.cache_ttl = self.config.get("cache_ttl", 3600)
        
        # Initialize the embedding cache
        self.embedding_cache = EmbeddingCache(self.cache_size, self.cache_ttl)
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Stats
        self.stats = {
            "searches": 0,
            "inserts": 0,
            "updates": 0,
            "deletes": 0,
            "avg_search_time": 0,
            "total_search_time": 0
        }
        
        # Initialize backend
        self._init_backend()
        
        logger.info(f"Initialized vector database manager with backend: {self.db_type}")
    
    def _init_backend(self) -> None:
        """Initialize the vector database backend."""
        if self.db_type == "qdrant":
            self._init_qdrant()
        elif self.db_type == "weaviate":
            self._init_weaviate()
        elif self.db_type == "milvus":
            self._init_milvus()
        else:
            raise ValueError(f"Unsupported vector database type: {self.db_type}")
    
    def _init_qdrant(self) -> None:
        """Initialize Qdrant vector database."""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http.models import Distance, VectorParams
            
            # Create client
            qdrant_url = self.config.get("qdrant_url", "")
            qdrant_api_key = self.config.get("qdrant_api_key", "")
            
            if qdrant_url:
                self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
            else:
                qdrant_path = self.config.get("qdrant_path", "./qdrant_data")
                self.client = QdrantClient(path=qdrant_path)
            
            # Create collections for shards if enabled
            if self.sharding_enabled:
                for shard_id in range(self.num_shards):
                    collection_name = f"{self.collection_name}_shard_{shard_id}"
                    self._create_qdrant_collection(collection_name)
            else:
                self._create_qdrant_collection(self.collection_name)
            
            logger.info(f"Initialized Qdrant backend with {self.num_shards if self.sharding_enabled else 1} shard(s)")
        
        except ImportError:
            raise ImportError("qdrant-client package not installed. Run 'pip install qdrant-client'")
    
    def _create_qdrant_collection(self, collection_name: str) -> None:
        """Create a Qdrant collection if it doesn't exist.
        
        Args:
            collection_name: Name of the collection
        """
        from qdrant_client.http.models import Distance, VectorParams
        
        # Check if collection exists
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if collection_name not in collection_names:
            # Create new collection
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE)
            )
            logger.info(f"Created new Qdrant collection: {collection_name}")
    
    def _init_weaviate(self) -> None:
        """Initialize Weaviate vector database."""
        try:
            import weaviate
            
            # Create client
            weaviate_url = self.config.get("weaviate_url", "")
            weaviate_api_key = self.config.get("weaviate_api_key", "")
            
            auth_config = weaviate.auth.AuthApiKey(api_key=weaviate_api_key) if weaviate_api_key else None
            
            self.client = weaviate.Client(
                url=weaviate_url,
                auth_client_secret=auth_config
            )
            
            # Check if schema exists
            class_obj = {
                "class": self.collection_name,
                "vectorizer": "none",  # We will provide our own vectors
                "properties": [
                    {
                        "name": "content",
                        "dataType": ["text"]
                    },
                    {
                        "name": "metadata",
                        "dataType": ["text"]
                    }
                ]
            }
            
            # Create schema if it doesn't exist
            if not self.client.schema.contains({"classes": [{"class": self.collection_name}]}):
                self.client.schema.create_class(class_obj)
                logger.info(f"Created Weaviate schema for class: {self.collection_name}")
            
            logger.info("Initialized Weaviate backend")
        
        except ImportError:
            raise ImportError("weaviate-client package not installed. Run 'pip install weaviate-client'")
    
    def _init_milvus(self) -> None:
        """Initialize Milvus vector database."""
        try:
            from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType
            
            # Connect to Milvus
            milvus_host = self.config.get("milvus_host", "localhost")
            milvus_port = self.config.get("milvus_port", "19530")
            
            connections.connect(host=milvus_host, port=milvus_port)
            
            # Define collection schema
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim)
            ]
            
            schema = CollectionSchema(fields=fields, description=f"Lucky Train knowledge base collection")
            
            # Create collection if it doesn't exist
            if not utility.has_collection(self.collection_name):
                Collection(name=self.collection_name, schema=schema)
                logger.info(f"Created Milvus collection: {self.collection_name}")
            
            self.collection = Collection(name=self.collection_name)
            self.collection.load()
            
            logger.info("Initialized Milvus backend")
        
        except ImportError:
            raise ImportError("pymilvus package not installed. Run 'pip install pymilvus'")
    
    def _get_shard_id(self, text: str) -> int:
        """Get shard ID for a given text.
        
        Args:
            text: Input text
            
        Returns:
            Shard ID
        """
        if not self.sharding_enabled:
            return 0
        
        # Use a hash function to determine the shard
        hash_value = int(hashlib.md5(text.encode()).hexdigest(), 16)
        return hash_value % self.num_shards
    
    def _get_collection_name(self, shard_id: int) -> str:
        """Get collection name for a given shard ID.
        
        Args:
            shard_id: Shard ID
            
        Returns:
            Collection name
        """
        if not self.sharding_enabled:
            return self.collection_name
        
        return f"{self.collection_name}_shard_{shard_id}"
    
    def _get_embedding(self, text: str) -> Tuple[np.ndarray, float]:
        """Get embedding for a text.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (embedding vector, generation time)
        """
        # Check cache first
        if self.enable_cache:
            cache_key = hashlib.md5(text.encode()).hexdigest()
            cached_embedding = self.embedding_cache.get(cache_key)
            
            if cached_embedding is not None:
                return cached_embedding, 0
        
        # Generate embedding
        embedding_model = self.config.get("embedding_model", {"type": "openai", "name": "text-embedding-3-small"})
        
        start_time = time.time()
        
        if embedding_model["type"] == "openai":
            import openai
            
            openai.api_key = os.getenv("OPENAI_API_KEY")
            
            response = openai.Embedding.create(
                input=text,
                model=embedding_model["name"]
            )
            
            embedding = np.array(response["data"][0]["embedding"], dtype=np.float32)
        
        elif embedding_model["type"] == "huggingface":
            from sentence_transformers import SentenceTransformer
            
            model = SentenceTransformer(embedding_model["name"])
            embedding = model.encode(text, normalize_embeddings=True)
        
        else:
            raise ValueError(f"Unsupported embedding model type: {embedding_model['type']}")
        
        generation_time = time.time() - start_time
        
        # Add to cache
        if self.enable_cache:
            self.embedding_cache.set(cache_key, embedding, generation_time)
        
        return embedding, generation_time
    
    def add_documents(self, documents: List[Dict]) -> bool:
        """Add documents to the vector database.
        
        Args:
            documents: List of documents with 'content' and 'metadata' fields
            
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            try:
                # Process each document
                for doc in documents:
                    content = doc["content"]
                    metadata = doc.get("metadata", {})
                    doc_id = doc.get("id", str(time.time()) + "_" + hashlib.md5(content.encode()).hexdigest()[:8])
                    
                    # Get embedding
                    embedding, _ = self._get_embedding(content)
                    
                    # Determine shard
                    shard_id = self._get_shard_id(content)
                    collection_name = self._get_collection_name(shard_id)
                    
                    # Insert into database
                    if self.db_type == "qdrant":
                        from qdrant_client.http.models import PointStruct
                        
                        point = PointStruct(
                            id=doc_id,
                            vector=embedding.tolist(),
                            payload={"content": content, "metadata": json.dumps(metadata)}
                        )
                        
                        self.client.upsert(
                            collection_name=collection_name,
                            points=[point]
                        )
                    
                    elif self.db_type == "weaviate":
                        self.client.data_object.create(
                            {
                                "content": content,
                                "metadata": json.dumps(metadata)
                            },
                            class_name=self.collection_name,
                            uuid=doc_id,
                            vector=embedding.tolist()
                        )
                    
                    elif self.db_type == "milvus":
                        self.collection.insert([
                            [doc_id],
                            [content],
                            [embedding.tolist()]
                        ])
                
                self.stats["inserts"] += len(documents)
                logger.info(f"Added {len(documents)} documents to vector database")
                return True
            
            except Exception as e:
                logger.error(f"Error adding documents to vector database: {e}")
                return False
    
    def search(self, query: str, top_k: int = 5, filter_criteria: Dict = None) -> List[Dict]:
        """Search for similar documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_criteria: Filter criteria
            
        Returns:
            List of search results
        """
        with self.lock:
            try:
                start_time = time.time()
                
                # Get embedding
                embedding, _ = self._get_embedding(query)
                
                results = []
                
                if self.sharding_enabled:
                    # Search across all shards
                    for shard_id in range(self.num_shards):
                        collection_name = self._get_collection_name(shard_id)
                        shard_results = self._search_in_collection(collection_name, embedding, top_k, filter_criteria)
                        results.extend(shard_results)
                    
                    # Sort and limit to top_k
                    results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
                else:
                    # Search in single collection
                    results = self._search_in_collection(self.collection_name, embedding, top_k, filter_criteria)
                
                # Update stats
                search_time = time.time() - start_time
                self.stats["searches"] += 1
                self.stats["total_search_time"] += search_time
                self.stats["avg_search_time"] = self.stats["total_search_time"] / self.stats["searches"]
                
                return results
            
            except Exception as e:
                logger.error(f"Error searching vector database: {e}")
                return []
    
    def _search_in_collection(self, collection_name: str, embedding: np.ndarray, top_k: int, filter_criteria: Dict = None) -> List[Dict]:
        """Search in a specific collection.
        
        Args:
            collection_name: Collection name
            embedding: Query embedding
            top_k: Number of results
            filter_criteria: Filter criteria
            
        Returns:
            List of search results
        """
        if self.db_type == "qdrant":
            from qdrant_client.http.models import Filter, FieldCondition, MatchValue
            
            # Build filter if provided
            filter_obj = None
            if filter_criteria:
                conditions = []
                for field, value in filter_criteria.items():
                    if field.startswith("metadata."):
                        # Extract field from metadata JSON
                        field_path = field.replace("metadata.", "")
                        conditions.append(
                            FieldCondition(
                                key=f"metadata.{field_path}",
                                match=MatchValue(value=value)
                            )
                        )
                    else:
                        conditions.append(
                            FieldCondition(
                                key=field,
                                match=MatchValue(value=value)
                            )
                        )
                
                filter_obj = Filter(must=conditions)
            
            # Perform search
            search_result = self.client.search(
                collection_name=collection_name,
                query_vector=embedding.tolist(),
                limit=top_k,
                query_filter=filter_obj
            )
            
            # Format results
            results = []
            for hit in search_result:
                metadata = json.loads(hit.payload.get("metadata", "{}"))
                results.append({
                    "id": hit.id,
                    "content": hit.payload.get("content", ""),
                    "metadata": metadata,
                    "score": hit.score
                })
            
            return results
        
        elif self.db_type == "weaviate":
            # Build filter if provided
            where_filter = None
            if filter_criteria:
                where_clauses = []
                for field, value in filter_criteria.items():
                    if field.startswith("metadata."):
                        # We can't easily filter on metadata JSON fields
                        # This would require additional processing
                        continue
                    else:
                        where_clauses.append({
                            "path": [field],
                            "operator": "Equal",
                            "valueText": value
                        })
                
                if where_clauses:
                    where_filter = {"operator": "And", "operands": where_clauses}
            
            # Perform search
            result = (
                self.client.query
                .get(self.collection_name, ["content", "metadata"])
                .with_near_vector({"vector": embedding.tolist()})
                .with_limit(top_k)
            )
            
            if where_filter:
                result = result.with_where(where_filter)
            
            response = result.do()
            
            # Format results
            results = []
            for hit in response["data"]["Get"][self.collection_name]:
                metadata = json.loads(hit.get("metadata", "{}"))
                results.append({
                    "id": hit["_additional"]["id"],
                    "content": hit.get("content", ""),
                    "metadata": metadata,
                    "score": hit["_additional"]["certainty"]
                })
            
            return results
        
        elif self.db_type == "milvus":
            # Perform search
            search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
            
            result = self.collection.search(
                data=[embedding.tolist()],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                output_fields=["content"]
            )
            
            # Format results
            results = []
            for hits in result:
                for hit in hits:
                    results.append({
                        "id": hit.id,
                        "content": hit.entity.get("content", ""),
                        "metadata": {},  # Milvus doesn't support complex metadata without schema changes
                        "score": hit.distance
                    })
            
            return results
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        if self.enable_cache:
            self.embedding_cache.clear()
            logger.info("Cleared embedding cache")
    
    def get_stats(self) -> Dict:
        """Get vector database statistics.
        
        Returns:
            Dictionary of statistics
        """
        with self.lock:
            stats = self.stats.copy()
            
            if self.enable_cache:
                stats["cache"] = self.embedding_cache.get_stats()
            
            # Add backend-specific stats
            if self.db_type == "qdrant":
                collections_info = []
                
                if self.sharding_enabled:
                    for shard_id in range(self.num_shards):
                        collection_name = self._get_collection_name(shard_id)
                        try:
                            collection_info = self.client.get_collection(collection_name)
                            collections_info.append({
                                "name": collection_name,
                                "vectors_count": collection_info.vectors_count,
                                "status": collection_info.status
                            })
                        except Exception as e:
                            logger.error(f"Error getting collection info: {e}")
                else:
                    try:
                        collection_info = self.client.get_collection(self.collection_name)
                        collections_info.append({
                            "name": self.collection_name,
                            "vectors_count": collection_info.vectors_count,
                            "status": collection_info.status
                        })
                    except Exception as e:
                        logger.error(f"Error getting collection info: {e}")
                
                stats["collections"] = collections_info
            
            return stats
    
    def optimize(self) -> None:
        """Optimize the vector database."""
        with self.lock:
            if self.db_type == "qdrant":
                # Optimize each collection if sharding is enabled
                if self.sharding_enabled:
                    for shard_id in range(self.num_shards):
                        collection_name = self._get_collection_name(shard_id)
                        try:
                            # Force optimization
                            self.client.optimize_collection(collection_name, cpu_resources=2)
                            logger.info(f"Optimized collection: {collection_name}")
                        except Exception as e:
                            logger.error(f"Error optimizing collection {collection_name}: {e}")
                else:
                    try:
                        # Force optimization
                        self.client.optimize_collection(self.collection_name, cpu_resources=2)
                        logger.info(f"Optimized collection: {self.collection_name}")
                    except Exception as e:
                        logger.error(f"Error optimizing collection {self.collection_name}: {e}")
            
            # For other backends, we don't have a standard optimization method

# Singleton instance
_vector_db_manager = None

def get_vector_db_manager(config: Dict = None) -> VectorDBManager:
    """Get the vector database manager instance.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Vector database manager instance
    """
    global _vector_db_manager
    
    if _vector_db_manager is None:
        _vector_db_manager = VectorDBManager(config)
    
    return _vector_db_manager 