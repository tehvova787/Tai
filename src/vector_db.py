"""
Vector Database Handler for Lucky Train AI Assistant

This module provides vector database functionality for semantic search
and knowledge base management.
"""

import os
import json
import logging
import time
import hashlib
from typing import Dict, List, Optional, Union, Any
from dotenv import load_dotenv
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Optional imports for different vector DB backends
try:
    # Import for OpenAI embeddings
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI package not available. OpenAI embeddings will not be supported.")

try:
    # Import for sentence transformers (local embeddings)
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence_transformers package not available. Local embeddings will not be supported.")

try:
    # Import for Qdrant
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qdrant_models
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logger.warning("qdrant_client package not available. Qdrant vector database will not be supported.")

try:
    # Import for Pinecone
    import pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    logger.warning("pinecone-client package not available. Pinecone vector database will not be supported.")

try:
    # Import for Weaviate
    import weaviate
    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False
    logger.warning("weaviate-client package not available. Weaviate vector database will not be supported.")

class VectorDBHandler:
    """Handler for vector database operations."""
    
    def __init__(self, config_path: str = "./config/config.json"):
        """Initialize the vector database handler.
        
        Args:
            config_path: Path to the configuration file.
        """
        self.config = self._load_config(config_path)
        self.vector_db_settings = self.config.get("vector_db_settings", {})
        self.db_type = self.vector_db_settings.get("db_type", "qdrant")
        self.collection_name = self.vector_db_settings.get("collection_name", "lucky_train_kb")
        
        # Initialize embedding model
        self.embedding_model = self._initialize_embedding_model()
        self.embedding_dim = self._get_embedding_dimension()
        
        # Cache for embeddings to reduce API calls and computation
        self.embedding_cache = {}
        self.cache_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                     "data", "embedding_cache.json")
        self._load_embedding_cache()
        
        # Initialize the vector database client
        self.client = self._initialize_vector_db()
        
        # Create collection if it doesn't exist
        if self.client is not None:
            self._ensure_collection_exists()
        
        logger.info(f"Vector database handler initialized with backend: {self.db_type}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load the configuration from a JSON file.
        
        Args:
            config_path: Path to the configuration file.
            
        Returns:
            The configuration as a dictionary.
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            # Return default configuration
            return {}
    
    def _initialize_embedding_model(self) -> Any:
        """Initialize the embedding model based on configuration.
        
        Returns:
            Embedding model instance.
        """
        embedding_config = self.vector_db_settings.get("embedding_model", {})
        model_type = embedding_config.get("type", "openai")
        model_name = embedding_config.get("name", "text-embedding-3-small")
        
        if model_type == "openai" and OPENAI_AVAILABLE:
            # Initialize OpenAI client
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if openai_api_key:
                self.openai_client = openai.OpenAI(api_key=openai_api_key)
                logger.info(f"Using OpenAI embedding model: {model_name}")
                return {"type": "openai", "name": model_name}
            else:
                logger.warning("OpenAI API key not found, falling back to local embeddings")
                model_type = "sentence_transformers"
        
        if model_type == "sentence_transformers" and SENTENCE_TRANSFORMERS_AVAILABLE:
            # Use a smaller model by default for local embeddings
            local_model_name = model_name
            if model_name == "text-embedding-3-small":
                local_model_name = "all-MiniLM-L6-v2"
            
            try:
                model = SentenceTransformer(local_model_name)
                logger.info(f"Using SentenceTransformer model: {local_model_name}")
                return {"type": "sentence_transformers", "model": model, "name": local_model_name}
            except Exception as e:
                logger.error(f"Failed to load SentenceTransformer model: {e}")
                return None
        
        logger.error("No embedding model could be initialized")
        return None
    
    def _get_embedding_dimension(self) -> int:
        """Get the embedding dimension based on the model.
        
        Returns:
            Embedding dimension.
        """
        if self.embedding_model is None:
            return 384  # Default dimension
        
        if self.embedding_model.get("type") == "openai":
            # OpenAI embedding dimensions
            model_name = self.embedding_model.get("name")
            if model_name == "text-embedding-3-small":
                return 1536
            elif model_name == "text-embedding-3-large":
                return 3072
            elif model_name == "text-embedding-ada-002":
                return 1536
            else:
                return 1536  # Default for unknown OpenAI models
        
        elif self.embedding_model.get("type") == "sentence_transformers":
            # Get dimension from the model
            try:
                model = self.embedding_model.get("model")
                return model.get_sentence_embedding_dimension()
            except:
                # Default dimensions for common models
                model_name = self.embedding_model.get("name")
                if model_name == "all-MiniLM-L6-v2":
                    return 384
                elif model_name == "all-mpnet-base-v2":
                    return 768
                else:
                    return 384  # Default
        
        return 384  # Default dimension
    
    def _initialize_vector_db(self) -> Any:
        """Initialize the vector database client based on configuration.
        
        Returns:
            Vector database client.
        """
        if self.db_type == "qdrant" and QDRANT_AVAILABLE:
            return self._initialize_qdrant()
        elif self.db_type == "pinecone" and PINECONE_AVAILABLE:
            return self._initialize_pinecone()
        elif self.db_type == "weaviate" and WEAVIATE_AVAILABLE:
            return self._initialize_weaviate()
        else:
            logger.error(f"Unsupported vector database type: {self.db_type}")
            return None
    
    def _initialize_qdrant(self) -> Optional[QdrantClient]:
        """Initialize Qdrant client.
        
        Returns:
            Qdrant client instance or None if initialization failed.
        """
        try:
            qdrant_url = self.vector_db_settings.get("qdrant_url")
            qdrant_api_key = self.vector_db_settings.get("qdrant_api_key")
            qdrant_path = self.vector_db_settings.get("qdrant_path")
            
            # Use environment variables if not in config
            if not qdrant_url:
                qdrant_url = os.getenv("QDRANT_URL")
            if not qdrant_api_key:
                qdrant_api_key = os.getenv("QDRANT_API_KEY")
            
            # Connect to Qdrant
            if qdrant_url:
                # Cloud or remote Qdrant
                client = QdrantClient(
                    url=qdrant_url,
                    api_key=qdrant_api_key
                )
                logger.info(f"Connected to Qdrant at {qdrant_url}")
            else:
                # Local Qdrant
                if not qdrant_path:
                    qdrant_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                              "data", "qdrant_data")
                os.makedirs(qdrant_path, exist_ok=True)
                client = QdrantClient(path=qdrant_path)
                logger.info(f"Using local Qdrant at {qdrant_path}")
            
            return client
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            return None
    
    def _initialize_pinecone(self) -> Optional[Any]:
        """Initialize Pinecone client.
        
        Returns:
            Pinecone index instance or None if initialization failed.
        """
        try:
            pinecone_api_key = self.vector_db_settings.get("pinecone_api_key")
            pinecone_environment = self.vector_db_settings.get("pinecone_environment")
            pinecone_index_name = self.vector_db_settings.get("pinecone_index", "lucky-train")
            
            # Use environment variables if not in config
            if not pinecone_api_key:
                pinecone_api_key = os.getenv("PINECONE_API_KEY")
            if not pinecone_environment:
                pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
            
            if not pinecone_api_key or not pinecone_environment:
                logger.error("Pinecone API key or environment not found")
                return None
            
            # Initialize Pinecone
            pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
            
            # Check if index exists
            if pinecone_index_name not in pinecone.list_indexes():
                # Create index
                pinecone.create_index(
                    name=pinecone_index_name,
                    dimension=self.embedding_dim,
                    metric="cosine"
                )
                logger.info(f"Created Pinecone index: {pinecone_index_name}")
            
            # Connect to index
            index = pinecone.Index(pinecone_index_name)
            logger.info(f"Connected to Pinecone index: {pinecone_index_name}")
            
            return index
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone client: {e}")
            return None
    
    def _initialize_weaviate(self) -> Optional[Any]:
        """Initialize Weaviate client.
        
        Returns:
            Weaviate client instance or None if initialization failed.
        """
        try:
            weaviate_url = self.vector_db_settings.get("weaviate_url")
            weaviate_api_key = self.vector_db_settings.get("weaviate_api_key")
            
            # Use environment variables if not in config
            if not weaviate_url:
                weaviate_url = os.getenv("WEAVIATE_URL")
            if not weaviate_api_key:
                weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
            
            if not weaviate_url:
                logger.error("Weaviate URL not found")
                return None
            
            # Connect to Weaviate
            auth_config = weaviate.auth.AuthApiKey(api_key=weaviate_api_key) if weaviate_api_key else None
            client = weaviate.Client(
                url=weaviate_url,
                auth_client_secret=auth_config
            )
            logger.info(f"Connected to Weaviate at {weaviate_url}")
            
            return client
        except Exception as e:
            logger.error(f"Failed to initialize Weaviate client: {e}")
            return None
    
    def _ensure_collection_exists(self) -> None:
        """Ensure that the vector collection exists in the database."""
        if self.client is None:
            logger.error("Vector database client not initialized")
            return
        
        try:
            if self.db_type == "qdrant":
                # Check if collection exists in Qdrant
                collections = self.client.get_collections().collections
                collection_names = [collection.name for collection in collections]
                
                if self.collection_name not in collection_names:
                    # Create collection
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=qdrant_models.VectorParams(
                            size=self.embedding_dim,
                            distance=qdrant_models.Distance.COSINE
                        )
                    )
                    logger.info(f"Created Qdrant collection: {self.collection_name}")
            
            elif self.db_type == "weaviate":
                # Check if class exists in Weaviate
                class_name = self.collection_name.title().replace("_", "")
                schema = self.client.schema.get()
                
                class_exists = False
                for cls in schema["classes"]:
                    if cls["class"] == class_name:
                        class_exists = True
                        break
                
                if not class_exists:
                    # Create class
                    class_obj = {
                        "class": class_name,
                        "vectorizer": "none",  # We'll provide vectors manually
                        "properties": [
                            {
                                "name": "text",
                                "dataType": ["text"]
                            },
                            {
                                "name": "source",
                                "dataType": ["text"]
                            },
                            {
                                "name": "metadata",
                                "dataType": ["text"]
                            }
                        ]
                    }
                    self.client.schema.create_class(class_obj)
                    logger.info(f"Created Weaviate class: {class_name}")
        
        except Exception as e:
            logger.error(f"Failed to ensure collection exists: {e}")
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding vector for text.
        
        Args:
            text: Text to embed.
            
        Returns:
            Embedding vector.
        """
        # Check embedding cache first
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        if self.embedding_model is None:
            logger.error("Embedding model not initialized")
            # Return a zero vector as fallback
            return [0.0] * self.embedding_dim
        
        try:
            if self.embedding_model.get("type") == "openai":
                # Get embedding from OpenAI
                model_name = self.embedding_model.get("name")
                response = self.openai_client.embeddings.create(
                    input=text,
                    model=model_name
                )
                embedding = response.data[0].embedding
            
            elif self.embedding_model.get("type") == "sentence_transformers":
                # Get embedding from sentence transformers
                model = self.embedding_model.get("model")
                embedding = model.encode(text).tolist()
            
            else:
                logger.error(f"Unsupported embedding model type: {self.embedding_model.get('type')}")
                # Return a zero vector as fallback
                return [0.0] * self.embedding_dim
            
            # Cache the embedding
            self.embedding_cache[cache_key] = embedding
            
            # Periodically save the cache (every 100 new embeddings)
            if len(self.embedding_cache) % 100 == 0:
                self._save_embedding_cache()
            
            return embedding
        
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Return a zero vector as fallback
            return [0.0] * self.embedding_dim
    
    def _process_text_chunks(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Process text into chunks for more effective retrieval.
        
        Args:
            text: Text to process
            chunk_size: Maximum chunk size in characters
            overlap: Overlap between chunks in characters
            
        Returns:
            List of text chunks
        """
        # Simple implementation - split by sentences
        import re
        
        # Break text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed chunk size, save current chunk and start a new one
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk)
                # Start new chunk with overlap
                overlap_point = max(0, len(current_chunk) - overlap)
                current_chunk = current_chunk[overlap_point:] + sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _load_embedding_cache(self) -> None:
        """Load embedding cache from file."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    self.embedding_cache = cache_data
                    logger.info(f"Loaded {len(self.embedding_cache)} embeddings from cache")
        except Exception as e:
            logger.error(f"Failed to load embedding cache: {e}")
            self.embedding_cache = {}
    
    def _save_embedding_cache(self) -> None:
        """Save embedding cache to file."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.embedding_cache, f)
            
            logger.info(f"Saved {len(self.embedding_cache)} embeddings to cache")
        except Exception as e:
            logger.error(f"Failed to save embedding cache: {e}")
    
    def load_knowledge_base(self, knowledge_base: Dict) -> bool:
        """Load a knowledge base into the vector database.
        
        Args:
            knowledge_base: Knowledge base as a dictionary
            
        Returns:
            True if successful, False otherwise
        """
        if self.client is None:
            logger.error("Vector database client not initialized")
            return False
        
        try:
            # Extract text chunks and metadata from the knowledge base
            chunks = []
            
            # Process the knowledge base into text chunks
            for section_name, section_data in knowledge_base.items():
                processed_chunks = self._process_knowledge_base_section(section_name, section_data)
                chunks.extend(processed_chunks)
            
            # Log number of chunks
            logger.info(f"Extracted {len(chunks)} chunks from knowledge base")
            
            # Store chunks in the vector database
            if self.db_type == "qdrant":
                return self._store_chunks_in_qdrant(chunks)
            elif self.db_type == "pinecone":
                return self._store_chunks_in_pinecone(chunks)
            elif self.db_type == "weaviate":
                return self._store_chunks_in_weaviate(chunks)
            else:
                logger.error(f"Unsupported vector database type: {self.db_type}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load knowledge base into vector database: {e}")
            return False
    
    def _process_knowledge_base_section(self, section_name: str, data: Any, path: List[str] = None) -> List[Dict]:
        """Process a section of the knowledge base into text chunks.
        
        Args:
            section_name: Section name
            data: Section data
            path: Current path in the knowledge base structure
            
        Returns:
            List of chunks with text and metadata
        """
        chunks = []
        current_path = path or [section_name]
        
        if isinstance(data, dict):
            # Process dictionary
            for key, value in data.items():
                new_path = current_path + [key]
                processed_chunks = self._process_knowledge_base_section(key, value, new_path)
                chunks.extend(processed_chunks)
        
        elif isinstance(data, list):
            # Process list
            if all(isinstance(item, str) for item in data):
                # List of strings - join them
                text = ". ".join(data)
                text_chunks = self._process_text_chunks(text)
                
                for i, chunk in enumerate(text_chunks):
                    chunks.append({
                        "text": chunk,
                        "metadata": {
                            "source": "/".join(current_path),
                            "chunk_index": i,
                            "total_chunks": len(text_chunks)
                        }
                    })
            else:
                # List of other items - process each one
                for i, item in enumerate(data):
                    new_path = current_path + [f"item_{i}"]
                    processed_chunks = self._process_knowledge_base_section(f"item_{i}", item, new_path)
                    chunks.extend(processed_chunks)
        
        elif isinstance(data, str) and len(data.split()) > 3:
            # Process string (only if it has more than 3 words)
            text_chunks = self._process_text_chunks(data)
            
            for i, chunk in enumerate(text_chunks):
                chunks.append({
                    "text": chunk,
                    "metadata": {
                        "source": "/".join(current_path),
                        "chunk_index": i,
                        "total_chunks": len(text_chunks)
                    }
                })
        
        return chunks
    
    def _store_chunks_in_qdrant(self, chunks: List[Dict]) -> bool:
        """Store text chunks in Qdrant.
        
        Args:
            chunks: List of chunks with text and metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Prepare points for batch upload
            points = []
            
            for i, chunk in enumerate(chunks):
                # Get embedding for the chunk
                embedding = self._get_embedding(chunk["text"])
                
                # Create point
                point = qdrant_models.PointStruct(
                    id=str(i),
                    vector=embedding,
                    payload={
                        "text": chunk["text"],
                        "metadata": json.dumps(chunk["metadata"])
                    }
                )
                
                points.append(point)
                
                # Upload in batches of 100
                if len(points) >= 100 or i == len(chunks) - 1:
                    # Upload batch
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=points
                    )
                    
                    # Clear points
                    points = []
                    
                    logger.info(f"Uploaded {min(i+1, len(chunks))} of {len(chunks)} chunks to Qdrant")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store chunks in Qdrant: {e}")
            return False
    
    def _store_chunks_in_pinecone(self, chunks: List[Dict]) -> bool:
        """Store text chunks in Pinecone.
        
        Args:
            chunks: List of chunks with text and metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Prepare vectors for batch upload
            vectors = []
            
            for i, chunk in enumerate(chunks):
                # Get embedding for the chunk
                embedding = self._get_embedding(chunk["text"])
                
                # Create vector
                vector = {
                    "id": f"chunk_{i}",
                    "values": embedding,
                    "metadata": {
                        "text": chunk["text"],
                        "source": chunk["metadata"]["source"],
                        "chunk_index": chunk["metadata"]["chunk_index"],
                        "total_chunks": chunk["metadata"]["total_chunks"]
                    }
                }
                
                vectors.append(vector)
                
                # Upload in batches of 100
                if len(vectors) >= 100 or i == len(chunks) - 1:
                    # Upload batch
                    self.client.upsert(vectors=vectors)
                    
                    # Clear vectors
                    vectors = []
                    
                    logger.info(f"Uploaded {min(i+1, len(chunks))} of {len(chunks)} chunks to Pinecone")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store chunks in Pinecone: {e}")
            return False
    
    def _store_chunks_in_weaviate(self, chunks: List[Dict]) -> bool:
        """Store text chunks in Weaviate.
        
        Args:
            chunks: List of chunks with text and metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get class name
            class_name = self.collection_name.title().replace("_", "")
            
            with self.client.batch as batch:
                # Set batch size
                batch.batch_size = 100
                
                # Process each chunk
                for i, chunk in enumerate(chunks):
                    # Get embedding for the chunk
                    embedding = self._get_embedding(chunk["text"])
                    
                    # Create data object
                    data_object = {
                        "text": chunk["text"],
                        "source": chunk["metadata"]["source"],
                        "metadata": json.dumps(chunk["metadata"])
                    }
                    
                    # Add to batch
                    batch.add_data_object(
                        data_object=data_object,
                        class_name=class_name,
                        vector=embedding
                    )
                    
                    # Log progress
                    if (i + 1) % 100 == 0 or i == len(chunks) - 1:
                        logger.info(f"Uploaded {min(i+1, len(chunks))} of {len(chunks)} chunks to Weaviate")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store chunks in Weaviate: {e}")
            return False
    
    def search(self, query: str, top_k: int = 5, filter_criteria: Dict = None) -> List[Dict]:
        """Search for relevant documents in the vector database.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            filter_criteria: Filter criteria for search
            
        Returns:
            List of relevant documents
        """
        if self.client is None:
            logger.error("Vector database client not initialized")
            return []
        
        try:
            # Get embedding for the query
            query_embedding = self._get_embedding(query)
            
            # Search in the vector database
            if self.db_type == "qdrant":
                return self._search_in_qdrant(query_embedding, top_k, filter_criteria)
            elif self.db_type == "pinecone":
                return self._search_in_pinecone(query_embedding, top_k, filter_criteria)
            elif self.db_type == "weaviate":
                return self._search_in_weaviate(query_embedding, top_k, filter_criteria)
            else:
                logger.error(f"Unsupported vector database type: {self.db_type}")
                return []
                
        except Exception as e:
            logger.error(f"Error searching in vector database: {e}")
            return []
    
    def _search_in_qdrant(self, query_embedding: List[float], top_k: int, filter_criteria: Dict = None) -> List[Dict]:
        """Search in Qdrant.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            filter_criteria: Filter criteria for search
            
        Returns:
            List of relevant documents
        """
        try:
            # Create search filter if provided
            search_filter = None
            if filter_criteria:
                # Convert filter criteria to Qdrant filter
                # This is a simplified implementation
                search_filter = qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key=f"metadata.{key}",
                            match=qdrant_models.MatchValue(value=value)
                        )
                        for key, value in filter_criteria.items()
                    ]
                )
            
            # Search in Qdrant
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                query_filter=search_filter
            )
            
            # Process results
            results = []
            for hit in search_result:
                metadata = json.loads(hit.payload.get("metadata", "{}"))
                results.append({
                    "text": hit.payload.get("text"),
                    "source": metadata.get("source"),
                    "relevance": hit.score,
                    "metadata": metadata
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching in Qdrant: {e}")
            return []
    
    def _search_in_pinecone(self, query_embedding: List[float], top_k: int, filter_criteria: Dict = None) -> List[Dict]:
        """Search in Pinecone.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            filter_criteria: Filter criteria for search
            
        Returns:
            List of relevant documents
        """
        try:
            # Search in Pinecone
            search_result = self.client.query(
                vector=query_embedding,
                top_k=top_k,
                filter=filter_criteria,
                include_metadata=True
            )
            
            # Process results
            results = []
            for match in search_result.matches:
                results.append({
                    "text": match.metadata.get("text"),
                    "source": match.metadata.get("source"),
                    "relevance": match.score,
                    "metadata": {
                        "source": match.metadata.get("source"),
                        "chunk_index": match.metadata.get("chunk_index"),
                        "total_chunks": match.metadata.get("total_chunks")
                    }
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching in Pinecone: {e}")
            return []
    
    def _search_in_weaviate(self, query_embedding: List[float], top_k: int, filter_criteria: Dict = None) -> List[Dict]:
        """Search in Weaviate.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            filter_criteria: Filter criteria for search
            
        Returns:
            List of relevant documents
        """
        try:
            # Get class name
            class_name = self.collection_name.title().replace("_", "")
            
            # Create filter if provided
            where_filter = None
            if filter_criteria:
                # Convert filter criteria to Weaviate where filter
                # This is a simplified implementation
                where_filter = {
                    "path": list(filter_criteria.keys())[0],
                    "operator": "Equal",
                    "valueString": list(filter_criteria.values())[0]
                }
            
            # Search in Weaviate
            search_result = (
                self.client.query
                .get(class_name, ["text", "source", "metadata"])
                .with_near_vector({
                    "vector": query_embedding
                })
                .with_limit(top_k)
            )
            
            if where_filter:
                search_result = search_result.with_where(where_filter)
            
            # Execute query
            result = search_result.do()
            
            # Process results
            results = []
            for hit in result["data"]["Get"][class_name]:
                try:
                    metadata = json.loads(hit.get("metadata", "{}"))
                except:
                    metadata = {}
                
                results.append({
                    "text": hit.get("text"),
                    "source": hit.get("source"),
                    "relevance": hit.get("_additional", {}).get("certainty", 0.0),
                    "metadata": metadata
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching in Weaviate: {e}")
            return []
    
    def optimize_index(self) -> bool:
        """Optimize the vector index for better performance.
        
        Returns:
            True if successful, False otherwise
        """
        if self.client is None:
            logger.error("Vector database client not initialized")
            return False
        
        try:
            if self.db_type == "qdrant":
                # Qdrant doesn't require explicit optimization
                logger.info("Qdrant index optimization not required")
                return True
            
            elif self.db_type == "pinecone":
                # Pinecone doesn't have explicit optimization
                logger.info("Pinecone index optimization not required")
                return True
            
            elif self.db_type == "weaviate":
                # Trigger Weaviate index optimization
                self.client.schema.get()  # This refreshes the schema
                logger.info("Weaviate index optimization triggered")
                return True
            
            else:
                logger.error(f"Unsupported vector database type: {self.db_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error optimizing vector index: {e}")
            return False
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self.embedding_cache = {}
        logger.info("Embedding cache cleared")
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the vector collection.
        
        Returns:
            Statistics about the collection
        """
        if self.client is None:
            logger.error("Vector database client not initialized")
            return {"error": "Vector database client not initialized"}
        
        try:
            if self.db_type == "qdrant":
                # Get collection info from Qdrant
                collection_info = self.client.get_collection(self.collection_name)
                return {
                    "vector_count": collection_info.vectors_count,
                    "status": str(collection_info.status),
                    "dimension": collection_info.config.params.vectors.size
                }
            
            elif self.db_type == "pinecone":
                # Get index stats from Pinecone
                stats = self.client.describe_index_stats()
                return {
                    "vector_count": stats.get("total_vector_count", 0),
                    "dimension": self.embedding_dim,
                    "namespaces": stats.get("namespaces", {})
                }
            
            elif self.db_type == "weaviate":
                # Get class stats from Weaviate
                class_name = self.collection_name.title().replace("_", "")
                stats = self.client.get_meta()
                return {
                    "vector_count": next((c["objectCount"] for c in stats["classes"] if c["class"] == class_name), 0),
                    "status": "ready",
                    "dimension": self.embedding_dim
                }
            
            else:
                logger.error(f"Unsupported vector database type: {self.db_type}")
                return {"error": f"Unsupported vector database type: {self.db_type}"}
                
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}
    
    def process_query_for_optimization(self, query: str) -> str:
        """Process a query to optimize it for vector search.
        
        Args:
            query: Original query
            
        Returns:
            Optimized query
        """
        # Remove stop words
        stop_words = ["и", "в", "на", "с", "по", "к", "у", "о", "из", "что", "как", "а", "the", "a", "an", "in", "on", "at", "with", "by", "to", "for", "of", "from"]
        tokens = query.lower().split()
        filtered_tokens = [token for token in tokens if token not in stop_words]
        
        # If query becomes too short after filtering, use original
        if len(filtered_tokens) < 2:
            return query
        
        # Add context keywords based on the query content
        context_keywords = []
        
        # Check for keywords related to project topics
        if any(word in tokens for word in ["проект", "lucky", "train", "поезд", "project"]):
            context_keywords.append("Lucky Train проект")
        
        if any(word in tokens for word in ["токен", "token", "ltt", "стоимость", "цена", "price"]):
            context_keywords.append("LTT токен")
        
        if any(word in tokens for word in ["метавселенная", "metaverse", "виртуальный", "мир", "virtual", "world"]):
            context_keywords.append("метавселенная")
        
        if any(word in tokens for word in ["блокчейн", "blockchain", "ton", "тон"]):
            context_keywords.append("блокчейн TON")
        
        # Combine original query with context keywords
        optimized_query = " ".join(filtered_tokens)
        if context_keywords:
            optimized_query += " " + " ".join(context_keywords)
        
        return optimized_query

# Example usage
if __name__ == "__main__":
    # Initialize vector database handler
    vector_db = VectorDBHandler()
    
    # Load knowledge base
    with open("knowledge_base/lucky_train.json", "r", encoding="utf-8") as f:
        knowledge_base = json.load(f)
    
    success = vector_db.load_knowledge_base({"lucky_train": knowledge_base})
    print(f"Knowledge base loaded: {success}")
    
    # Optimize index
    success = vector_db.optimize_index()
    print(f"Index optimized: {success}")
    
    # Search
    results = vector_db.search("Расскажи о проекте Lucky Train")
    print(f"Found {len(results)} results") 