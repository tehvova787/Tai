"""
Tests for Vector Database functionality of Lucky Train AI Assistant.

This module contains unit tests for the vector database components of the system.
"""

import os
import sys
import json
import unittest
from unittest.mock import patch, MagicMock, mock_open
import tempfile
import numpy as np

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vector_db import VectorDBHandler

class TestVectorDB(unittest.TestCase):
    """Test cases for vector database functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary config file
        self.temp_config = tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.json')
        self.temp_config.write(json.dumps({
            "vector_db_settings": {
                "db_type": "qdrant",
                "collection_name": "test_collection",
                "embedding_model": {
                    "type": "sentence_transformers",
                    "name": "all-MiniLM-L6-v2"
                },
                "qdrant_path": "./data/test_qdrant"
            }
        }))
        self.temp_config.close()
        
        # Mock the embedding model
        self.mock_model = MagicMock()
        self.mock_model.encode.return_value = np.array([0.1, 0.2, 0.3, 0.4])
        self.mock_model.get_sentence_embedding_dimension.return_value = 4
        
        # Mock Qdrant client
        self.mock_qdrant_client = MagicMock()
        self.mock_collection_info = MagicMock()
        self.mock_collection_info.vectors_count = 100
        self.mock_collection_info.status = "green"
        self.mock_collection_info.config.params.vectors.size = 4
        self.mock_qdrant_client.get_collection.return_value = self.mock_collection_info
        
        # Mock search results
        mock_hit1 = MagicMock()
        mock_hit1.payload = {"text": "Test text 1", "metadata": '{"source": "test/path1"}'}
        mock_hit1.score = 0.95
        
        mock_hit2 = MagicMock()
        mock_hit2.payload = {"text": "Test text 2", "metadata": '{"source": "test/path2"}'}
        mock_hit2.score = 0.85
        
        self.mock_qdrant_client.search.return_value = [mock_hit1, mock_hit2]
        
        # Configure patches
        self.sentence_transformers_patcher = patch('src.vector_db.SentenceTransformer', return_value=self.mock_model)
        self.qdrant_client_patcher = patch('src.vector_db.QdrantClient', return_value=self.mock_qdrant_client)
        self.os_makedirs_patcher = patch('os.makedirs')
        
        # Start patches
        self.sentence_transformers_patcher.start()
        self.qdrant_client_patcher.start()
        self.os_makedirs_patcher.start()
        
        # Create the vector database handler
        self.vector_db = VectorDBHandler(self.temp_config.name)
    
    def tearDown(self):
        """Clean up after tests."""
        # Stop patches
        self.sentence_transformers_patcher.stop()
        self.qdrant_client_patcher.stop()
        self.os_makedirs_patcher.stop()
        
        # Remove temporary config file
        os.unlink(self.temp_config.name)
    
    def test_initialization(self):
        """Test initialization of vector database handler."""
        self.assertEqual(self.vector_db.db_type, "qdrant")
        self.assertEqual(self.vector_db.collection_name, "test_collection")
        self.assertEqual(self.vector_db.embedding_dim, 4)
        self.assertIsNotNone(self.vector_db.client)
        self.assertIsNotNone(self.vector_db.embedding_model)
    
    def test_get_embedding(self):
        """Test embedding generation."""
        embedding = self.vector_db._get_embedding("Test text")
        
        # Verify that the model's encode method was called
        self.mock_model.encode.assert_called_once_with("Test text")
        
        # Verify that the embedding was converted to list and returned
        self.assertIsInstance(embedding, list)
        self.assertEqual(len(embedding), 4)
    
    def test_process_text_chunks(self):
        """Test text chunking functionality."""
        text = "This is the first sentence. This is the second sentence. This is the third sentence."
        chunks = self.vector_db._process_text_chunks(text, chunk_size=50, overlap=10)
        
        # Verify that the text was split into chunks
        self.assertGreaterEqual(len(chunks), 1)
        
        # Verify that each chunk is a string and not empty
        for chunk in chunks:
            self.assertIsInstance(chunk, str)
            self.assertGreater(len(chunk), 0)
    
    def test_process_knowledge_base_section(self):
        """Test processing of knowledge base sections."""
        # Test processing a dictionary
        data_dict = {
            "key1": "Value 1",
            "key2": "Value with more than three words here"
        }
        chunks_dict = self.vector_db._process_knowledge_base_section("test_section", data_dict)
        self.assertGreaterEqual(len(chunks_dict), 1)
        
        # Test processing a list of strings
        data_list = ["Item 1", "Item 2", "Item with more words"]
        chunks_list = self.vector_db._process_knowledge_base_section("test_section", data_list)
        self.assertGreaterEqual(len(chunks_list), 1)
        
        # Test processing a string
        data_string = "This is a test string with more than three words"
        chunks_string = self.vector_db._process_knowledge_base_section("test_section", data_string)
        self.assertGreaterEqual(len(chunks_string), 1)
    
    def test_search(self):
        """Test search functionality."""
        results = self.vector_db.search("Test query", top_k=2)
        
        # Verify that search was called on the client
        self.mock_qdrant_client.search.assert_called_once()
        
        # Verify that the correct number of results were returned
        self.assertEqual(len(results), 2)
        
        # Verify the structure of the results
        self.assertEqual(results[0]["text"], "Test text 1")
        self.assertEqual(results[0]["source"], "test/path1")
        self.assertEqual(results[0]["relevance"], 0.95)
        
        self.assertEqual(results[1]["text"], "Test text 2")
        self.assertEqual(results[1]["source"], "test/path2")
        self.assertEqual(results[1]["relevance"], 0.85)
    
    def test_get_collection_stats(self):
        """Test collection statistics retrieval."""
        stats = self.vector_db.get_collection_stats()
        
        # Verify that get_collection was called on the client
        self.mock_qdrant_client.get_collection.assert_called_once_with("test_collection")
        
        # Verify the structure of the stats
        self.assertEqual(stats["vector_count"], 100)
        self.assertEqual(stats["status"], "green")
        self.assertEqual(stats["dimension"], 4)
    
    def test_load_knowledge_base(self):
        """Test loading knowledge base into vector database."""
        # Mock the store_chunks method
        self.vector_db._store_chunks_in_qdrant = MagicMock(return_value=True)
        
        # Test data
        knowledge_base = {
            "section1": {
                "subsection1": "Test content 1",
                "subsection2": "Test content 2"
            },
            "section2": ["Item 1", "Item 2", "Item 3"]
        }
        
        # Call the method
        result = self.vector_db.load_knowledge_base(knowledge_base)
        
        # Verify that the method returned True
        self.assertTrue(result)
        
        # Verify that store_chunks_in_qdrant was called
        self.vector_db._store_chunks_in_qdrant.assert_called_once()
        
        # Verify that chunks were passed to the method
        args, kwargs = self.vector_db._store_chunks_in_qdrant.call_args
        chunks = args[0]
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
    
    def test_process_query_for_optimization(self):
        """Test query optimization for vector search."""
        # Test with project-related query
        project_query = "расскажи о проекте lucky train"
        optimized_project_query = self.vector_db.process_query_for_optimization(project_query)
        self.assertIn("Lucky Train проект", optimized_project_query)
        
        # Test with token-related query
        token_query = "какая стоимость токена ltt"
        optimized_token_query = self.vector_db.process_query_for_optimization(token_query)
        self.assertIn("LTT токен", optimized_token_query)
        
        # Test with metaverse-related query
        metaverse_query = "что такое метавселенная в проекте"
        optimized_metaverse_query = self.vector_db.process_query_for_optimization(metaverse_query)
        self.assertIn("метавселенная", optimized_metaverse_query)
        
        # Test with blockchain-related query
        blockchain_query = "как работает блокчейн ton"
        optimized_blockchain_query = self.vector_db.process_query_for_optimization(blockchain_query)
        self.assertIn("блокчейн TON", optimized_blockchain_query)
        
        # Test with very short query
        short_query = "что это"
        optimized_short_query = self.vector_db.process_query_for_optimization(short_query)
        self.assertEqual(short_query, optimized_short_query)

if __name__ == '__main__':
    unittest.main() 