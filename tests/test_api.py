"""
Tests for API functionality of Lucky Train AI Assistant.

This module contains unit tests for the API components of the system.
"""

import os
import sys
import json
import unittest
from unittest.mock import patch, MagicMock
import tempfile
import time

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.web_interface import LuckyTrainWebInterface

class TestAPI(unittest.TestCase):
    """Test cases for API functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary config file
        self.temp_config = tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.json')
        self.temp_config.write(json.dumps({
            "language": "ru",
            "max_tokens": 100,
            "temperature": 0.7,
            "knowledge_base_path": "./knowledge_base",
            "supported_platforms": ["telegram", "website", "metaverse", "console"],
            "llm_settings": {
                "llm_model": "gpt-3.5-turbo",
                "streaming": True
            },
            "security_settings": {
                "api_key_required": True,
                "api_keys": ["test_api_key_123"]
            },
            "web_interface_settings": {
                "title": "Test Interface",
                "theme": "light"
            }
        }))
        self.temp_config.close()
        
        # Create a mock assistant
        self.mock_assistant = MagicMock()
        self.mock_assistant.handle_message.return_value = "Test response"
        self.mock_assistant.find_relevant_information.return_value = [{"text": "Test info", "source": "test"}]
        self.mock_assistant._determine_topic.return_value = "project"
        self.mock_assistant._detect_language.return_value = "ru"
        self.mock_assistant.supported_languages = {"ru": "Russian", "en": "English"}
        self.mock_assistant.analytics_manager = MagicMock()
        
        # Create a mock openai client
        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_choice.delta.content = "Test content"
        mock_completion.choices = [mock_choice]
        
        mock_openai_client = MagicMock()
        mock_openai_client.chat.completions.create.return_value = [mock_completion]
        
        # Patch dependencies
        patcher1 = patch('src.web_interface.LuckyTrainAssistant', return_value=self.mock_assistant)
        patcher2 = patch.dict('os.environ', {"JWT_SECRET_KEY": "test_secret_key"})
        
        self.addCleanup(patcher1.stop)
        self.addCleanup(patcher2.stop)
        
        patcher1.start()
        patcher2.start()
        
        # Set up the Flask test client
        self.web_interface = LuckyTrainWebInterface(self.temp_config.name)
        self.app = self.web_interface.app.test_client()
        self.mock_assistant.openai_client = mock_openai_client
    
    def tearDown(self):
        """Clean up after tests."""
        os.unlink(self.temp_config.name)
    
    def test_chat_api_without_auth(self):
        """Test chat API without authentication."""
        response = self.app.post('/api/chat', json={
            "message": "Test message"
        })
        
        self.assertEqual(response.status_code, 401)
        data = json.loads(response.data)
        self.assertIn("error", data)
    
    def test_chat_api_with_auth(self):
        """Test chat API with authentication."""
        response = self.app.post('/api/chat', 
                               json={"message": "Test message"},
                               headers={"X-API-Key": "test_api_key_123"})
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("response", data)
        self.assertEqual(data["response"], "Test response")
    
    def test_api_token_endpoint(self):
        """Test API token generation endpoint."""
        response = self.app.post('/api/token', 
                               json={"api_key": "test_api_key_123"})
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("token", data)
        self.assertIn("user_id", data)
        self.assertIn("role", data)
        self.assertIn("expires_in", data)
    
    def test_api_token_invalid_key(self):
        """Test API token with invalid API key."""
        response = self.app.post('/api/token', 
                               json={"api_key": "invalid_key"})
        
        self.assertEqual(response.status_code, 401)
        data = json.loads(response.data)
        self.assertIn("error", data)
    
    def test_jwt_secured_endpoint_missing_auth(self):
        """Test JWT-secured endpoint with missing authorization header."""
        response = self.app.post('/api/v1/chat', 
                               json={"message": "Test message"})
        
        self.assertEqual(response.status_code, 401)
        data = json.loads(response.data)
        self.assertIn("error", data)
        self.assertEqual(data["error"], "Authorization header is missing")
    
    def test_jwt_secured_endpoint_invalid_format(self):
        """Test JWT-secured endpoint with invalid authorization format."""
        response = self.app.post('/api/v1/chat', 
                               json={"message": "Test message"},
                               headers={"Authorization": "InvalidFormat token123"})
        
        self.assertEqual(response.status_code, 401)
        data = json.loads(response.data)
        self.assertIn("error", data)
        self.assertEqual(data["error"], "Invalid Authorization header format")
    
    def test_jwt_secured_endpoint_valid_token(self):
        """Test JWT-secured endpoint with valid token."""
        # First get a token
        token_response = self.app.post('/api/token', 
                                     json={"api_key": "test_api_key_123"})
        token_data = json.loads(token_response.data)
        token = token_data["token"]
        
        # Use the token to access a secured endpoint
        response = self.app.post('/api/v1/chat', 
                               json={"message": "Test message"},
                               headers={"Authorization": f"Bearer {token}"})
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("response", data)
        self.assertEqual(data["response"], "Test response")
    
    def test_blockchain_info_api(self):
        """Test blockchain info API endpoint."""
        # Mock the blockchain integration
        with patch('src.blockchain_integration.TONBlockchainIntegration') as mock_blockchain:
            mock_instance = mock_blockchain.return_value
            mock_instance.get_blockchain_info.return_value = {
                "name": "TON",
                "active_validators": 345,
                "current_block": 20123456
            }
            
            # First get a token
            token_response = self.app.post('/api/token', 
                                         json={"api_key": "test_api_key_123"})
            token_data = json.loads(token_response.data)
            token = token_data["token"]
            
            # Use the token to access the blockchain info endpoint
            response = self.app.get('/api/v1/blockchain/info', 
                                  headers={"Authorization": f"Bearer {token}"})
            
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertEqual(data["name"], "TON")
            self.assertEqual(data["active_validators"], 345)
            self.assertEqual(data["current_block"], 20123456)
    
    def test_token_info_api(self):
        """Test token info API endpoint."""
        # Mock the blockchain integration
        with patch('src.blockchain_integration.TONBlockchainIntegration') as mock_blockchain:
            mock_instance = mock_blockchain.return_value
            mock_instance.get_market_price.return_value = {
                "symbol": "LTT",
                "name": "Lucky Train Token",
                "price_usd": 0.15
            }
            
            # First get a token
            token_response = self.app.post('/api/token', 
                                         json={"api_key": "test_api_key_123"})
            token_data = json.loads(token_response.data)
            token = token_data["token"]
            
            # Use the token to access the token info endpoint
            response = self.app.get('/api/v1/blockchain/token-info?token=LTT', 
                                  headers={"Authorization": f"Bearer {token}"})
            
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertEqual(data["symbol"], "LTT")
            self.assertEqual(data["name"], "Lucky Train Token")
            self.assertEqual(data["price_usd"], 0.15)
            
            # Verify that the correct token was requested
            mock_instance.get_market_price.assert_called_once_with("LTT")

if __name__ == '__main__':
    unittest.main() 