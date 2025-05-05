"""
Tests for JWT Authentication functionality of Lucky Train AI Assistant.

This module contains unit tests for the JWT authentication components of the system.
"""

import os
import sys
import json
import unittest
from unittest.mock import patch, MagicMock
import tempfile
import time
import jwt
from datetime import datetime, timedelta

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.security import SecurityManager, JWTAuthMiddleware
from flask import Flask, request, jsonify, g

class TestJWTAuth(unittest.TestCase):
    """Test cases for JWT authentication functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary config file
        self.temp_config = tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.json')
        self.temp_config.write(json.dumps({
            "security_settings": {
                "jwt_secret": "test_jwt_secret",
                "jwt_expiry_hours": 24,
                "rate_limit": 60
            }
        }))
        self.temp_config.close()
        
        # Create a security manager with the config
        with open(self.temp_config.name, 'r') as f:
            config = json.load(f)
        
        self.security_manager = SecurityManager(config["security_settings"])
        
        # Create a Flask app for testing
        self.app = Flask(__name__)
        self.app.config['TESTING'] = True
        
        # Set up the app context
        self.app_context = self.app.app_context()
        self.app_context.push()
        
        # Configure middleware
        self.jwt_middleware = JWTAuthMiddleware(
            app=self.app,
            security_manager=self.security_manager,
            exempt_routes=['/exempt'],
            exempt_prefixes=['/public/']
        )
        
        # Configure routes for testing
        @self.app.route('/protected')
        def protected_route():
            return jsonify({"user_id": g.user_id, "role": g.user_role})
            
        @self.app.route('/exempt')
        def exempt_route():
            return jsonify({"message": "This route is exempt from JWT auth"})
            
        @self.app.route('/public/resource')
        def public_resource():
            return jsonify({"message": "This is a public resource"})
        
        # Create a test client
        self.client = self.app.test_client()
    
    def tearDown(self):
        """Clean up after tests."""
        self.app_context.pop()
        os.unlink(self.temp_config.name)
    
    def test_generate_jwt_token(self):
        """Test JWT token generation."""
        # Generate a token
        token = self.security_manager.generate_jwt_token("test_user", "admin")
        
        # Verify the token is a string
        self.assertIsInstance(token, str)
        
        # Decode the token to verify payload
        payload = jwt.decode(token, "test_jwt_secret", algorithms=["HS256"])
        
        self.assertEqual(payload["sub"], "test_user")
        self.assertEqual(payload["role"], "admin")
        self.assertIn("iat", payload)
        self.assertIn("exp", payload)
    
    def test_verify_jwt_token_valid(self):
        """Test verification of a valid JWT token."""
        # Generate a token
        token = self.security_manager.generate_jwt_token("test_user", "admin")
        
        # Verify the token
        result = self.security_manager.verify_jwt_token(token)
        
        self.assertTrue(result["valid"])
        self.assertIn("payload", result)
        self.assertEqual(result["payload"]["sub"], "test_user")
        self.assertEqual(result["payload"]["role"], "admin")
    
    def test_verify_jwt_token_expired(self):
        """Test verification of an expired JWT token."""
        # Generate a token that is already expired
        now = datetime.utcnow()
        expiry = now - timedelta(hours=1)  # Expired 1 hour ago
        
        # Create payload
        payload = {
            "sub": "test_user",
            "role": "admin",
            "iat": now,
            "exp": expiry
        }
        
        # Encode token
        token = jwt.encode(payload, "test_jwt_secret", algorithm="HS256")
        
        # Verify the token
        result = self.security_manager.verify_jwt_token(token)
        
        self.assertFalse(result["valid"])
        self.assertEqual(result["error"], "Token expired")
    
    def test_verify_jwt_token_invalid(self):
        """Test verification of an invalid JWT token."""
        # Create an invalid token
        token = "invalid.token.format"
        
        # Verify the token
        result = self.security_manager.verify_jwt_token(token)
        
        self.assertFalse(result["valid"])
        self.assertIn("error", result)
    
    def test_middleware_protected_route_no_auth(self):
        """Test accessing a protected route without authentication."""
        response = self.client.get('/protected')
        
        self.assertEqual(response.status_code, 401)
        data = json.loads(response.data)
        self.assertIn("error", data)
        self.assertEqual(data["error"], "Authorization header is missing")
    
    def test_middleware_protected_route_invalid_format(self):
        """Test accessing a protected route with invalid auth format."""
        response = self.client.get(
            '/protected',
            headers={"Authorization": "InvalidFormat token123"}
        )
        
        self.assertEqual(response.status_code, 401)
        data = json.loads(response.data)
        self.assertIn("error", data)
        self.assertEqual(data["error"], "Invalid Authorization header format")
    
    def test_middleware_protected_route_valid_token(self):
        """Test accessing a protected route with a valid token."""
        # Generate a token
        token = self.security_manager.generate_jwt_token("test_user", "admin")
        
        # Access protected route
        response = self.client.get(
            '/protected',
            headers={"Authorization": f"Bearer {token}"}
        )
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["user_id"], "test_user")
        self.assertEqual(data["role"], "admin")
    
    def test_middleware_exempt_route(self):
        """Test accessing an exempt route without authentication."""
        response = self.client.get('/exempt')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["message"], "This route is exempt from JWT auth")
    
    def test_middleware_exempt_prefix(self):
        """Test accessing a route with exempt prefix without authentication."""
        response = self.client.get('/public/resource')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["message"], "This is a public resource")

if __name__ == '__main__':
    unittest.main() 