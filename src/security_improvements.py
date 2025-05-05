"""
Security Improvements Module for Lucky Train AI Assistant

This module provides enhanced security features:
- Secure secrets management
- Database connection pool management
- Proper key rotation
- Enhanced input validation
"""

import os
import logging
import json
import time
import secrets
import hashlib
import hmac
import base64
from typing import Dict, Any, Optional, List, Union, Tuple
from contextlib import contextmanager
from functools import wraps
import threading
import sys
import uuid
from datetime import datetime, timedelta
import re

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SecretManager:
    """Manager for securely handling application secrets."""
    
    def __init__(self, config: Dict = None):
        """Initialize the secret manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.secrets_file = self.config.get("secrets_file", os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'secrets.json'))
        self.master_key_env = self.config.get("master_key_env", "LUCKYTRAINAI_MASTER_KEY")
        self.secrets_cache = {}
        self.last_refresh = 0
        self.refresh_interval = self.config.get("refresh_interval", 300)  # 5 minutes
        self.auto_create = self.config.get("auto_create", True)
        
        # Initialize encryption key
        self.master_key = self._get_master_key()
        if not self.master_key:
            logger.warning("No master key found. Secure storage not available.")
        
        # Load initial secrets
        self._load_secrets()
    
    def _get_master_key(self) -> Optional[bytes]:
        """Get or create the master encryption key.
        
        Returns:
            Master key as bytes or None if not available
        """
        # Try to get from environment
        key_str = os.environ.get(self.master_key_env)
        
        if key_str:
            try:
                # Decode from base64
                return base64.urlsafe_b64decode(key_str)
            except Exception as e:
                logger.error(f"Error decoding master key: {e}")
        
        # If auto-create is enabled, generate a new key
        if self.auto_create:
            logger.warning("Generating new master key. Store this in environment variable for production.")
            key = Fernet.generate_key()
            
            # Print to console for initial setup
            print(f"Generated master key: {key.decode()}")
            print(f"Set this as environment variable {self.master_key_env}")
            
            return key
        
        return None
    
    def _get_fernet(self) -> Optional[Fernet]:
        """Get a Fernet encryption instance.
        
        Returns:
            Fernet instance or None if master key not available
        """
        if not self.master_key:
            return None
        
        return Fernet(self.master_key)
    
    def _load_secrets(self) -> None:
        """Load secrets from the secrets file."""
        # Skip if no master key
        if not self.master_key:
            return
        
        # Create file if it doesn't exist
        if not os.path.exists(self.secrets_file):
            if self.auto_create:
                # Create directory if needed
                os.makedirs(os.path.dirname(self.secrets_file), exist_ok=True)
                
                # Create empty secrets file
                with open(self.secrets_file, 'w') as f:
                    json.dump({}, f)
                
                logger.info(f"Created new secrets file: {self.secrets_file}")
            else:
                logger.warning(f"Secrets file not found: {self.secrets_file}")
            
            self.secrets_cache = {}
            self.last_refresh = time.time()
            return
        
        try:
            # Read encrypted file
            with open(self.secrets_file, 'r') as f:
                data = json.load(f)
            
            # Decrypt secrets
            fernet = self._get_fernet()
            if not fernet:
                logger.error("Cannot decrypt secrets: No master key")
                return
            
            decrypted = {}
            for key, encrypted_value in data.items():
                try:
                    value_bytes = base64.urlsafe_b64decode(encrypted_value)
                    decrypted_bytes = fernet.decrypt(value_bytes)
                    decrypted[key] = decrypted_bytes.decode('utf-8')
                except Exception as e:
                    logger.error(f"Error decrypting secret {key}: {e}")
            
            self.secrets_cache = decrypted
            self.last_refresh = time.time()
            logger.info(f"Loaded {len(decrypted)} secrets from {self.secrets_file}")
        
        except Exception as e:
            logger.error(f"Error loading secrets: {e}")
    
    def _save_secrets(self) -> None:
        """Save secrets to the secrets file."""
        # Skip if no master key
        if not self.master_key:
            return
        
        try:
            # Create directory if needed
            os.makedirs(os.path.dirname(self.secrets_file), exist_ok=True)
            
            # Encrypt secrets
            fernet = self._get_fernet()
            if not fernet:
                logger.error("Cannot encrypt secrets: No master key")
                return
            
            encrypted = {}
            for key, value in self.secrets_cache.items():
                try:
                    # Convert to bytes if necessary
                    if isinstance(value, str):
                        value_bytes = value.encode('utf-8')
                    else:
                        value_bytes = str(value).encode('utf-8')
                    
                    encrypted_bytes = fernet.encrypt(value_bytes)
                    encrypted[key] = base64.urlsafe_b64encode(encrypted_bytes).decode('utf-8')
                except Exception as e:
                    logger.error(f"Error encrypting secret {key}: {e}")
            
            # Write to file
            with open(self.secrets_file, 'w') as f:
                json.dump(encrypted, f)
            
            logger.info(f"Saved {len(encrypted)} secrets to {self.secrets_file}")
        
        except Exception as e:
            logger.error(f"Error saving secrets: {e}")
    
    def refresh(self) -> None:
        """Refresh secrets from the secrets file."""
        # Check if refresh is needed
        if time.time() - self.last_refresh < self.refresh_interval:
            return
        
        self._load_secrets()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a secret.
        
        Args:
            key: Secret key
            default: Default value if not found
            
        Returns:
            Secret value or default
        """
        # Refresh if needed
        self.refresh()
        
        return self.secrets_cache.get(key, default)
    
    def set(self, key: str, value: str) -> bool:
        """Set a secret.
        
        Args:
            key: Secret key
            value: Secret value
            
        Returns:
            True if successful, False otherwise
        """
        # Skip if no master key
        if not self.master_key:
            logger.error("Cannot set secret: No master key")
            return False
        
        try:
            # Update cache
            self.secrets_cache[key] = value
            
            # Save to file
            self._save_secrets()
            
            return True
        except Exception as e:
            logger.error(f"Error setting secret {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete a secret.
        
        Args:
            key: Secret key
            
        Returns:
            True if successful, False otherwise
        """
        # Skip if not in cache
        if key not in self.secrets_cache:
            return False
        
        try:
            # Update cache
            del self.secrets_cache[key]
            
            # Save to file
            self._save_secrets()
            
            return True
        except Exception as e:
            logger.error(f"Error deleting secret {key}: {e}")
            return False
    
    def rotate_key(self) -> bool:
        """Rotate the master encryption key.
        
        Returns:
            True if successful, False otherwise
        """
        # Skip if no master key
        if not self.master_key:
            logger.error("Cannot rotate key: No master key")
            return False
        
        try:
            # Generate new key
            new_key = Fernet.generate_key()
            
            # Re-encrypt all secrets with new key
            old_fernet = self._get_fernet()
            new_fernet = Fernet(new_key)
            
            encrypted = {}
            for key, value in self.secrets_cache.items():
                try:
                    # Convert to bytes if necessary
                    if isinstance(value, str):
                        value_bytes = value.encode('utf-8')
                    else:
                        value_bytes = str(value).encode('utf-8')
                    
                    encrypted_bytes = new_fernet.encrypt(value_bytes)
                    encrypted[key] = base64.urlsafe_b64encode(encrypted_bytes).decode('utf-8')
                except Exception as e:
                    logger.error(f"Error re-encrypting secret {key}: {e}")
            
            # Write to file
            with open(self.secrets_file, 'w') as f:
                json.dump(encrypted, f)
            
            # Update master key
            self.master_key = new_key
            
            # Print new key for updating environment variable
            print(f"Rotated master key. New key: {new_key.decode()}")
            print(f"Update your environment variable {self.master_key_env}")
            
            logger.info(f"Rotated master key and re-encrypted {len(encrypted)} secrets")
            
            return True
        except Exception as e:
            logger.error(f"Error rotating master key: {e}")
            return False

class ConnectionPool:
    """Database connection pooling manager."""
    
    def __init__(self, db_type: str, config: Dict = None):
        """Initialize the connection pool.
        
        Args:
            db_type: Type of database (sqlite, mysql, postgres, etc.)
            config: Configuration dictionary
        """
        self.db_type = db_type
        self.config = config or {}
        self.min_connections = self.config.get("min_connections", 1)
        self.max_connections = self.config.get("max_connections", 10)
        self.connection_timeout = self.config.get("connection_timeout", 30)  # seconds
        self.idle_timeout = self.config.get("idle_timeout", 600)  # 10 minutes
        
        # Pool state
        self.pool = []  # [{"connection": conn, "last_used": timestamp, "in_use": bool}]
        self.lock = threading.RLock()
        
        # Initialize pool
        for _ in range(self.min_connections):
            self._add_connection()
    
    def _create_connection(self):
        """Create a new database connection.
        
        Returns:
            Database connection
        """
        if self.db_type == "sqlite":
            import sqlite3
            db_path = self.config.get("db_path", ":memory:")
            conn = sqlite3.connect(db_path, timeout=self.connection_timeout)
            conn.row_factory = sqlite3.Row
            return conn
        
        elif self.db_type == "mysql":
            import pymysql
            conn = pymysql.connect(
                host=self.config.get("host", "localhost"),
                port=int(self.config.get("port", 3306)),
                user=self.config.get("user", "root"),
                password=self.config.get("password", ""),
                database=self.config.get("database", ""),
                charset=self.config.get("charset", "utf8mb4"),
                connect_timeout=self.connection_timeout,
                cursorclass=pymysql.cursors.DictCursor
            )
            return conn
        
        elif self.db_type == "postgres":
            import psycopg2
            import psycopg2.extras
            conn = psycopg2.connect(
                host=self.config.get("host", "localhost"),
                port=int(self.config.get("port", 5432)),
                user=self.config.get("user", "postgres"),
                password=self.config.get("password", ""),
                dbname=self.config.get("database", ""),
                connect_timeout=self.connection_timeout
            )
            return conn
        
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")
    
    def _add_connection(self) -> bool:
        """Add a new connection to the pool.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = self._create_connection()
            self.pool.append({
                "connection": conn,
                "last_used": time.time(),
                "in_use": False
            })
            return True
        except Exception as e:
            logger.error(f"Error creating database connection: {e}")
            return False
    
    def _close_connection(self, conn_info: Dict) -> None:
        """Close a database connection.
        
        Args:
            conn_info: Connection info dictionary
        """
        try:
            conn_info["connection"].close()
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")
    
    def _cleanup_idle_connections(self) -> None:
        """Close idle connections exceeding the minimum pool size."""
        with self.lock:
            now = time.time()
            
            # Sort by last used time (oldest first)
            idle_connections = sorted(
                [c for c in self.pool if not c["in_use"]],
                key=lambda c: c["last_used"]
            )
            
            # Keep min_connections, close others that are idle too long
            connections_to_keep = max(self.min_connections, len(self.pool) - len(idle_connections))
            
            for i, conn_info in enumerate(idle_connections):
                if i >= connections_to_keep or (now - conn_info["last_used"] > self.idle_timeout):
                    self._close_connection(conn_info)
                    self.pool.remove(conn_info)
    
    def get_connection(self):
        """Get a connection from the pool.
        
        Returns:
            Database connection
        """
        with self.lock:
            # Try to find an unused connection
            for conn_info in self.pool:
                if not conn_info["in_use"]:
                    conn_info["in_use"] = True
                    conn_info["last_used"] = time.time()
                    return conn_info["connection"]
            
            # If pool is not at max size, add a new connection
            if len(self.pool) < self.max_connections:
                if self._add_connection():
                    conn_info = self.pool[-1]
                    conn_info["in_use"] = True
                    return conn_info["connection"]
            
            # Otherwise wait for a connection to become available
            raise ValueError("Connection pool exhausted")
    
    def release_connection(self, connection) -> None:
        """Release a connection back to the pool.
        
        Args:
            connection: Database connection
        """
        with self.lock:
            for conn_info in self.pool:
                if conn_info["connection"] is connection:
                    conn_info["in_use"] = False
                    conn_info["last_used"] = time.time()
                    break
            
            # Clean up idle connections
            self._cleanup_idle_connections()
    
    @contextmanager
    def connection(self):
        """Context manager for getting and releasing a connection.
        
        Yields:
            Database connection
        """
        connection = None
        try:
            connection = self.get_connection()
            yield connection
        finally:
            if connection:
                self.release_connection(connection)
    
    def close_all(self) -> None:
        """Close all connections in the pool."""
        with self.lock:
            for conn_info in self.pool:
                self._close_connection(conn_info)
            
            self.pool = []

class InputValidator:
    """Input validation utilities."""
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate an email address.
        
        Args:
            email: Email to validate
            
        Returns:
            True if valid, False otherwise
        """
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def validate_username(username: str) -> Tuple[bool, str]:
        """Validate a username.
        
        Args:
            username: Username to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if len(username) < 3:
            return False, "Username must be at least 3 characters long"
        
        if len(username) > 30:
            return False, "Username must be at most 30 characters long"
        
        if not re.match(r'^[a-zA-Z0-9_.-]+$', username):
            return False, "Username can only contain letters, numbers, underscores, dots, and hyphens"
        
        return True, ""
    
    @staticmethod
    def validate_password(password: str) -> Tuple[bool, str]:
        """Validate a password.
        
        Args:
            password: Password to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if len(password) < 8:
            return False, "Password must be at least 8 characters long"
        
        if not re.search(r'[A-Z]', password):
            return False, "Password must contain at least one uppercase letter"
        
        if not re.search(r'[a-z]', password):
            return False, "Password must contain at least one lowercase letter"
        
        if not re.search(r'\d', password):
            return False, "Password must contain at least one digit"
        
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            return False, "Password must contain at least one special character"
        
        return True, ""
    
    @staticmethod
    def sanitize_input(input_str: str) -> str:
        """Sanitize user input to prevent injection attacks.
        
        Args:
            input_str: Input string to sanitize
            
        Returns:
            Sanitized string
        """
        # Remove potential script tags
        sanitized = re.sub(r'<script.*?>.*?</script>', '', input_str, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove other potentially dangerous HTML tags
        sanitized = re.sub(r'<[a-z].*?>', '', sanitized, flags=re.DOTALL | re.IGNORECASE)
        
        # Replace multiple spaces with a single space
        sanitized = re.sub(r'\s+', ' ', sanitized)
        
        return sanitized.strip()
    
    @staticmethod
    def validate_json_input(json_str: str) -> Tuple[bool, Union[Dict, str]]:
        """Validate and parse JSON input.
        
        Args:
            json_str: JSON string to validate
            
        Returns:
            Tuple of (is_valid, parsed_json or error_message)
        """
        try:
            parsed = json.loads(json_str)
            return True, parsed
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {str(e)}"
    
    @staticmethod
    def validate_api_key(api_key: str) -> bool:
        """Validate an API key format.
        
        Args:
            api_key: API key to validate
            
        Returns:
            True if valid format, False otherwise
        """
        # Check prefix and length
        if not api_key.startswith("lt_"):
            return False
        
        if len(api_key) != 35:  # "lt_" + 32 chars
            return False
        
        # Check the remaining characters are hexadecimal
        key_part = api_key[3:]
        return bool(re.match(r'^[0-9a-f]{32}$', key_part))

# Key rotation utilities
def rotate_jwt_secret(secret_manager: SecretManager) -> bool:
    """Rotate the JWT secret key.
    
    Args:
        secret_manager: Secret manager instance
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Generate new secret
        new_secret = secrets.token_hex(32)
        
        # Store with timestamp
        rotation_time = datetime.utcnow().isoformat()
        
        # Store previous secret for transition period
        old_secret = secret_manager.get("jwt_secret")
        if old_secret:
            secret_manager.set("jwt_secret_previous", old_secret)
            secret_manager.set("jwt_secret_rotation_time", rotation_time)
        
        # Set new secret
        success = secret_manager.set("jwt_secret", new_secret)
        
        if success:
            logger.info(f"Rotated JWT secret key at {rotation_time}")
        
        return success
    except Exception as e:
        logger.error(f"Error rotating JWT secret: {e}")
        return False

def verify_rotated_jwt(token: str, secret_manager: SecretManager) -> Dict:
    """Verify a JWT token, handling key rotation.
    
    Args:
        token: JWT token to verify
        secret_manager: Secret manager instance
        
    Returns:
        Verification result dictionary
    """
    import jwt as pyjwt
    
    # Get current secret
    current_secret = secret_manager.get("jwt_secret")
    if not current_secret:
        return {"valid": False, "error": "JWT secret not configured"}
    
    # Try to decode with current secret
    try:
        payload = pyjwt.decode(token, current_secret, algorithms=["HS256"])
        return {"valid": True, "payload": payload}
    except pyjwt.ExpiredSignatureError:
        return {"valid": False, "error": "Token expired"}
    except pyjwt.InvalidTokenError:
        # Try with previous secret if available
        previous_secret = secret_manager.get("jwt_secret_previous")
        if not previous_secret:
            return {"valid": False, "error": "Invalid token"}
        
        try:
            payload = pyjwt.decode(token, previous_secret, algorithms=["HS256"])
            
            # Check token rotation grace period
            rotation_time = secret_manager.get("jwt_secret_rotation_time")
            if rotation_time:
                rotation_dt = datetime.fromisoformat(rotation_time)
                grace_period = timedelta(days=1)  # 1 day grace period
                
                if datetime.utcnow() > rotation_dt + grace_period:
                    return {"valid": False, "error": "Token uses old key beyond grace period"}
            
            return {
                "valid": True, 
                "payload": payload, 
                "needs_refresh": True
            }
        except Exception:
            return {"valid": False, "error": "Invalid token"} 