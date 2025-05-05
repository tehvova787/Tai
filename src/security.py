"""
Security Module for Lucky Train AI Assistant

This module provides security features including:
- User authentication (password, token, JWT)
- Role-based authorization
- Input validation and sanitization
- Rate limiting
- Encryption utilities
"""

import logging
import time
import hashlib
import hmac
import os
import base64
import json
import re
import secrets
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from datetime import datetime, timedelta
import jwt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from dotenv import load_dotenv
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
TOKEN_EXPIRY_HOURS = 24
PASSWORD_MIN_LENGTH = 8
MAX_LOGIN_ATTEMPTS = 5
LOGIN_TIMEOUT_MINUTES = 30
DEFAULT_RATE_LIMIT = 60  # requests per minute

class SecurityManager:
    """Main security manager class."""
    
    def __init__(self, config: Dict = None):
        """Initialize the security manager.
        
        Args:
            config: Security configuration dictionary
        """
        self.config = config or {}
        
        # Get security settings
        self.secret_key = self.config.get("secret_key", os.getenv("SECURITY_SECRET_KEY")) or secrets.token_hex(32)
        self.jwt_secret = self.config.get("jwt_secret", os.getenv("JWT_SECRET_KEY")) or secrets.token_hex(32)
        self.pepper = self.config.get("pepper", os.getenv("PASSWORD_PEPPER")) or secrets.token_hex(16)
        
        # Initialize encryption key
        encryption_key = self.config.get("encryption_key", os.getenv("ENCRYPTION_KEY"))
        if encryption_key:
            self.encryption_key = base64.urlsafe_b64decode(encryption_key)
        else:
            # Generate a new encryption key
            self.encryption_key = Fernet.generate_key()
            logger.info("Generated new encryption key")
        
        self.fernet = Fernet(self.encryption_key)
        
        # Login attempt tracking
        self.login_attempts = {}
        
        # Rate limiting
        self.rate_limits = {}
        self.default_rate_limit = self.config.get("rate_limit", DEFAULT_RATE_LIMIT)
        
        self.rate_limit_data = {}
        self.blocked_ips = set()
        self.temp_blocked_ips = {}
        self.auth_tokens = {}
        self.jwt_expiry_hours = self.config.get("jwt_expiry_hours", 24)
        self.jwt_algorithm = "HS256"
        
        logger.info("SecurityManager initialized")
    
    # Password handling methods
    
    def hash_password(self, password: str, salt: Optional[str] = None) -> Tuple[str, str]:
        """Hash a password with salt and pepper.
        
        Args:
            password: The plaintext password
            salt: Optional salt. If not provided, a new salt will be generated.
            
        Returns:
            Tuple of (hashed_password, salt)
        """
        if not salt:
            salt = secrets.token_hex(16)
        
        # Add the pepper
        peppered = password + self.pepper
        
        # Hash with salt
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            peppered.encode(),
            salt.encode(),
            100000  # Number of iterations
        ).hex()
        
        return password_hash, salt
    
    def verify_password(self, password: str, stored_hash: str, salt: str) -> bool:
        """Verify a password against its hash.
        
        Args:
            password: The plaintext password to verify
            stored_hash: The stored hash to compare against
            salt: The salt used to create the hash
            
        Returns:
            True if the password matches, False otherwise
        """
        # Add the pepper
        peppered = password + self.pepper
        
        # Hash with salt
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            peppered.encode(),
            salt.encode(),
            100000  # Number of iterations
        ).hex()
        
        return hmac.compare_digest(password_hash, stored_hash)
    
    def password_meets_requirements(self, password: str) -> Tuple[bool, str]:
        """Check if a password meets security requirements.
        
        Args:
            password: The password to check
            
        Returns:
            Tuple of (meets_requirements, error_message)
        """
        if len(password) < PASSWORD_MIN_LENGTH:
            return False, f"Password must be at least {PASSWORD_MIN_LENGTH} characters long"
        
        # Check for at least one uppercase letter
        if not re.search(r'[A-Z]', password):
            return False, "Password must contain at least one uppercase letter"
        
        # Check for at least one lowercase letter
        if not re.search(r'[a-z]', password):
            return False, "Password must contain at least one lowercase letter"
        
        # Check for at least one digit
        if not re.search(r'\d', password):
            return False, "Password must contain at least one digit"
        
        # Check for at least one special character
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            return False, "Password must contain at least one special character"
        
        return True, ""
    
    # JWT token methods
    
    def generate_jwt_token(self, user_id: str, role: str = "user", custom_claims: Dict = None) -> str:
        """Generate a JWT token for API authentication.
        
        Args:
            user_id: User ID
            role: User role
            custom_claims: Additional claims to include in the token
            
        Returns:
            JWT token string
        """
        now = datetime.utcnow()
        expiry = now + timedelta(hours=self.jwt_expiry_hours)
        
        payload = {
            "sub": user_id,
            "role": role,
            "iat": now,
            "exp": expiry
        }
        
        # Add custom claims if provided
        if custom_claims:
            payload.update(custom_claims)
        
        token = jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
        
        logger.info(f"Generated JWT token for user {user_id} with role {role}")
        return token
    
    def verify_jwt_token(self, token: str) -> Dict:
        """Verify a JWT token and return its payload.
        
        Args:
            token: JWT token string
            
        Returns:
            Token payload or error message
        """
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            return {"valid": True, "payload": payload}
        except jwt.ExpiredSignatureError:
            logger.warning("Expired JWT token")
            return {"valid": False, "error": "Token expired"}
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return {"valid": False, "error": "Invalid token"}
    
    # Login attempt tracking
    
    def record_login_attempt(self, user_id: str, success: bool) -> Dict:
        """Record a login attempt for rate limiting and blocking.
        
        Args:
            user_id: User identifier
            success: Whether the login was successful
            
        Returns:
            Dictionary with login status information
        """
        now = time.time()
        
        # Initialize if first attempt
        if user_id not in self.login_attempts:
            self.login_attempts[user_id] = {
                "attempts": [],
                "blocked_until": None
            }
        
        # If blocked, check if block has expired
        if self.login_attempts[user_id]["blocked_until"]:
            if now < self.login_attempts[user_id]["blocked_until"]:
                # Still blocked
                blocked_for = int(self.login_attempts[user_id]["blocked_until"] - now)
                return {
                    "allowed": False,
                    "blocked": True,
                    "blocked_for": blocked_for,
                    "message": f"Account temporarily locked. Try again in {blocked_for} seconds."
                }
            else:
                # Block expired, reset attempts
                self.login_attempts[user_id]["attempts"] = []
                self.login_attempts[user_id]["blocked_until"] = None
        
        # Record this attempt
        self.login_attempts[user_id]["attempts"].append({
            "time": now,
            "success": success
        })
        
        # Remove attempts older than 30 minutes
        cutoff = now - (LOGIN_TIMEOUT_MINUTES * 60)
        self.login_attempts[user_id]["attempts"] = [
            a for a in self.login_attempts[user_id]["attempts"] if a["time"] > cutoff
        ]
        
        # If successful, reset failed attempts
        if success:
            self.login_attempts[user_id]["attempts"] = [
                a for a in self.login_attempts[user_id]["attempts"] if a["success"]
            ]
            return {"allowed": True, "blocked": False}
        
        # Count recent failed attempts
        recent_failures = sum(1 for a in self.login_attempts[user_id]["attempts"] if not a["success"])
        
        # Block if too many failures
        if recent_failures >= MAX_LOGIN_ATTEMPTS:
            block_time = now + (LOGIN_TIMEOUT_MINUTES * 60)
            self.login_attempts[user_id]["blocked_until"] = block_time
            return {
                "allowed": False,
                "blocked": True,
                "blocked_for": LOGIN_TIMEOUT_MINUTES * 60,
                "message": f"Too many failed login attempts. Account locked for {LOGIN_TIMEOUT_MINUTES} minutes."
            }
        
        # Not blocked
        return {
            "allowed": True,
            "blocked": False,
            "attempts_left": MAX_LOGIN_ATTEMPTS - recent_failures
        }
    
    # Rate limiting
    
    def check_rate_limit(self, identifier: str, action: str = "api") -> bool:
        """Check if an action exceeds the rate limit.
        
        Args:
            identifier: User or IP identifier
            action: Action type to check rate limit for
            
        Returns:
            True if allowed, False if rate limit exceeded
        """
        # Get rate limit configuration
        rate_limit = self.config.get("rate_limits", {}).get(action, self.config.get("rate_limit", 60))
        
        # Get current timestamp
        current_time = time.time()
        
        # Initialize rate limit data for this identifier and action if not exists
        rate_limit_key = f"{identifier}:{action}"
        if rate_limit_key not in self.rate_limit_data:
            self.rate_limit_data[rate_limit_key] = {
                "count": 0,
                "reset_time": current_time + 60  # Reset after 60 seconds
            }
            
        # Check if reset time has passed
        if current_time > self.rate_limit_data[rate_limit_key]["reset_time"]:
            # Reset count and update reset time
            self.rate_limit_data[rate_limit_key] = {
                "count": 1,
                "reset_time": current_time + 60
            }
            return True
            
        # Increment count
        self.rate_limit_data[rate_limit_key]["count"] += 1
        
        # Check if rate limit exceeded
        if self.rate_limit_data[rate_limit_key]["count"] > rate_limit:
            logger.warning(f"Rate limit exceeded for {identifier} on action {action}")
            
            # Temporary block if excessive abuse
            if self.rate_limit_data[rate_limit_key]["count"] > rate_limit * 2:
                self.temp_blocked_ips[identifier] = current_time + 300  # Block for 5 minutes
                logger.warning(f"Temporarily blocking {identifier} for 5 minutes due to excessive requests")
                
            return False
            
        return True
    
    def validate_ip(self, ip: str) -> bool:
        """Validate if an IP address is allowed.
        
        Args:
            ip: IP address to validate
            
        Returns:
            True if allowed, False if blocked
        """
        # Check permanent block list
        if ip in self.blocked_ips:
            logger.warning(f"Blocked IP attempted access: {ip}")
            return False
            
        # Check temporary block list
        current_time = time.time()
        if ip in self.temp_blocked_ips and current_time < self.temp_blocked_ips[ip]:
            logger.warning(f"Temporarily blocked IP attempted access: {ip}")
            return False
            
        # Remove expired temporary blocks
        expired_blocks = [ip for ip, expire_time in self.temp_blocked_ips.items() if current_time >= expire_time]
        for expired_ip in expired_blocks:
            del self.temp_blocked_ips[expired_ip]
            
        return True
    
    # Encryption methods
    
    def encrypt_data(self, data: Union[str, Dict, List]) -> str:
        """Encrypt data.
        
        Args:
            data: Data to encrypt (string, dictionary, or list)
            
        Returns:
            Encrypted data as a string
        """
        if isinstance(data, (dict, list)):
            data = json.dumps(data)
        
        encrypted = self.fernet.encrypt(data.encode())
        return encrypted.decode()
    
    def decrypt_data(self, encrypted_data: str) -> Union[str, Dict, List]:
        """Decrypt data.
        
        Args:
            encrypted_data: Encrypted data string
            
        Returns:
            Decrypted data (string, dictionary, or list)
        """
        try:
            decrypted = self.fernet.decrypt(encrypted_data.encode()).decode()
            
            # Try to parse as JSON
            try:
                return json.loads(decrypted)
            except json.JSONDecodeError:
                # Return as string if not valid JSON
                return decrypted
        except Exception as e:
            logger.error(f"Decryption error: {e}")
            raise ValueError("Failed to decrypt data")
    
    def generate_api_key(self, user_id: str, permissions: List[str] = None) -> str:
        """Generate an API key for a user.
        
        Args:
            user_id: User identifier
            permissions: List of permission strings
            
        Returns:
            API key string
        """
        # Create payload
        payload = {
            "user_id": user_id,
            "created": time.time(),
            "permissions": permissions or []
        }
        
        # Convert to string and add secret
        payload_str = json.dumps(payload) + self.secret_key
        
        # Generate hash
        key_hash = hashlib.sha256(payload_str.encode()).hexdigest()
        
        # Create key with prefix
        prefix = "lt_"
        api_key = f"{prefix}{key_hash[:32]}"
        
        return api_key
    
    # Input validation and sanitization
    
    def sanitize_input(self, input_str: str) -> str:
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
    
    def validate_email(self, email: str) -> bool:
        """Validate an email address.
        
        Args:
            email: Email string to validate
            
        Returns:
            True if valid, False otherwise
        """
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    # Role-based authorization
    
    def check_permission(self, user_role: str, required_role: str, role_hierarchy: Dict = None) -> bool:
        """Check if a user's role has permission for an action.
        
        Args:
            user_role: The user's role
            required_role: The role required for the action
            role_hierarchy: Optional dictionary mapping roles to their hierarchy level
            
        Returns:
            True if the user has permission, False otherwise
        """
        # Default role hierarchy if none provided
        if role_hierarchy is None:
            role_hierarchy = {
                "admin": 100,
                "moderator": 50,
                "premium": 20,
                "user": 10,
                "guest": 1
            }
        
        # Get hierarchy levels
        user_level = role_hierarchy.get(user_role, 0)
        required_level = role_hierarchy.get(required_role, 0)
        
        # Check if user's role is high enough
        return user_level >= required_level

    def jwt_required(self, roles: List[str] = None):
        """Decorator for API routes that require JWT authentication.
        
        Args:
            roles: List of allowed roles
            
        Returns:
            Decorator function
        """
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                from flask import request, jsonify
                
                # Get token from Authorization header
                auth_header = request.headers.get('Authorization')
                
                if not auth_header:
                    return jsonify({"error": "Authorization header is missing"}), 401
                
                # Extract token from Authorization: Bearer <token>
                parts = auth_header.split()
                if parts[0].lower() != 'bearer' or len(parts) != 2:
                    return jsonify({"error": "Invalid Authorization header format"}), 401
                
                token = parts[1]
                
                # Verify token
                result = self.verify_jwt_token(token)
                
                if not result["valid"]:
                    return jsonify({"error": result["error"]}), 401
                
                # Check role if roles specified
                if roles and result["payload"]["role"] not in roles:
                    return jsonify({"error": "Insufficient permissions"}), 403
                
                # Add token payload to kwargs
                kwargs['token_payload'] = result["payload"]
                
                return f(*args, **kwargs)
            
            return decorated_function
        
        return decorator

class RBAC:
    """Role-Based Access Control system."""
    
    def __init__(self, config: Dict = None):
        """Initialize the RBAC system.
        
        Args:
            config: RBAC configuration dictionary
        """
        self.config = config or {}
        
        # Default roles and permissions
        self.roles = self.config.get("roles", {
            "admin": {
                "description": "Administrator with full access",
                "permissions": ["*"]
            },
            "moderator": {
                "description": "Moderator with content management access",
                "permissions": ["read:*", "write:content", "edit:content", "delete:content"]
            },
            "premium": {
                "description": "Premium user with enhanced access",
                "permissions": ["read:*", "write:own", "edit:own"]
            },
            "user": {
                "description": "Standard user",
                "permissions": ["read:public", "write:own", "edit:own"]
            },
            "guest": {
                "description": "Guest user with limited access",
                "permissions": ["read:public"]
            }
        })
        
        # Role hierarchy
        self.role_hierarchy = self.config.get("role_hierarchy", {
            "admin": 100,
            "moderator": 50,
            "premium": 20,
            "user": 10,
            "guest": 1
        })
    
    def has_permission(self, role: str, permission: str) -> bool:
        """Check if a role has a specific permission.
        
        Args:
            role: Role name
            permission: Permission to check
            
        Returns:
            True if the role has the permission, False otherwise
        """
        if role not in self.roles:
            return False
        
        role_permissions = self.roles[role]["permissions"]
        
        # Wildcard permission
        if "*" in role_permissions:
            return True
        
        # Check for category wildcard (e.g., "read:*")
        if permission.count(":") > 0:
            category = permission.split(":")[0]
            if f"{category}:*" in role_permissions:
                return True
        
        # Direct permission check
        return permission in role_permissions
    
    def get_role_permissions(self, role: str) -> List[str]:
        """Get all permissions for a role.
        
        Args:
            role: Role name
            
        Returns:
            List of permission strings
        """
        if role not in self.roles:
            return []
        
        return self.roles[role]["permissions"]
    
    def add_role(self, role: str, description: str, permissions: List[str], level: int = 1) -> bool:
        """Add a new role to the RBAC system.
        
        Args:
            role: Role name
            description: Role description
            permissions: List of permission strings
            level: Hierarchy level for the role
            
        Returns:
            True if successful, False if role already exists
        """
        if role in self.roles:
            return False
        
        self.roles[role] = {
            "description": description,
            "permissions": permissions
        }
        
        self.role_hierarchy[role] = level
        return True
    
    def remove_role(self, role: str) -> bool:
        """Remove a role from the RBAC system.
        
        Args:
            role: Role name
            
        Returns:
            True if successful, False if role doesn't exist
        """
        if role not in self.roles:
            return False
        
        del self.roles[role]
        
        if role in self.role_hierarchy:
            del self.role_hierarchy[role]
        
        return True
    
    def add_permission_to_role(self, role: str, permission: str) -> bool:
        """Add a permission to a role.
        
        Args:
            role: Role name
            permission: Permission to add
            
        Returns:
            True if successful, False if role doesn't exist
        """
        if role not in self.roles:
            return False
        
        if permission not in self.roles[role]["permissions"]:
            self.roles[role]["permissions"].append(permission)
        
        return True
    
    def remove_permission_from_role(self, role: str, permission: str) -> bool:
        """Remove a permission from a role.
        
        Args:
            role: Role name
            permission: Permission to remove
            
        Returns:
            True if successful, False if role doesn't exist
        """
        if role not in self.roles:
            return False
        
        if permission in self.roles[role]["permissions"]:
            self.roles[role]["permissions"].remove(permission)
        
        return True

class JWTAuthMiddleware:
    """JWT Authentication Middleware for Flask applications.
    
    This middleware can be used to protect routes with JWT authentication
    at the application level instead of at individual route level.
    """
    
    def __init__(self, app=None, security_manager=None, exempt_routes=None, exempt_prefixes=None):
        """Initialize the middleware.
        
        Args:
            app: Flask application instance
            security_manager: SecurityManager instance
            exempt_routes: List of route paths exempt from authentication
            exempt_prefixes: List of URL prefixes exempt from authentication
        """
        self.security_manager = security_manager
        self.exempt_routes = exempt_routes or []
        self.exempt_prefixes = exempt_prefixes or []
        
        if app is not None:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize the middleware with a Flask application.
        
        Args:
            app: Flask application instance
        """
        # If security_manager not provided, try to get it from app config
        if self.security_manager is None:
            if hasattr(app, 'config') and 'security_manager' in app.config:
                self.security_manager = app.config['security_manager']
            else:
                # Import here to avoid circular imports
                from system_init import get_system
                system = get_system()
                if system:
                    self.security_manager = system.security_manager
                else:
                    raise ValueError("SecurityManager not provided and could not be retrieved from system")
        
        # Register before_request handler
        @app.before_request
        def verify_jwt():
            """Verify JWT token before handling the request."""
            from flask import request, jsonify, g
            
            # Skip exempt routes
            if request.path in self.exempt_routes:
                return None
            
            # Skip exempt prefixes
            for prefix in self.exempt_prefixes:
                if request.path.startswith(prefix):
                    return None
            
            # Skip OPTIONS requests (for CORS)
            if request.method == 'OPTIONS':
                return None
            
            # Get token from Authorization header
            auth_header = request.headers.get('Authorization')
            
            if not auth_header:
                return jsonify({"error": "Authorization header is missing"}), 401
            
            # Extract token from Authorization: Bearer <token>
            parts = auth_header.split()
            if parts[0].lower() != 'bearer' or len(parts) != 2:
                return jsonify({"error": "Invalid Authorization header format"}), 401
            
            token = parts[1]
            
            # Verify token
            result = self.security_manager.verify_jwt_token(token)
            
            if not result["valid"]:
                return jsonify({"error": result["error"]}), 401
            
            # Store token payload in g for access in route handlers
            g.token_payload = result["payload"]
            g.user_id = result["payload"].get("sub")
            g.user_role = result["payload"].get("role", "user")
            
            # Continue processing the request
            return None 