"""
System Demonstration for Lucky Train AI Assistant

This script demonstrates the functionality of the system components:
- Security (authentication, authorization)
- Internal database
- Caching and question caching
- Logging
"""

import os
import json
import time
import argparse
from datetime import datetime, timedelta
import getpass
import random
from typing import Dict, List, Any, Optional

# Import system components
from system_init import init_system, get_system

def security_demo():
    """Demonstrate security features."""
    system = get_system()
    logger = system.logger
    security = system.security_manager
    rbac = system.rbac
    
    print("\n========== SECURITY DEMONSTRATION ==========\n")
    
    # Password hashing
    print("Password Hashing Demo:")
    password = "SecureP@ssw0rd"
    salt = None
    
    print(f"Original Password: {password}")
    password_hash, salt = security.hash_password(password, salt)
    print(f"Password Hash: {password_hash}")
    print(f"Salt: {salt}")
    
    # Password verification
    print("\nPassword Verification Demo:")
    verification_result = security.verify_password(password, password_hash, salt)
    print(f"Correct Password Verification: {verification_result}")
    
    verification_result = security.verify_password("WrongPassword", password_hash, salt)
    print(f"Incorrect Password Verification: {verification_result}")
    
    # JWT token generation
    print("\nJWT Token Demo:")
    user_id = "demo_user_1"
    token = security.generate_jwt_token(user_id, "admin")
    print(f"Generated JWT Token: {token}")
    
    # JWT token verification
    print("\nJWT Token Verification Demo:")
    result = security.verify_jwt_token(token)
    print(f"Token Verification Result: {result}")
    
    # RBAC permissions
    print("\nRole-Based Access Control Demo:")
    print("Checking permissions for different roles:")
    
    roles = ["admin", "moderator", "user", "guest"]
    permissions = ["read:public", "write:content", "delete:content", "admin:users"]
    
    for role in roles:
        print(f"\nRole: {role}")
        for permission in permissions:
            has_perm = rbac.has_permission(role, permission)
            print(f"  - {permission}: {has_perm}")
    
    # Rate limiting
    print("\nRate Limiting Demo:")
    print("Simulating API requests...")
    
    user_id = "demo_user_1"
    endpoint = "api"
    
    for i in range(10):
        rate_limit_check = security.check_rate_limit(user_id, endpoint)
        status = "Allowed" if rate_limit_check["allowed"] else "Blocked"
        remaining = rate_limit_check.get("remaining", 0)
        print(f"Request {i+1}: {status} (Remaining: {remaining})")
    
    # Data encryption
    print("\nData Encryption Demo:")
    sensitive_data = {
        "user_id": "demo_user_1",
        "credit_card": "1234-5678-9012-3456",
        "expiry": "12/25",
        "cvv": "123"
    }
    
    print(f"Original Data: {sensitive_data}")
    encrypted = security.encrypt_data(sensitive_data)
    print(f"Encrypted Data: {encrypted}")
    
    decrypted = security.decrypt_data(encrypted)
    print(f"Decrypted Data: {decrypted}")
    
    print("\nSecurity demo completed.")

def database_demo():
    """Demonstrate internal database features."""
    system = get_system()
    logger = system.logger
    db = system.db
    security = system.security_manager
    
    print("\n========== DATABASE DEMONSTRATION ==========\n")
    
    # User management
    print("User Management Demo:")
    
    # Create a test user
    username = "test_user"
    password = "TestPassword123!"
    email = "test@example.com"
    
    password_hash, salt = security.hash_password(password)
    
    print(f"Creating user: {username}")
    user_id = db.add_user(
        username=username,
        password_hash=password_hash,
        salt=salt,
        email=email,
        role="user",
        settings={"theme": "dark", "notifications": True},
        metadata={"registration_source": "demo"}
    )
    
    print(f"Created user with ID: {user_id}")
    
    # Get user
    user = db.get_user(username=username)
    print(f"Retrieved user: {user['username']} (ID: {user['id']})")
    
    # Update user
    print("\nUpdating user settings...")
    db.update_user(user_id, {
        "settings": {"theme": "light", "notifications": False, "language": "en"}
    })
    
    # Get updated user
    user = db.get_user(user_id=user_id)
    print(f"Updated user settings: {user['settings']}")
    
    # Session management
    print("\nSession Management Demo:")
    token = security.generate_jwt_token(user_id)
    expires_at = (datetime.utcnow() + timedelta(hours=24)).isoformat()
    
    session_id = db.create_session(
        user_id=user_id,
        token=token,
        expires_at=expires_at,
        ip_address="127.0.0.1",
        user_agent="Demo/1.0"
    )
    
    print(f"Created session with ID: {session_id}")
    
    # Get session
    session = db.get_session(token)
    print(f"Retrieved session for user: {session['user_id']}")
    
    # Question history
    print("\nQuestion History Demo:")
    
    questions = [
        "What is Lucky Train?",
        "How do I buy an NFT?",
        "What is the current price of TON?",
        "How do I connect my wallet?",
        "What are the game mechanics?"
    ]
    
    for question in questions:
        response = f"This is a sample response to: {question}"
        
        question_id = db.add_question(
            user_id=user_id,
            question=question,
            response=response,
            context={"language": "en", "topic": "general"},
            model="gpt-3.5-turbo",
            execution_time=random.uniform(100, 500),
            token_count=random.randint(50, 200),
            cached=False
        )
        
        print(f"Added question: '{question}' with ID: {question_id}")
    
    # Get question history
    history = db.get_question_history(user_id, limit=3)
    print(f"\nRetrieved {len(history)} recent questions for user {user_id}:")
    for item in history:
        print(f"  - {item['question']}")
    
    # Settings
    print("\nSettings Demo:")
    
    settings = {
        "app_name": "Lucky Train AI",
        "version": "1.0.0",
        "max_tokens": 2000,
        "features": {
            "voice": True,
            "image_generation": True,
            "blockchain": True
        }
    }
    
    db.set_setting("app_settings", settings)
    print(f"Saved settings: {settings}")
    
    retrieved_settings = db.get_setting("app_settings")
    print(f"Retrieved settings: {retrieved_settings}")
    
    # Analytics
    print("\nAnalytics Demo:")
    
    event_types = ["login", "logout", "question", "registration", "error"]
    
    for _ in range(5):
        event_type = random.choice(event_types)
        event_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "details": f"Sample {event_type} event",
            "success": random.choice([True, False, True, True])
        }
        
        event_id = db.log_analytics_event(
            event_type=event_type,
            event_data=event_data,
            user_id=user_id if random.random() > 0.3 else None
        )
        
        print(f"Logged {event_type} event with ID: {event_id}")
    
    # Get analytics
    analytics = db.get_analytics(limit=3)
    print(f"\nRetrieved {len(analytics)} recent analytics events:")
    for event in analytics:
        print(f"  - {event['event_type']}: {event['event_data']}")
    
    # API keys
    print("\nAPI Keys Demo:")
    
    api_key = security.generate_api_key(user_id, ["read:data", "write:own"])
    api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    
    key_id = db.add_api_key(
        user_id=user_id,
        key_hash=api_key_hash,
        name="Demo API Key",
        permissions=["read:data", "write:own"]
    )
    
    print(f"Created API key: {api_key} with ID: {key_id}")
    
    # Clean up
    print("\nCleaning up database...")
    db.delete_user(user_id)
    print(f"User {username} deleted.")
    
    print("\nDatabase demo completed.")

def caching_demo():
    """Demonstrate caching features."""
    system = get_system()
    logger = system.logger
    cache_manager = system.cache_manager
    question_cache = system.question_cache
    
    print("\n========== CACHING DEMONSTRATION ==========\n")
    
    # Basic caching
    print("Basic Caching Demo:")
    
    # Store some values
    cache_manager.set("greeting", "Hello, World!")
    cache_manager.set("lucky_number", 42)
    cache_manager.set("user_data", {"name": "John", "age": 30})
    
    # Retrieve values
    greeting = cache_manager.get("greeting")
    lucky_number = cache_manager.get("lucky_number")
    user_data = cache_manager.get("user_data")
    
    print(f"Retrieved from cache:")
    print(f"  - greeting: {greeting}")
    print(f"  - lucky_number: {lucky_number}")
    print(f"  - user_data: {user_data}")
    
    # Non-existent key
    nonexistent = cache_manager.get("nonexistent")
    print(f"  - nonexistent: {nonexistent}")
    
    # Question caching
    print("\nQuestion Caching Demo:")
    
    # Check if question caching is enabled
    cache_stats = question_cache.stats()
    print(f"Question Cache Statistics:")
    print(f"  - Enabled: {cache_stats.get('enabled', False)}")
    
    if not cache_stats.get('enabled', False):
        print("Question caching is disabled, skipping demonstration.")
    else:
        # Add some questions to the cache
        original_questions = [
            "What is Lucky Train?",
            "How do I connect my wallet to the platform?",
            "What is the price of TON?",
            "How do I buy Lucky Train NFTs?",
            "Explain the game mechanics."
        ]
        
        for i, question in enumerate(original_questions):
            answer = f"This is the answer to question {i+1}: {question}"
            question_cache.set(question, answer)
            print(f"Added to cache: '{question}'")
        
        # Try to retrieve from cache with similar questions
        similar_questions = [
            "Tell me about Lucky Train",
            "How to connect wallet to the platform?",
            "What's the current TON price?",
            "How can I purchase Lucky Train NFTs?",
            "Can you explain the game mechanics?"
        ]
        
        print("\nTrying to retrieve similar questions:")
        for question in similar_questions:
            result, similarity, original = question_cache.get(question)
            
            if result:
                print(f"  - Q: '{question}'")
                print(f"    - Cache hit! Similarity: {similarity:.2f}")
                print(f"    - Original question: '{original}'")
            else:
                print(f"  - Q: '{question}'")
                print(f"    - Cache miss!")
        
        # Clear the cache
        question_cache.clear()
        print("\nQuestion cache cleared.")
    
    print("\nCaching demo completed.")

def logging_demo():
    """Demonstrate logging features."""
    system = get_system()
    logger = system.logger
    
    print("\n========== LOGGING DEMONSTRATION ==========\n")
    
    print("Logging Demo:")
    print("(Check the log files for results)")
    
    # Log some messages
    logger.info("This is an informational message from the demo")
    logger.warning("This is a warning message from the demo")
    logger.error("This is an error message from the demo")
    
    # Log with extra data
    logger.info(
        "Message with extra data",
        extra={
            "user_id": "demo_user",
            "action": "demo",
            "data": {"key1": "value1", "key2": "value2"}
        }
    )
    
    # Log access
    logger.log_access(
        user_id="demo_user",
        endpoint="/api/demo",
        method="GET",
        status_code=200,
        duration_ms=35.2,
        request_data={"param1": "value1"},
        response_data={"result": "success"}
    )
    
    # Log security event
    logger.log_security(
        event_type="login",
        user_id="demo_user",
        result="success",
        details={
            "ip_address": "127.0.0.1",
            "user_agent": "Demo/1.0",
            "method": "password"
        }
    )
    
    # Log a fake exception
    try:
        # Simulate an exception
        result = 1 / 0
    except Exception as e:
        logger.exception(f"Caught an exception: {str(e)}")
    
    print("\nLogging demo completed. Check the log files in the 'logs' directory.")

def full_demo():
    """Run a comprehensive demonstration of all features."""
    try:
        security_demo()
        time.sleep(1)
        database_demo()
        time.sleep(1)
        caching_demo()
        time.sleep(1)
        logging_demo()
        
        print("\n========== DEMONSTRATION COMPLETED ==========\n")
        print("All system components have been demonstrated successfully.")
    except Exception as e:
        print(f"\nDemonstration error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Initialize the system
    system = init_system()
    
    parser = argparse.ArgumentParser(description="Lucky Train AI System Demo")
    parser.add_argument(
        '--component',
        choices=['security', 'database', 'caching', 'logging', 'all'],
        default='all',
        help="The component to demonstrate"
    )
    
    args = parser.parse_args()
    
    try:
        if args.component == 'security':
            security_demo()
        elif args.component == 'database':
            database_demo()
        elif args.component == 'caching':
            caching_demo()
        elif args.component == 'logging':
            logging_demo()
        else:
            full_demo()
    except KeyboardInterrupt:
        print("\nDemonstration interrupted.")
    finally:
        # Clean up resources
        system.shutdown() 