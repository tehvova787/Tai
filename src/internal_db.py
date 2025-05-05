"""
Internal Database Module for Lucky Train AI Assistant

This module provides a simple internal database using SQLite:
- User management
- Question history tracking
- Settings storage
- Analytics data
"""

import os
import sqlite3
import json
import time
import logging
import threading
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import uuid
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default database file path
DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'lucky_train.db')

# Ensure data directory exists
data_dir = os.path.dirname(DEFAULT_DB_PATH)
os.makedirs(data_dir, exist_ok=True)

class InternalDatabase:
    """Internal database for Lucky Train AI Assistant."""
    
    def __init__(self, db_path: str = DEFAULT_DB_PATH, config: Dict = None):
        """Initialize the internal database.
        
        Args:
            db_path: Path to the database file
            config: Database configuration dictionary
        """
        self.db_path = db_path
        self.config = config or {}
        
        # Thread local storage for connections
        self.local = threading.local()
        
        # Initialize database
        self._init_db()
        
        logger.info(f"Internal database initialized at {db_path}")
    
    @contextmanager
    def get_connection(self):
        """Get a database connection from the thread-local storage or create a new one.
        
        Yields:
            SQLite connection
        """
        if not hasattr(self.local, 'connection'):
            self.local.connection = sqlite3.connect(self.db_path)
            self.local.connection.row_factory = sqlite3.Row
        
        try:
            yield self.local.connection
        except Exception as e:
            self.local.connection.rollback()
            raise e
    
    def _init_db(self) -> None:
        """Initialize the database schema if it doesn't exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Create users table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT NOT NULL UNIQUE,
                email TEXT UNIQUE,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                created_at TEXT NOT NULL,
                last_login TEXT,
                settings TEXT,
                metadata TEXT
            )
            ''')
            
            # Create sessions table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                token TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                ip_address TEXT,
                user_agent TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
            ''')
            
            # Create questions table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS questions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                question TEXT NOT NULL,
                response TEXT NOT NULL,
                created_at TEXT NOT NULL,
                context TEXT,
                model TEXT,
                execution_time REAL,
                token_count INTEGER,
                similarity_score REAL,
                cached BOOLEAN,
                feedback TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
            ''')
            
            # Create settings table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            ''')
            
            # Create analytics table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS analytics (
                id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                event_data TEXT,
                user_id TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
            ''')
            
            # Create api_keys table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_keys (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                key_hash TEXT NOT NULL,
                name TEXT NOT NULL,
                permissions TEXT,
                created_at TEXT NOT NULL,
                expires_at TEXT,
                last_used TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
            ''')
            
            # Create user_logs table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_logs (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                details TEXT,
                ip_address TEXT,
                user_agent TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
            ''')
            
            conn.commit()
    
    # User management methods
    
    def add_user(self, username: str, password_hash: str, salt: str, email: str = None, 
                role: str = 'user', settings: Dict = None, metadata: Dict = None) -> str:
        """Add a new user to the database.
        
        Args:
            username: User's username
            password_hash: Hashed password
            salt: Salt used in password hashing
            email: User's email address
            role: User's role (e.g., user, admin)
            settings: User settings dictionary
            metadata: User metadata dictionary
            
        Returns:
            User ID
        """
        user_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT INTO users 
            (id, username, email, password_hash, salt, role, created_at, settings, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                username,
                email,
                password_hash,
                salt,
                role,
                now,
                json.dumps(settings or {}),
                json.dumps(metadata or {})
            ))
            
            conn.commit()
        
        return user_id
    
    def get_user(self, user_id: str = None, username: str = None, email: str = None) -> Optional[Dict]:
        """Get a user from the database.
        
        Args:
            user_id: User ID
            username: Username
            email: Email address
            
        Returns:
            User dictionary or None if not found
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if user_id:
                cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
            elif username:
                cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
            elif email:
                cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
            else:
                return None
            
            row = cursor.fetchone()
            
            if not row:
                return None
            
            user = dict(row)
            
            # Parse JSON fields
            for field in ['settings', 'metadata']:
                if user[field]:
                    user[field] = json.loads(user[field])
                else:
                    user[field] = {}
            
            return user
    
    def update_user(self, user_id: str, updates: Dict) -> bool:
        """Update a user in the database.
        
        Args:
            user_id: User ID
            updates: Dictionary of fields to update
            
        Returns:
            True if successful, False otherwise
        """
        allowed_fields = {
            'username': str,
            'email': str,
            'password_hash': str,
            'salt': str,
            'role': str,
            'last_login': str,
            'settings': dict,
            'metadata': dict
        }
        
        update_sql = []
        params = []
        
        for field, value in updates.items():
            if field not in allowed_fields:
                continue
            
            if field in ['settings', 'metadata'] and isinstance(value, dict):
                update_sql.append(f"{field} = ?")
                params.append(json.dumps(value))
            else:
                update_sql.append(f"{field} = ?")
                params.append(value)
        
        if not update_sql:
            return False
        
        # Add user_id to params
        params.append(user_id)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute(f'''
            UPDATE users SET {', '.join(update_sql)} WHERE id = ?
            ''', params)
            
            conn.commit()
            
            return cursor.rowcount > 0
    
    def delete_user(self, user_id: str) -> bool:
        """Delete a user from the database.
        
        Args:
            user_id: User ID
            
        Returns:
            True if successful, False otherwise
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Delete related records
            for table in ['sessions', 'questions', 'analytics', 'api_keys', 'user_logs']:
                cursor.execute(f'DELETE FROM {table} WHERE user_id = ?', (user_id,))
            
            # Delete user
            cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
            
            conn.commit()
            
            return cursor.rowcount > 0
    
    def update_last_login(self, user_id: str) -> bool:
        """Update a user's last login timestamp.
        
        Args:
            user_id: User ID
            
        Returns:
            True if successful, False otherwise
        """
        now = datetime.utcnow().isoformat()
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('UPDATE users SET last_login = ? WHERE id = ?', (now, user_id))
            
            conn.commit()
            
            return cursor.rowcount > 0
    
    # Session management methods
    
    def create_session(self, user_id: str, token: str, expires_at: str, 
                      ip_address: str = None, user_agent: str = None) -> str:
        """Create a new session for a user.
        
        Args:
            user_id: User ID
            token: Session token
            expires_at: Expiration timestamp
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT INTO sessions 
            (id, user_id, token, created_at, expires_at, ip_address, user_agent)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                session_id,
                user_id,
                token,
                now,
                expires_at,
                ip_address,
                user_agent
            ))
            
            conn.commit()
        
        return session_id
    
    def get_session(self, token: str) -> Optional[Dict]:
        """Get a session by token.
        
        Args:
            token: Session token
            
        Returns:
            Session dictionary or None if not found
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM sessions WHERE token = ?', (token,))
            
            row = cursor.fetchone()
            
            if not row:
                return None
            
            return dict(row)
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            True if successful, False otherwise
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM sessions WHERE id = ?', (session_id,))
            
            conn.commit()
            
            return cursor.rowcount > 0
    
    def cleanup_expired_sessions(self) -> int:
        """Remove expired sessions from the database.
        
        Returns:
            Number of sessions removed
        """
        now = datetime.utcnow().isoformat()
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM sessions WHERE expires_at < ?', (now,))
            
            conn.commit()
            
            return cursor.rowcount
    
    # Question history methods
    
    def add_question(self, user_id: str, question: str, response: str, context: Dict = None, 
                    model: str = None, execution_time: float = None, token_count: int = None,
                    similarity_score: float = None, cached: bool = False) -> str:
        """Add a question to the history.
        
        Args:
            user_id: User ID
            question: The question asked
            response: The response given
            context: Question context
            model: AI model used
            execution_time: Execution time in ms
            token_count: Number of tokens used
            similarity_score: Similarity score if matched with cached question
            cached: Whether the response was from cache
            
        Returns:
            Question ID
        """
        question_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT INTO questions 
            (id, user_id, question, response, created_at, context, model, 
             execution_time, token_count, similarity_score, cached)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                question_id,
                user_id,
                question,
                response,
                now,
                json.dumps(context) if context else None,
                model,
                execution_time,
                token_count,
                similarity_score,
                1 if cached else 0
            ))
            
            conn.commit()
        
        return question_id
    
    def get_question_history(self, user_id: str, limit: int = 100, offset: int = 0) -> List[Dict]:
        """Get a user's question history.
        
        Args:
            user_id: User ID
            limit: Maximum number of questions to return
            offset: Offset for pagination
            
        Returns:
            List of question dictionaries
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT * FROM questions 
            WHERE user_id = ? 
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            ''', (user_id, limit, offset))
            
            rows = cursor.fetchall()
            
            questions = []
            for row in rows:
                question = dict(row)
                
                # Parse context JSON
                if question['context']:
                    question['context'] = json.loads(question['context'])
                
                # Convert cached to boolean
                question['cached'] = bool(question['cached'])
                
                questions.append(question)
            
            return questions
    
    def add_question_feedback(self, question_id: str, feedback: Dict) -> bool:
        """Add feedback for a question.
        
        Args:
            question_id: Question ID
            feedback: Feedback dictionary
            
        Returns:
            True if successful, False otherwise
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
            UPDATE questions 
            SET feedback = ?
            WHERE id = ?
            ''', (json.dumps(feedback), question_id))
            
            conn.commit()
            
            return cursor.rowcount > 0
    
    # Settings methods
    
    def set_setting(self, key: str, value: Any) -> bool:
        """Set a global setting.
        
        Args:
            key: Setting key
            value: Setting value
            
        Returns:
            True if successful, False otherwise
        """
        now = datetime.utcnow().isoformat()
        
        # Convert value to JSON if it's not a string
        if not isinstance(value, str):
            value = json.dumps(value)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT OR REPLACE INTO settings 
            (key, value, updated_at)
            VALUES (?, ?, ?)
            ''', (key, value, now))
            
            conn.commit()
            
            return True
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a global setting.
        
        Args:
            key: Setting key
            default: Default value if not found
            
        Returns:
            Setting value
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT value FROM settings WHERE key = ?', (key,))
            
            row = cursor.fetchone()
            
            if not row:
                return default
            
            value = row['value']
            
            # Try to parse as JSON
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
    
    def get_all_settings(self) -> Dict:
        """Get all global settings.
        
        Returns:
            Dictionary of settings
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT key, value FROM settings')
            
            rows = cursor.fetchall()
            
            settings = {}
            for row in rows:
                key = row['key']
                value = row['value']
                
                # Try to parse as JSON
                try:
                    settings[key] = json.loads(value)
                except json.JSONDecodeError:
                    settings[key] = value
            
            return settings
    
    # Analytics methods
    
    def log_analytics_event(self, event_type: str, event_data: Dict = None, user_id: str = None) -> str:
        """Log an analytics event.
        
        Args:
            event_type: Type of event
            event_data: Event data
            user_id: User ID
            
        Returns:
            Event ID
        """
        event_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT INTO analytics 
            (id, event_type, event_data, user_id, created_at)
            VALUES (?, ?, ?, ?, ?)
            ''', (
                event_id,
                event_type,
                json.dumps(event_data) if event_data else None,
                user_id,
                now
            ))
            
            conn.commit()
        
        return event_id
    
    def get_analytics(self, event_type: str = None, user_id: str = None, 
                     start_date: str = None, end_date: str = None, 
                     limit: int = 1000, offset: int = 0) -> List[Dict]:
        """Get analytics events.
        
        Args:
            event_type: Filter by event type
            user_id: Filter by user ID
            start_date: Filter by start date (ISO format)
            end_date: Filter by end date (ISO format)
            limit: Maximum number of events to return
            offset: Offset for pagination
            
        Returns:
            List of event dictionaries
        """
        query = 'SELECT * FROM analytics WHERE 1=1'
        params = []
        
        if event_type:
            query += ' AND event_type = ?'
            params.append(event_type)
        
        if user_id:
            query += ' AND user_id = ?'
            params.append(user_id)
        
        if start_date:
            query += ' AND created_at >= ?'
            params.append(start_date)
        
        if end_date:
            query += ' AND created_at <= ?'
            params.append(end_date)
        
        query += ' ORDER BY created_at DESC LIMIT ? OFFSET ?'
        params.extend([limit, offset])
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute(query, params)
            
            rows = cursor.fetchall()
            
            events = []
            for row in rows:
                event = dict(row)
                
                # Parse event data JSON
                if event['event_data']:
                    event['event_data'] = json.loads(event['event_data'])
                
                events.append(event)
            
            return events
    
    # API key methods
    
    def add_api_key(self, user_id: str, key_hash: str, name: str, 
                   permissions: List[str] = None, expires_at: str = None) -> str:
        """Add an API key for a user.
        
        Args:
            user_id: User ID
            key_hash: Hashed API key
            name: Key name
            permissions: List of permission strings
            expires_at: Expiration timestamp
            
        Returns:
            API key ID
        """
        key_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT INTO api_keys 
            (id, user_id, key_hash, name, permissions, created_at, expires_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                key_id,
                user_id,
                key_hash,
                name,
                json.dumps(permissions) if permissions else None,
                now,
                expires_at
            ))
            
            conn.commit()
        
        return key_id
    
    def get_api_key(self, key_hash: str) -> Optional[Dict]:
        """Get an API key by hash.
        
        Args:
            key_hash: Hashed API key
            
        Returns:
            API key dictionary or None if not found
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM api_keys WHERE key_hash = ?', (key_hash,))
            
            row = cursor.fetchone()
            
            if not row:
                return None
            
            api_key = dict(row)
            
            # Parse JSON fields
            if api_key['permissions']:
                api_key['permissions'] = json.loads(api_key['permissions'])
            else:
                api_key['permissions'] = []
            
            return api_key
    
    def update_api_key_usage(self, key_id: str) -> bool:
        """Update the last used timestamp for an API key.
        
        Args:
            key_id: API key ID
            
        Returns:
            True if successful, False otherwise
        """
        now = datetime.utcnow().isoformat()
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('UPDATE api_keys SET last_used = ? WHERE id = ?', (now, key_id))
            
            conn.commit()
            
            return cursor.rowcount > 0
    
    def delete_api_key(self, key_id: str) -> bool:
        """Delete an API key.
        
        Args:
            key_id: API key ID
            
        Returns:
            True if successful, False otherwise
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM api_keys WHERE id = ?', (key_id,))
            
            conn.commit()
            
            return cursor.rowcount > 0
    
    # User logs methods
    
    def log_user_event(self, user_id: str, event_type: str, details: Dict = None, 
                      ip_address: str = None, user_agent: str = None) -> str:
        """Log a user event.
        
        Args:
            user_id: User ID
            event_type: Type of event
            details: Event details
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            Log entry ID
        """
        log_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT INTO user_logs 
            (id, user_id, event_type, details, ip_address, user_agent, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                log_id,
                user_id,
                event_type,
                json.dumps(details) if details else None,
                ip_address,
                user_agent,
                now
            ))
            
            conn.commit()
        
        return log_id
    
    def get_user_logs(self, user_id: str, event_type: str = None, 
                     start_date: str = None, end_date: str = None, 
                     limit: int = 100, offset: int = 0) -> List[Dict]:
        """Get logs for a user.
        
        Args:
            user_id: User ID
            event_type: Filter by event type
            start_date: Filter by start date (ISO format)
            end_date: Filter by end date (ISO format)
            limit: Maximum number of logs to return
            offset: Offset for pagination
            
        Returns:
            List of log dictionaries
        """
        query = 'SELECT * FROM user_logs WHERE user_id = ?'
        params = [user_id]
        
        if event_type:
            query += ' AND event_type = ?'
            params.append(event_type)
        
        if start_date:
            query += ' AND created_at >= ?'
            params.append(start_date)
        
        if end_date:
            query += ' AND created_at <= ?'
            params.append(end_date)
        
        query += ' ORDER BY created_at DESC LIMIT ? OFFSET ?'
        params.extend([limit, offset])
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute(query, params)
            
            rows = cursor.fetchall()
            
            logs = []
            for row in rows:
                log = dict(row)
                
                # Parse details JSON
                if log['details']:
                    log['details'] = json.loads(log['details'])
                
                logs.append(log)
            
            return logs

# Singleton database instance
_db_instance = None

def get_db(config: Dict = None) -> InternalDatabase:
    """Get or create the database instance.
    
    Args:
        config: Database configuration
        
    Returns:
        Database instance
    """
    global _db_instance
    
    if _db_instance is None:
        db_path = config.get('db_path', DEFAULT_DB_PATH) if config else DEFAULT_DB_PATH
        _db_instance = InternalDatabase(db_path, config)
    
    return _db_instance 