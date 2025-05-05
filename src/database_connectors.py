"""
Database Connectors for Lucky Train AI Assistant

This module provides connectors to various databases:
- Chat2DB
- Google BigQuery
- Amazon Aurora
- Microsoft Azure Cosmos DB
- Snowflake
- IBM Db2 AI
- Kaggle Datasets
- Google Dataset Search
- UCI Machine Learning Repository
- Amazon's AWS Public Datasets
"""

import logging
import os
import json
from typing import Dict, List, Optional, Union, Any
from abc import ABC, abstractmethod
from dotenv import load_dotenv
import requests
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class BaseDBConnector(ABC):
    """Base class for all database connectors."""
    
    def __init__(self, config: Dict = None):
        """Initialize the database connector.
        
        Args:
            config: Configuration for the connector
        """
        self.config = config or {}
        self.name = self.__class__.__name__
        self.connection = None
        logger.info(f"Initializing {self.name}")
    
    @abstractmethod
    def connect(self) -> bool:
        """Connect to the database.
        
        Returns:
            True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def execute_query(self, query: str, params: Dict = None) -> Dict:
        """Execute a query on the database.
        
        Args:
            query: Query string
            params: Query parameters
            
        Returns:
            Query results
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """Disconnect from the database.
        
        Returns:
            True if disconnection successful, False otherwise
        """
        pass
    
    def is_connected(self) -> bool:
        """Check if connected to the database.
        
        Returns:
            True if connected, False otherwise
        """
        return self.connection is not None

class Chat2DBConnector(BaseDBConnector):
    """Connector for Chat2DB."""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.api_endpoint = self.config.get("api_endpoint", os.getenv("CHAT2DB_API_ENDPOINT"))
        self.api_key = self.config.get("api_key", os.getenv("CHAT2DB_API_KEY"))
        self.headers = {
            "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
            "Content-Type": "application/json"
        }
    
    def connect(self) -> bool:
        """Connect to Chat2DB.
        
        Returns:
            True if connection successful, False otherwise
        """
        if not self.api_endpoint:
            logger.error("Chat2DB API endpoint not configured")
            return False
        
        try:
            # Test connection with a simple request
            response = requests.get(
                f"{self.api_endpoint}/status",
                headers=self.headers
            )
            
            if response.status_code == 200:
                self.connection = True
                logger.info("Connected to Chat2DB successfully")
                return True
            else:
                logger.error(f"Failed to connect to Chat2DB: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error connecting to Chat2DB: {e}")
            return False
    
    def execute_query(self, query: str, params: Dict = None) -> Dict:
        """Execute a query on Chat2DB.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            Query results
        """
        if not self.is_connected():
            logger.warning("Not connected to Chat2DB, attempting to connect")
            if not self.connect():
                return {"error": "Not connected to Chat2DB", "success": False}
        
        try:
            data = {
                "sql": query,
                "params": params or {}
            }
            
            response = requests.post(
                f"{self.api_endpoint}/execute",
                headers=self.headers,
                json=data
            )
            
            if response.status_code == 200:
                result = response.json()
                return {"data": result, "success": True}
            else:
                error_msg = f"Query execution failed: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return {"error": error_msg, "success": False}
        except Exception as e:
            error_msg = f"Error executing query on Chat2DB: {e}"
            logger.error(error_msg)
            return {"error": error_msg, "success": False}
    
    def disconnect(self) -> bool:
        """Disconnect from Chat2DB.
        
        Returns:
            True if disconnection successful, False otherwise
        """
        self.connection = None
        return True

class BigQueryConnector(BaseDBConnector):
    """Connector for Google BigQuery."""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.project_id = self.config.get("project_id", os.getenv("BIGQUERY_PROJECT_ID"))
        self.credentials_path = self.config.get("credentials_path", os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
    
    def connect(self) -> bool:
        """Connect to Google BigQuery.
        
        Returns:
            True if connection successful, False otherwise
        """
        if not self.project_id:
            logger.error("BigQuery project ID not configured")
            return False
        
        try:
            # Import here to avoid dependency issues if not using BigQuery
            from google.cloud import bigquery
            
            # Create client using default or specified credentials
            self.connection = bigquery.Client(project=self.project_id)
            logger.info(f"Connected to BigQuery project: {self.project_id}")
            return True
        except ImportError:
            logger.error("google-cloud-bigquery package not installed. Install with: pip install google-cloud-bigquery")
            return False
        except Exception as e:
            logger.error(f"Error connecting to BigQuery: {e}")
            return False
    
    def execute_query(self, query: str, params: Dict = None) -> Dict:
        """Execute a query on Google BigQuery.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            Query results
        """
        if not self.is_connected():
            logger.warning("Not connected to BigQuery, attempting to connect")
            if not self.connect():
                return {"error": "Not connected to BigQuery", "success": False}
        
        try:
            # Import here to avoid dependency issues if not using BigQuery
            from google.cloud import bigquery
            
            # Create job config with query parameters if provided
            job_config = bigquery.QueryJobConfig()
            
            if params:
                # Convert params to query parameters
                query_params = []
                for key, value in params.items():
                    if isinstance(value, str):
                        query_params.append(bigquery.ScalarQueryParameter(key, "STRING", value))
                    elif isinstance(value, int):
                        query_params.append(bigquery.ScalarQueryParameter(key, "INT64", value))
                    elif isinstance(value, float):
                        query_params.append(bigquery.ScalarQueryParameter(key, "FLOAT64", value))
                    elif isinstance(value, bool):
                        query_params.append(bigquery.ScalarQueryParameter(key, "BOOL", value))
                
                job_config.query_parameters = query_params
            
            # Execute query
            query_job = self.connection.query(query, job_config=job_config)
            results = query_job.result()
            
            # Convert to list of dicts
            rows = []
            for row in results:
                rows.append(dict(row.items()))
            
            return {"data": rows, "success": True}
        except Exception as e:
            error_msg = f"Error executing query on BigQuery: {e}"
            logger.error(error_msg)
            return {"error": error_msg, "success": False}
    
    def disconnect(self) -> bool:
        """Disconnect from Google BigQuery.
        
        Returns:
            True if disconnection successful, False otherwise
        """
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("Disconnected from BigQuery")
        return True

class AuroraConnector(BaseDBConnector):
    """Connector for Amazon Aurora."""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.host = self.config.get("host", os.getenv("AURORA_HOST"))
        self.port = self.config.get("port", os.getenv("AURORA_PORT", "3306"))
        self.database = self.config.get("database", os.getenv("AURORA_DATABASE"))
        self.user = self.config.get("user", os.getenv("AURORA_USER"))
        self.password = self.config.get("password", os.getenv("AURORA_PASSWORD"))
        self.ssl_ca = self.config.get("ssl_ca", os.getenv("AURORA_SSL_CA"))
    
    def connect(self) -> bool:
        """Connect to Amazon Aurora.
        
        Returns:
            True if connection successful, False otherwise
        """
        if not all([self.host, self.database, self.user, self.password]):
            logger.error("Aurora connection parameters not fully configured")
            return False
        
        try:
            # Import here to avoid dependency issues if not using Aurora
            import pymysql
            
            # SSL configuration if provided
            ssl_config = None
            if self.ssl_ca:
                ssl_config = {"ca": self.ssl_ca}
            
            # Connect to Aurora
            self.connection = pymysql.connect(
                host=self.host,
                port=int(self.port),
                user=self.user,
                password=self.password,
                database=self.database,
                ssl=ssl_config,
                cursorclass=pymysql.cursors.DictCursor
            )
            
            logger.info(f"Connected to Aurora database: {self.database}")
            return True
        except ImportError:
            logger.error("pymysql package not installed. Install with: pip install pymysql")
            return False
        except Exception as e:
            logger.error(f"Error connecting to Aurora: {e}")
            return False
    
    def execute_query(self, query: str, params: Dict = None) -> Dict:
        """Execute a query on Amazon Aurora.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            Query results
        """
        if not self.is_connected():
            logger.warning("Not connected to Aurora, attempting to connect")
            if not self.connect():
                return {"error": "Not connected to Aurora", "success": False}
        
        try:
            with self.connection.cursor() as cursor:
                # Execute query with parameters if provided
                cursor.execute(query, params or ())
                
                # Fetch results for SELECT queries
                if query.strip().lower().startswith("select"):
                    rows = cursor.fetchall()
                    return {"data": rows, "success": True}
                else:
                    # For non-SELECT queries, commit and return affected rows
                    self.connection.commit()
                    return {"affected_rows": cursor.rowcount, "success": True}
        except Exception as e:
            error_msg = f"Error executing query on Aurora: {e}"
            logger.error(error_msg)
            return {"error": error_msg, "success": False}
    
    def disconnect(self) -> bool:
        """Disconnect from Amazon Aurora.
        
        Returns:
            True if disconnection successful, False otherwise
        """
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("Disconnected from Aurora")
        return True

class CosmosDBConnector(BaseDBConnector):
    """Connector for Microsoft Azure Cosmos DB."""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.endpoint = self.config.get("endpoint", os.getenv("COSMOS_ENDPOINT"))
        self.key = self.config.get("key", os.getenv("COSMOS_KEY"))
        self.database_id = self.config.get("database_id", os.getenv("COSMOS_DATABASE"))
        self.container_id = self.config.get("container_id", os.getenv("COSMOS_CONTAINER"))
    
    def connect(self) -> bool:
        """Connect to Azure Cosmos DB.
        
        Returns:
            True if connection successful, False otherwise
        """
        if not all([self.endpoint, self.key, self.database_id]):
            logger.error("Cosmos DB connection parameters not fully configured")
            return False
        
        try:
            # Import here to avoid dependency issues if not using Cosmos DB
            from azure.cosmos import CosmosClient
            
            # Connect to Cosmos DB
            self.client = CosmosClient(self.endpoint, credential=self.key)
            self.database = self.client.get_database_client(self.database_id)
            
            # Connect to container if specified
            if self.container_id:
                self.container = self.database.get_container_client(self.container_id)
            
            self.connection = True
            logger.info(f"Connected to Cosmos DB database: {self.database_id}")
            return True
        except ImportError:
            logger.error("azure-cosmos package not installed. Install with: pip install azure-cosmos")
            return False
        except Exception as e:
            logger.error(f"Error connecting to Cosmos DB: {e}")
            return False
    
    def execute_query(self, query: str, params: Dict = None) -> Dict:
        """Execute a query on Azure Cosmos DB.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            Query results
        """
        if not self.is_connected():
            logger.warning("Not connected to Cosmos DB, attempting to connect")
            if not self.connect():
                return {"error": "Not connected to Cosmos DB", "success": False}
        
        try:
            # For Cosmos DB, we use the container for queries
            if not hasattr(self, 'container'):
                logger.error("No container specified for Cosmos DB query")
                return {"error": "No container specified", "success": False}
            
            # Execute query with parameters if provided
            if params:
                # Convert params to Cosmos DB parameter format
                parameters = [{"name": f"@{k}", "value": v} for k, v in params.items()]
                # Replace parameter placeholders in query
                for param in parameters:
                    query = query.replace(param["name"], str(param["value"]))
            
            # Execute query
            items = list(self.container.query_items(
                query=query,
                enable_cross_partition_query=True
            ))
            
            return {"data": items, "success": True}
        except Exception as e:
            error_msg = f"Error executing query on Cosmos DB: {e}"
            logger.error(error_msg)
            return {"error": error_msg, "success": False}
    
    def disconnect(self) -> bool:
        """Disconnect from Azure Cosmos DB.
        
        Returns:
            True if disconnection successful, False otherwise
        """
        self.connection = None
        logger.info("Disconnected from Cosmos DB")
        return True

class SnowflakeConnector(BaseDBConnector):
    """Connector for Snowflake."""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.account = self.config.get("account", os.getenv("SNOWFLAKE_ACCOUNT"))
        self.user = self.config.get("user", os.getenv("SNOWFLAKE_USER"))
        self.password = self.config.get("password", os.getenv("SNOWFLAKE_PASSWORD"))
        self.warehouse = self.config.get("warehouse", os.getenv("SNOWFLAKE_WAREHOUSE"))
        self.database = self.config.get("database", os.getenv("SNOWFLAKE_DATABASE"))
        self.schema = self.config.get("schema", os.getenv("SNOWFLAKE_SCHEMA"))
        self.role = self.config.get("role", os.getenv("SNOWFLAKE_ROLE"))
    
    def connect(self) -> bool:
        """Connect to Snowflake.
        
        Returns:
            True if connection successful, False otherwise
        """
        if not all([self.account, self.user, self.password]):
            logger.error("Snowflake connection parameters not fully configured")
            return False
        
        try:
            # Import here to avoid dependency issues if not using Snowflake
            import snowflake.connector
            
            # Connect to Snowflake
            self.connection = snowflake.connector.connect(
                user=self.user,
                password=self.password,
                account=self.account,
                warehouse=self.warehouse,
                database=self.database,
                schema=self.schema,
                role=self.role
            )
            
            logger.info(f"Connected to Snowflake account: {self.account}")
            return True
        except ImportError:
            logger.error("snowflake-connector-python package not installed. Install with: pip install snowflake-connector-python")
            return False
        except Exception as e:
            logger.error(f"Error connecting to Snowflake: {e}")
            return False
    
    def execute_query(self, query: str, params: Dict = None) -> Dict:
        """Execute a query on Snowflake.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            Query results
        """
        if not self.is_connected():
            logger.warning("Not connected to Snowflake, attempting to connect")
            if not self.connect():
                return {"error": "Not connected to Snowflake", "success": False}
        
        try:
            cursor = self.connection.cursor(as_dict=True)
            
            # Execute query with parameters if provided
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            # Fetch results for SELECT queries
            if query.strip().lower().startswith("select"):
                rows = cursor.fetchall()
                cursor.close()
                return {"data": rows, "success": True}
            else:
                # For non-SELECT queries, commit and return affected rows
                self.connection.commit()
                affected_rows = cursor.rowcount
                cursor.close()
                return {"affected_rows": affected_rows, "success": True}
        except Exception as e:
            error_msg = f"Error executing query on Snowflake: {e}"
            logger.error(error_msg)
            return {"error": error_msg, "success": False}
    
    def disconnect(self) -> bool:
        """Disconnect from Snowflake.
        
        Returns:
            True if disconnection successful, False otherwise
        """
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("Disconnected from Snowflake")
        return True

class DB2AIConnector(BaseDBConnector):
    """Connector for IBM Db2 AI."""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.host = self.config.get("host", os.getenv("DB2AI_HOST"))
        self.port = self.config.get("port", os.getenv("DB2AI_PORT", "50000"))
        self.database = self.config.get("database", os.getenv("DB2AI_DATABASE"))
        self.user = self.config.get("user", os.getenv("DB2AI_USER"))
        self.password = self.config.get("password", os.getenv("DB2AI_PASSWORD"))
        self.schema = self.config.get("schema", os.getenv("DB2AI_SCHEMA", "DB2INST1"))
    
    def connect(self) -> bool:
        """Connect to IBM Db2 AI.
        
        Returns:
            True if connection successful, False otherwise
        """
        if not all([self.host, self.database, self.user, self.password]):
            logger.error("Db2 connection parameters not fully configured")
            return False
        
        try:
            # Import here to avoid dependency issues if not using Db2
            import ibm_db
            import ibm_db_dbi
            
            # Create connection string
            conn_string = f"DATABASE={self.database};HOSTNAME={self.host};PORT={self.port};PROTOCOL=TCPIP;UID={self.user};PWD={self.password};"
            
            # Connect to Db2
            ibm_db_conn = ibm_db.connect(conn_string, "", "")
            self.connection = ibm_db_dbi.Connection(ibm_db_conn)
            
            logger.info(f"Connected to Db2 database: {self.database}")
            return True
        except ImportError:
            logger.error("ibm_db and ibm_db_dbi packages not installed. Install with: pip install ibm_db ibm_db_dbi")
            return False
        except Exception as e:
            logger.error(f"Error connecting to Db2: {e}")
            return False
    
    def execute_query(self, query: str, params: Dict = None) -> Dict:
        """Execute a query on IBM Db2 AI.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            Query results
        """
        if not self.is_connected():
            logger.warning("Not connected to Db2, attempting to connect")
            if not self.connect():
                return {"error": "Not connected to Db2", "success": False}
        
        try:
            cursor = self.connection.cursor()
            
            # Execute query with parameters if provided
            if params:
                # Convert dictionary params to sequence
                param_seq = tuple(params.values())
                cursor.execute(query, param_seq)
            else:
                cursor.execute(query)
            
            # Fetch results for SELECT queries
            if query.strip().lower().startswith("select"):
                # Get column names
                columns = [column[0] for column in cursor.description]
                
                # Fetch all rows
                rows = cursor.fetchall()
                
                # Convert to list of dictionaries
                results = []
                for row in rows:
                    results.append(dict(zip(columns, row)))
                
                cursor.close()
                return {"data": results, "success": True}
            else:
                # For non-SELECT queries, commit and return affected rows
                self.connection.commit()
                affected_rows = cursor.rowcount
                cursor.close()
                return {"affected_rows": affected_rows, "success": True}
        except Exception as e:
            error_msg = f"Error executing query on Db2: {e}"
            logger.error(error_msg)
            return {"error": error_msg, "success": False}
    
    def disconnect(self) -> bool:
        """Disconnect from IBM Db2 AI.
        
        Returns:
            True if disconnection successful, False otherwise
        """
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("Disconnected from Db2")
        return True

# Factory function to create database connectors
def create_db_connector(db_type: str, config: Dict = None) -> BaseDBConnector:
    """Create a database connector based on the specified type.
    
    Args:
        db_type: Type of database connector to create
        config: Configuration for the connector
        
    Returns:
        Database connector instance
    """
    connectors = {
        "chat2db": Chat2DBConnector,
        "bigquery": BigQueryConnector,
        "aurora": AuroraConnector,
        "cosmosdb": CosmosDBConnector,
        "snowflake": SnowflakeConnector,
        "db2ai": DB2AIConnector,
        # Dataset connectors are now in dataset_connectors.py
        "kaggle": "dataset",
        "google_dataset_search": "dataset",
        "uci_ml": "dataset",
        "imagenet": "dataset",
        "common_crawl": "dataset",
        "huggingface": "dataset",
        "datagov": "dataset",
        "zenodo": "dataset",
        "arxiv": "dataset"
    }
    
    connector_type = connectors.get(db_type.lower())
    
    if connector_type == "dataset":
        logger.info(f"Dataset connector {db_type} requested, use dataset_connectors.py instead")
        try:
            from dataset_connectors import create_dataset_connector
            return create_dataset_connector(db_type, config)
        except ImportError:
            logger.error("dataset_connectors module not found")
            return None
    elif connector_type:
        return connector_type(config)
    else:
        logger.warning(f"Unknown database type: {db_type}")
        return None 