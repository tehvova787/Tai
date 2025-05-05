"""
Dataset Connectors Usage Examples

This module demonstrates how to use the dataset connectors to access and download
various public datasets.
"""

import logging
import os
import json
from typing import Dict
from dataset_connectors import (
    create_dataset_connector,
    KaggleConnector,
    GoogleDatasetSearchConnector,
    UCIMLRepositoryConnector,
    ImageNetConnector,
    CommonCrawlConnector,
    HuggingFaceDatasetConnector,
    DataGovConnector,
    ZenodoConnector,
    ArXivDatasetConnector
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def demonstrate_kaggle_connector():
    """Demonstrate using the Kaggle connector."""
    logger.info("=== Demonstrating Kaggle Connector ===")
    
    # Create connector
    connector = create_dataset_connector("kaggle")
    
    # Connect
    if not connector.connect():
        logger.error("Failed to connect to Kaggle")
        return
    
    # Search for datasets
    search_results = connector.search("climate change", limit=5)
    
    if not search_results.get("success"):
        logger.error(f"Search failed: {search_results.get('error')}")
        return
    
    # Display results
    logger.info(f"Found {len(search_results['data'])} datasets")
    for i, dataset in enumerate(search_results["data"]):
        logger.info(f"{i+1}. {dataset['title']} - {dataset['url']}")
    
    # Download first dataset
    if search_results["data"]:
        dataset_id = search_results["data"][0]["id"]
        logger.info(f"Downloading dataset: {dataset_id}")
        
        download_result = connector.download(dataset_id)
        
        if download_result.get("success"):
            logger.info(f"Downloaded to: {download_result['path']}")
            logger.info(f"Files: {download_result['files']}")
        else:
            logger.error(f"Download failed: {download_result.get('error')}")

def demonstrate_huggingface_connector():
    """Demonstrate using the HuggingFace connector."""
    logger.info("=== Demonstrating HuggingFace Connector ===")
    
    # Create connector
    connector = create_dataset_connector("huggingface")
    
    # Connect
    if not connector.connect():
        logger.error("Failed to connect to HuggingFace")
        return
    
    # Search for datasets
    search_results = connector.search("squad", limit=5)
    
    if not search_results.get("success"):
        logger.error(f"Search failed: {search_results.get('error')}")
        return
    
    # Display results
    logger.info(f"Found {len(search_results['data'])} datasets")
    for i, dataset in enumerate(search_results["data"]):
        logger.info(f"{i+1}. {dataset['id']} - {dataset['url']}")
    
    # Download a small dataset
    logger.info("Downloading dataset: squad_v2 (small sample)")
    
    download_result = connector.download("squad_v2", path="./data/huggingface/squad_v2_sample")
    
    if download_result.get("success"):
        logger.info(f"Downloaded info: {download_result['data']}")
    else:
        logger.error(f"Download failed: {download_result.get('error')}")

def demonstrate_uci_connector():
    """Demonstrate using the UCI ML Repository connector."""
    logger.info("=== Demonstrating UCI ML Repository Connector ===")
    
    # Create connector
    connector = create_dataset_connector("uci_ml")
    
    # Connect
    if not connector.connect():
        logger.error("Failed to connect to UCI ML Repository")
        return
    
    # Search for datasets
    search_results = connector.search("iris", limit=5)
    
    if not search_results.get("success"):
        logger.error(f"Search failed: {search_results.get('error')}")
        return
    
    # Display results
    logger.info(f"Found {len(search_results['data'])} datasets")
    for i, dataset in enumerate(search_results["data"]):
        logger.info(f"{i+1}. {dataset['name']} - {dataset['url']}")
    
    # Download first dataset
    if search_results["data"]:
        dataset_id = search_results["data"][0]["id"]
        logger.info(f"Downloading dataset: {dataset_id}")
        
        download_result = connector.download(dataset_id)
        
        if download_result.get("success"):
            logger.info(f"Downloaded to: {download_result['path']}")
            logger.info(f"Files: {download_result['files']}")
        else:
            logger.error(f"Download failed: {download_result.get('error')}")

def demonstrate_arxiv_connector():
    """Demonstrate using the arXiv connector."""
    logger.info("=== Demonstrating arXiv Connector ===")
    
    # Create connector
    connector = create_dataset_connector("arxiv")
    
    # Connect
    if not connector.connect():
        logger.error("Failed to connect to arXiv")
        return
    
    # Search for dataset papers
    search_results = connector.search("image dataset", limit=5)
    
    if not search_results.get("success"):
        logger.error(f"Search failed: {search_results.get('error')}")
        return
    
    # Display results
    logger.info(f"Found {len(search_results['data'])} papers")
    for i, paper in enumerate(search_results["data"]):
        logger.info(f"{i+1}. {paper['title']} - {paper['abstract_url']}")
    
    # Download first paper
    if search_results["data"]:
        paper_id = search_results["data"][0]["id"]
        logger.info(f"Downloading paper: {paper_id}")
        
        download_result = connector.download(paper_id)
        
        if download_result.get("success"):
            logger.info(f"Downloaded to: {download_result['path']}")
            logger.info(f"Files: {download_result['files']}")
        else:
            logger.error(f"Download failed: {download_result.get('error')}")

def demonstrate_all_connectors():
    """Demonstrate basic functionality of all connectors."""
    connectors = [
        "kaggle",
        "google_dataset_search",
        "uci_ml",
        "huggingface",
        "arxiv",
        "datagov",
        "zenodo"
    ]
    
    for connector_type in connectors:
        logger.info(f"=== Testing {connector_type} Connector ===")
        
        # Create connector
        connector = create_dataset_connector(connector_type)
        
        if not connector:
            logger.error(f"Failed to create {connector_type} connector")
            continue
        
        # Connect
        if connector.connect():
            logger.info(f"Successfully connected to {connector_type}")
            
            # Search
            search_results = connector.search("climate", limit=3)
            
            if search_results.get("success"):
                logger.info(f"Search successful, found {len(search_results.get('data', []))} results")
            else:
                logger.error(f"Search failed: {search_results.get('error')}")
        else:
            logger.error(f"Failed to connect to {connector_type}")
        
        logger.info("\n")

def load_dataset_from_config(config_file: str) -> Dict:
    """Load datasets based on configuration.
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        Result of loading datasets
    """
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        datasets_config = config.get("data_source_settings", {})
        results = {}
        
        for dataset_type, settings in datasets_config.items():
            if settings.get("enabled", False):
                logger.info(f"Loading dataset from {dataset_type}")
                
                connector = create_dataset_connector(dataset_type, settings)
                
                if connector and connector.connect():
                    # Search for a dataset
                    search_query = settings.get("default_search", "machine learning")
                    search_results = connector.search(search_query, limit=1)
                    
                    if search_results.get("success") and search_results.get("data"):
                        dataset_id = search_results["data"][0].get("id")
                        
                        # Download the dataset
                        download_result = connector.download(dataset_id)
                        
                        results[dataset_type] = {
                            "success": download_result.get("success", False),
                            "path": download_result.get("path"),
                            "dataset_id": dataset_id
                        }
                    else:
                        results[dataset_type] = {
                            "success": False,
                            "error": "No datasets found"
                        }
                else:
                    results[dataset_type] = {
                        "success": False,
                        "error": "Failed to connect"
                    }
        
        return results
    except Exception as e:
        logger.error(f"Error loading datasets from config: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    # Create data directory
    os.makedirs("./data", exist_ok=True)
    
    # Uncomment the example you want to run
    # demonstrate_kaggle_connector()
    # demonstrate_huggingface_connector()
    # demonstrate_uci_connector()
    # demonstrate_arxiv_connector()
    demonstrate_all_connectors()
    
    # Load from config file
    # config_path = "../config/config.json"  # Adjust path as needed
    # results = load_dataset_from_config(config_path)
    # logger.info(f"Config loading results: {json.dumps(results, indent=2)}") 