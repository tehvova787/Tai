"""
Dataset Connectors for Lucky Train AI Assistant

This module provides connectors to various public datasets and knowledge bases:
- Kaggle (kaggle.com/datasets)
- Google Dataset Search (datasetsearch.research.google.com)
- UCI Machine Learning Repository (archive.ics.uci.edu/ml/)
- ImageNet (image-net.org)
- Common Crawl (commoncrawl.org)
- HuggingFace Datasets (huggingface.co/datasets)
- Data.gov
- Zenodo (zenodo.org)
- arXiv Dataset (arxiv.org)
"""

import logging
import os
import json
import requests
import tempfile
import zipfile
import pandas as pd
from typing import Dict, List, Optional, Union, Any
from abc import ABC, abstractmethod
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class BaseDatasetConnector(ABC):
    """Base class for all dataset connectors."""
    
    def __init__(self, config: Dict = None):
        """Initialize the dataset connector.
        
        Args:
            config: Configuration for the connector
        """
        self.config = config or {}
        self.name = self.__class__.__name__
        self.connected = False
        logger.info(f"Initializing {self.name}")
    
    @abstractmethod
    def connect(self) -> bool:
        """Connect to the dataset source.
        
        Returns:
            True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def search(self, query: str, limit: int = 10) -> Dict:
        """Search for datasets.
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            
        Returns:
            Search results
        """
        pass
    
    @abstractmethod
    def download(self, dataset_id: str, path: str = None) -> Dict:
        """Download a dataset.
        
        Args:
            dataset_id: ID of the dataset to download
            path: Path to save the dataset to
            
        Returns:
            Download status
        """
        pass
    
    def is_connected(self) -> bool:
        """Check if connected to the dataset source.
        
        Returns:
            True if connected, False otherwise
        """
        return self.connected

class KaggleConnector(BaseDatasetConnector):
    """Connector for Kaggle datasets."""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.username = self.config.get("username", os.getenv("KAGGLE_USERNAME"))
        self.api_key = self.config.get("api_key", os.getenv("KAGGLE_KEY"))
        self.cache_dir = self.config.get("cache_dir", "./data/kaggle")
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def connect(self) -> bool:
        """Connect to Kaggle.
        
        Returns:
            True if connection successful, False otherwise
        """
        if not all([self.username, self.api_key]):
            logger.error("Kaggle API credentials not configured. Set KAGGLE_USERNAME and KAGGLE_KEY environment variables.")
            return False
        
        try:
            # Import here to avoid dependency issues if not using Kaggle
            import kaggle
            
            # Set up Kaggle API credentials
            os.environ["KAGGLE_USERNAME"] = self.username
            os.environ["KAGGLE_KEY"] = self.api_key
            
            # Test authentication by listing datasets
            kaggle.api.authenticate()
            _ = kaggle.api.dataset_list(search="test", max_size=1)
            
            self.connected = True
            logger.info("Connected to Kaggle successfully")
            return True
        except ImportError:
            logger.error("kaggle package not installed. Install with: pip install kaggle")
            return False
        except Exception as e:
            logger.error(f"Error connecting to Kaggle: {e}")
            return False
    
    def search(self, query: str, limit: int = 10) -> Dict:
        """Search for datasets on Kaggle.
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            
        Returns:
            Search results
        """
        if not self.is_connected():
            logger.warning("Not connected to Kaggle, attempting to connect")
            if not self.connect():
                return {"error": "Not connected to Kaggle", "success": False}
        
        try:
            # Import here to avoid dependency issues if not using Kaggle
            import kaggle
            
            # Search for datasets
            datasets = kaggle.api.dataset_list(search=query, max_size=limit)
            
            results = []
            for dataset in datasets:
                results.append({
                    "id": dataset.ref,
                    "title": dataset.title,
                    "size": dataset.size,
                    "last_updated": dataset.lastUpdated,
                    "download_count": dataset.downloadCount,
                    "votes": dataset.voteCount,
                    "url": f"https://www.kaggle.com/datasets/{dataset.ref}"
                })
            
            return {"data": results, "success": True}
        except Exception as e:
            error_msg = f"Error searching Kaggle datasets: {e}"
            logger.error(error_msg)
            return {"error": error_msg, "success": False}
    
    def download(self, dataset_id: str, path: str = None) -> Dict:
        """Download a dataset from Kaggle.
        
        Args:
            dataset_id: ID of the dataset to download (format: owner/dataset-name)
            path: Path to save the dataset to
            
        Returns:
            Download status
        """
        if not self.is_connected():
            logger.warning("Not connected to Kaggle, attempting to connect")
            if not self.connect():
                return {"error": "Not connected to Kaggle", "success": False}
        
        try:
            # Import here to avoid dependency issues if not using Kaggle
            import kaggle
            
            # Set download path
            download_path = path or os.path.join(self.cache_dir, dataset_id.replace("/", "_"))
            os.makedirs(download_path, exist_ok=True)
            
            # Download dataset
            kaggle.api.dataset_download_files(dataset_id, path=download_path, unzip=True)
            
            # List downloaded files
            files = os.listdir(download_path)
            
            return {
                "success": True, 
                "path": download_path, 
                "files": files,
                "dataset_id": dataset_id
            }
        except Exception as e:
            error_msg = f"Error downloading Kaggle dataset: {e}"
            logger.error(error_msg)
            return {"error": error_msg, "success": False}
    
    def get_metadata(self, dataset_id: str) -> Dict:
        """Get metadata for a Kaggle dataset.
        
        Args:
            dataset_id: ID of the dataset (format: owner/dataset-name)
            
        Returns:
            Dataset metadata
        """
        if not self.is_connected():
            logger.warning("Not connected to Kaggle, attempting to connect")
            if not self.connect():
                return {"error": "Not connected to Kaggle", "success": False}
        
        try:
            # Import here to avoid dependency issues if not using Kaggle
            import kaggle
            
            # Get dataset metadata
            dataset = kaggle.api.dataset_view(dataset_id)
            
            metadata = {
                "id": dataset.ref,
                "title": dataset.title,
                "subtitle": dataset.subtitle,
                "description": dataset.description,
                "size": dataset.size,
                "last_updated": dataset.lastUpdated,
                "download_count": dataset.downloadCount,
                "votes": dataset.voteCount,
                "tags": [tag.name for tag in dataset.tags],
                "license": dataset.licenses[0].name if dataset.licenses else None,
                "url": f"https://www.kaggle.com/datasets/{dataset.ref}"
            }
            
            return {"data": metadata, "success": True}
        except Exception as e:
            error_msg = f"Error getting Kaggle dataset metadata: {e}"
            logger.error(error_msg)
            return {"error": error_msg, "success": False}

class GoogleDatasetSearchConnector(BaseDatasetConnector):
    """Connector for Google Dataset Search."""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.api_key = self.config.get("api_key", os.getenv("GOOGLE_API_KEY"))
        self.custom_search_engine_id = self.config.get("custom_search_engine_id", os.getenv("GOOGLE_CSE_ID"))
        self.search_base_url = "https://www.googleapis.com/customsearch/v1"
    
    def connect(self) -> bool:
        """Connect to Google Dataset Search.
        
        Returns:
            True if connection successful, False otherwise
        """
        if not all([self.api_key, self.custom_search_engine_id]):
            logger.error("Google API credentials not configured. Set GOOGLE_API_KEY and GOOGLE_CSE_ID environment variables.")
            return False
        
        try:
            # Test connection with a simple search
            params = {
                "key": self.api_key,
                "cx": self.custom_search_engine_id,
                "q": "test dataset",
                "num": 1
            }
            
            response = requests.get(self.search_base_url, params=params)
            
            if response.status_code == 200:
                self.connected = True
                logger.info("Connected to Google Dataset Search successfully")
                return True
            else:
                logger.error(f"Failed to connect to Google Dataset Search: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error connecting to Google Dataset Search: {e}")
            return False
    
    def search(self, query: str, limit: int = 10) -> Dict:
        """Search for datasets using Google Dataset Search.
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            
        Returns:
            Search results
        """
        if not self.is_connected():
            logger.warning("Not connected to Google Dataset Search, attempting to connect")
            if not self.connect():
                return {"error": "Not connected to Google Dataset Search", "success": False}
        
        try:
            # Append "dataset" to the query to focus on datasets
            search_query = f"{query} dataset"
            
            # Set up search parameters
            params = {
                "key": self.api_key,
                "cx": self.custom_search_engine_id,
                "q": search_query,
                "num": min(limit, 10)  # Google CSE has a max of 10 results per page
            }
            
            # Execute search
            response = requests.get(self.search_base_url, params=params)
            
            if response.status_code != 200:
                return {"error": f"Search failed: {response.status_code} - {response.text}", "success": False}
            
            results_data = response.json()
            
            # Format results
            results = []
            if "items" in results_data:
                for item in results_data["items"]:
                    results.append({
                        "title": item.get("title"),
                        "url": item.get("link"),
                        "snippet": item.get("snippet"),
                        "source": item.get("displayLink"),
                        "id": item.get("cacheId", item.get("link"))
                    })
            
            return {"data": results, "success": True}
        except Exception as e:
            error_msg = f"Error searching Google Dataset Search: {e}"
            logger.error(error_msg)
            return {"error": error_msg, "success": False}
    
    def download(self, dataset_id: str, path: str = None) -> Dict:
        """Download a dataset from Google Dataset Search.
        
        Note: This is not directly supported as Google Dataset Search is a search engine.
        This method attempts to follow the URL and download if possible.
        
        Args:
            dataset_id: URL of the dataset to download
            path: Path to save the dataset to
            
        Returns:
            Download status
        """
        if not dataset_id.startswith("http"):
            return {"error": "Dataset ID must be a URL", "success": False}
        
        try:
            # Create download directory
            download_path = path or tempfile.mkdtemp(prefix="google_dataset_")
            os.makedirs(download_path, exist_ok=True)
            
            # Follow URL and download if possible
            response = requests.get(dataset_id, stream=True)
            
            if response.status_code != 200:
                return {"error": f"Failed to download dataset: {response.status_code}", "success": False}
            
            # Try to get filename from headers
            content_disposition = response.headers.get("content-disposition")
            if content_disposition:
                import re
                filename = re.findall("filename=(.+)", content_disposition)
                if filename:
                    filename = filename[0].strip('"\'')
                else:
                    filename = "dataset.data"
            else:
                filename = "dataset.data"
            
            # Save file
            file_path = os.path.join(download_path, filename)
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Try to extract if it's a zip file
            if filename.endswith(".zip"):
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(download_path)
            
            # List downloaded files
            files = os.listdir(download_path)
            
            return {
                "success": True, 
                "path": download_path, 
                "files": files,
                "dataset_id": dataset_id
            }
        except Exception as e:
            error_msg = f"Error downloading dataset from URL: {e}"
            logger.error(error_msg)
            return {"error": error_msg, "success": False}

class UCIMLRepositoryConnector(BaseDatasetConnector):
    """Connector for UCI Machine Learning Repository."""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.api_endpoint = "https://archive.ics.uci.edu/api/dataset/"
        self.search_endpoint = "https://archive.ics.uci.edu/api/datasets/"
        self.cache_dir = self.config.get("cache_dir", "./data/uci")
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def connect(self) -> bool:
        """Connect to UCI ML Repository.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Test connection with a simple request
            response = requests.get(self.search_endpoint + "?limit=1")
            
            if response.status_code == 200:
                self.connected = True
                logger.info("Connected to UCI ML Repository successfully")
                return True
            else:
                logger.error(f"Failed to connect to UCI ML Repository: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error connecting to UCI ML Repository: {e}")
            return False
    
    def search(self, query: str, limit: int = 10) -> Dict:
        """Search for datasets in UCI ML Repository.
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            
        Returns:
            Search results
        """
        if not self.is_connected():
            logger.warning("Not connected to UCI ML Repository, attempting to connect")
            if not self.connect():
                return {"error": "Not connected to UCI ML Repository", "success": False}
        
        try:
            # Set up search parameters
            params = {
                "q": query,
                "limit": limit
            }
            
            # Execute search
            response = requests.get(self.search_endpoint, params=params)
            
            if response.status_code != 200:
                return {"error": f"Search failed: {response.status_code} - {response.text}", "success": False}
            
            results_data = response.json()
            
            # Format results
            results = []
            for item in results_data.get("data", []):
                results.append({
                    "id": item.get("id"),
                    "name": item.get("name"),
                    "abstract": item.get("abstract"),
                    "area": item.get("area"),
                    "url": f"https://archive.ics.uci.edu/dataset/{item.get('id')}",
                    "num_instances": item.get("num_instances"),
                    "num_features": item.get("num_features"),
                    "year": item.get("year")
                })
            
            return {"data": results, "success": True}
        except Exception as e:
            error_msg = f"Error searching UCI ML Repository: {e}"
            logger.error(error_msg)
            return {"error": error_msg, "success": False}
    
    def download(self, dataset_id: str, path: str = None) -> Dict:
        """Download a dataset from UCI ML Repository.
        
        Args:
            dataset_id: ID of the dataset to download
            path: Path to save the dataset to
            
        Returns:
            Download status
        """
        if not self.is_connected():
            logger.warning("Not connected to UCI ML Repository, attempting to connect")
            if not self.connect():
                return {"error": "Not connected to UCI ML Repository", "success": False}
        
        try:
            # Get dataset information
            response = requests.get(f"{self.api_endpoint}{dataset_id}")
            
            if response.status_code != 200:
                return {"error": f"Failed to get dataset information: {response.status_code}", "success": False}
            
            dataset_info = response.json()
            
            # Create download directory
            download_path = path or os.path.join(self.cache_dir, f"uci_{dataset_id}")
            os.makedirs(download_path, exist_ok=True)
            
            # Download files
            files = []
            for file_info in dataset_info.get("files", []):
                file_url = file_info["url"]
                file_name = file_info["name"]
                file_path = os.path.join(download_path, file_name)
                
                # Download file
                file_response = requests.get(file_url, stream=True)
                if file_response.status_code == 200:
                    with open(file_path, 'wb') as f:
                        for chunk in file_response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    files.append(file_name)
                else:
                    logger.warning(f"Failed to download file {file_name}: {file_response.status_code}")
            
            return {
                "success": True, 
                "path": download_path, 
                "files": files,
                "dataset_id": dataset_id,
                "dataset_name": dataset_info.get("name")
            }
        except Exception as e:
            error_msg = f"Error downloading UCI ML Repository dataset: {e}"
            logger.error(error_msg)
            return {"error": error_msg, "success": False}

class ImageNetConnector(BaseDatasetConnector):
    """Connector for ImageNet dataset."""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.username = self.config.get("username", os.getenv("IMAGENET_USERNAME"))
        self.access_key = self.config.get("access_key", os.getenv("IMAGENET_ACCESS_KEY"))
        self.api_endpoint = "http://www.image-net.org/api/text/"
        self.cache_dir = self.config.get("cache_dir", "./data/imagenet")
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def connect(self) -> bool:
        """Connect to ImageNet.
        
        Returns:
            True if connection successful, False otherwise
        """
        if not all([self.username, self.access_key]):
            logger.error("ImageNet credentials not configured. Set IMAGENET_USERNAME and IMAGENET_ACCESS_KEY environment variables.")
            return False
        
        try:
            # Test connection with a simple request
            params = {
                "username": self.username,
                "accesskey": self.access_key,
                "release": "latest",
                "wnid": "n02123045"  # Cat synset
            }
            
            response = requests.get(f"{self.api_endpoint}geturls", params=params)
            
            if response.status_code == 200 and not response.text.startswith("Error"):
                self.connected = True
                logger.info("Connected to ImageNet successfully")
                return True
            else:
                logger.error(f"Failed to connect to ImageNet: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error connecting to ImageNet: {e}")
            return False
    
    def search(self, query: str, limit: int = 10) -> Dict:
        """Search for synsets in ImageNet.
        
        Args:
            query: Search keyword
            limit: Maximum number of results to return
            
        Returns:
            Search results
        """
        if not self.is_connected():
            logger.warning("Not connected to ImageNet, attempting to connect")
            if not self.connect():
                return {"error": "Not connected to ImageNet", "success": False}
        
        try:
            # Search for WordNet synsets by keywords
            params = {
                "username": self.username,
                "accesskey": self.access_key,
                "release": "latest",
                "words": query
            }
            
            response = requests.get(f"{self.api_endpoint}search", params=params)
            
            if response.status_code != 200 or response.text.startswith("Error"):
                return {"error": f"Search failed: {response.text}", "success": False}
            
            # Parse results (tab-separated synset IDs and descriptions)
            synsets = []
            for line in response.text.strip().split('\n')[:limit]:
                if '\t' in line:
                    wnid, description = line.strip().split('\t', 1)
                    synsets.append({
                        "id": wnid,
                        "description": description,
                        "url": f"http://image-net.org/synset?wnid={wnid}"
                    })
            
            return {"data": synsets, "success": True}
        except Exception as e:
            error_msg = f"Error searching ImageNet: {e}"
            logger.error(error_msg)
            return {"error": error_msg, "success": False}
    
    def download(self, dataset_id: str, path: str = None) -> Dict:
        """Download images for a synset from ImageNet.
        
        Args:
            dataset_id: WordNet ID (wnid) of the synset to download
            path: Path to save the images to
            
        Returns:
            Download status
        """
        if not self.is_connected():
            logger.warning("Not connected to ImageNet, attempting to connect")
            if not self.connect():
                return {"error": "Not connected to ImageNet", "success": False}
        
        try:
            # Create download directory
            download_path = path or os.path.join(self.cache_dir, dataset_id)
            os.makedirs(download_path, exist_ok=True)
            
            # Get URLs for the synset
            params = {
                "username": self.username,
                "accesskey": self.access_key,
                "release": "latest",
                "wnid": dataset_id
            }
            
            response = requests.get(f"{self.api_endpoint}geturls", params=params)
            
            if response.status_code != 200 or response.text.startswith("Error"):
                return {"error": f"Failed to get URLs: {response.text}", "success": False}
            
            # Parse URLs and download images
            urls = [line.strip() for line in response.text.strip().split('\n') if line.strip()]
            
            downloaded_files = []
            # Download up to 100 images to avoid too many requests
            for i, url in enumerate(urls[:100]):
                try:
                    img_response = requests.get(url, stream=True, timeout=5)
                    if img_response.status_code == 200:
                        file_name = f"{dataset_id}_{i}.jpg"
                        file_path = os.path.join(download_path, file_name)
                        
                        with open(file_path, 'wb') as f:
                            for chunk in img_response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        
                        downloaded_files.append(file_name)
                except Exception as e:
                    logger.warning(f"Failed to download image from {url}: {e}")
            
            return {
                "success": True, 
                "path": download_path, 
                "files": downloaded_files,
                "dataset_id": dataset_id,
                "total_urls": len(urls),
                "downloaded": len(downloaded_files)
            }
        except Exception as e:
            error_msg = f"Error downloading ImageNet synset: {e}"
            logger.error(error_msg)
            return {"error": error_msg, "success": False}

class CommonCrawlConnector(BaseDatasetConnector):
    """Connector for Common Crawl dataset."""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.api_endpoint = "https://index.commoncrawl.org/"
        self.cache_dir = self.config.get("cache_dir", "./data/commoncrawl")
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Default to the latest index
        self.index = self.config.get("index", "CC-MAIN-2023-50")
    
    def connect(self) -> bool:
        """Connect to Common Crawl.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Test connection by getting the list of available indexes
            response = requests.get("https://index.commoncrawl.org/collinfo.json")
            
            if response.status_code == 200:
                # Update the index to the latest if available
                indexes = response.json()
                if indexes:
                    self.index = indexes[0]["id"]
                
                self.connected = True
                logger.info(f"Connected to Common Crawl successfully, using index: {self.index}")
                return True
            else:
                logger.error(f"Failed to connect to Common Crawl: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error connecting to Common Crawl: {e}")
            return False
    
    def search(self, query: str, limit: int = 10) -> Dict:
        """Search for URLs in Common Crawl index.
        
        Args:
            query: URL pattern or domain to search for
            limit: Maximum number of results to return
            
        Returns:
            Search results
        """
        if not self.is_connected():
            logger.warning("Not connected to Common Crawl, attempting to connect")
            if not self.connect():
                return {"error": "Not connected to Common Crawl", "success": False}
        
        try:
            # Search for URLs in the index
            params = {
                "url": query,
                "output": "json",
                "limit": limit
            }
            
            response = requests.get(f"https://index.commoncrawl.org/{self.index}-index", params=params)
            
            if response.status_code != 200:
                return {"error": f"Search failed: {response.status_code} - {response.text}", "success": False}
            
            # Parse results (each line is a JSON object)
            results = []
            for line in response.text.strip().split('\n'):
                try:
                    if line:
                        result = json.loads(line)
                        results.append({
                            "url": result.get("url"),
                            "mime": result.get("mime"),
                            "status": result.get("status"),
                            "digest": result.get("digest"),
                            "filename": result.get("filename"),
                            "offset": result.get("offset"),
                            "length": result.get("length")
                        })
                except json.JSONDecodeError:
                    continue
            
            return {"data": results, "success": True}
        except Exception as e:
            error_msg = f"Error searching Common Crawl: {e}"
            logger.error(error_msg)
            return {"error": error_msg, "success": False}
    
    def download(self, dataset_id: str, path: str = None) -> Dict:
        """Download a specific record from Common Crawl.
        
        Args:
            dataset_id: JSON stringified object with filename, offset, and length
            path: Path to save the data to
            
        Returns:
            Download status
        """
        if not self.is_connected():
            logger.warning("Not connected to Common Crawl, attempting to connect")
            if not self.connect():
                return {"error": "Not connected to Common Crawl", "success": False}
        
        try:
            # Parse the dataset_id
            try:
                record_info = json.loads(dataset_id)
            except json.JSONDecodeError:
                return {"error": "Invalid dataset_id format, must be JSON with filename, offset, and length", "success": False}
            
            if not all(k in record_info for k in ["filename", "offset", "length"]):
                return {"error": "Missing required fields in dataset_id: filename, offset, length", "success": False}
            
            # Create download directory
            download_path = path or os.path.join(self.cache_dir, "records")
            os.makedirs(download_path, exist_ok=True)
            
            # Format the WARC record URL
            filename = record_info["filename"]
            offset = record_info["offset"]
            length = record_info["length"]
            
            # Calculate CID from filename
            # Example: CC-MAIN-2023-50/segments/1701166149653.92/warc/CC-MAIN-20231129183925-20231129213925-00294.warc.gz
            parts = filename.split('/')
            cid = parts[0] if parts else self.index
            
            # Build S3 URL
            s3_url = f"https://data.commoncrawl.org/{filename}"
            
            # Request range of bytes
            headers = {
                "Range": f"bytes={offset}-{offset + length - 1}"
            }
            
            response = requests.get(s3_url, headers=headers, stream=True)
            
            if response.status_code not in [200, 206]:
                return {"error": f"Failed to download WARC record: {response.status_code}", "success": False}
            
            # Generate a filename for the record
            digest = record_info.get("digest", "record")
            record_filename = f"{digest}_{offset}.warc"
            file_path = os.path.join(download_path, record_filename)
            
            # Save the WARC record
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return {
                "success": True, 
                "path": file_path, 
                "filename": record_filename,
                "size": length,
                "s3_url": s3_url
            }
        except Exception as e:
            error_msg = f"Error downloading Common Crawl record: {e}"
            logger.error(error_msg)
            return {"error": error_msg, "success": False}

class HuggingFaceDatasetConnector(BaseDatasetConnector):
    """Connector for HuggingFace datasets."""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.api_key = self.config.get("api_key", os.getenv("HF_API_KEY"))
        self.cache_dir = self.config.get("cache_dir", "./data/huggingface")
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def connect(self) -> bool:
        """Connect to HuggingFace.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Import here to avoid dependency issues if not using HuggingFace
            from huggingface_hub import HfApi
            
            # Initialize the API
            self.api = HfApi(token=self.api_key)
            
            # Test connection by listing a dataset
            _ = self.api.list_datasets(limit=1)
            
            self.connected = True
            logger.info("Connected to HuggingFace successfully")
            return True
        except ImportError:
            logger.error("huggingface_hub package not installed. Install with: pip install huggingface_hub")
            return False
        except Exception as e:
            logger.error(f"Error connecting to HuggingFace: {e}")
            return False
    
    def search(self, query: str, limit: int = 10) -> Dict:
        """Search for datasets on HuggingFace.
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            
        Returns:
            Search results
        """
        if not self.is_connected():
            logger.warning("Not connected to HuggingFace, attempting to connect")
            if not self.connect():
                return {"error": "Not connected to HuggingFace", "success": False}
        
        try:
            # Import here to avoid dependency issues if not using HuggingFace
            from huggingface_hub import HfApi
            
            # Search for datasets
            datasets = self.api.list_datasets(search=query, limit=limit)
            
            results = []
            for dataset in datasets:
                results.append({
                    "id": dataset.id,
                    "author": dataset.author,
                    "tags": dataset.tags,
                    "downloads": dataset.downloads,
                    "likes": dataset.likes,
                    "url": f"https://huggingface.co/datasets/{dataset.id}"
                })
            
            return {"data": results, "success": True}
        except Exception as e:
            error_msg = f"Error searching HuggingFace datasets: {e}"
            logger.error(error_msg)
            return {"error": error_msg, "success": False}
    
    def download(self, dataset_id: str, path: str = None) -> Dict:
        """Download a dataset from HuggingFace.
        
        Args:
            dataset_id: ID of the dataset to download
            path: Path to save the dataset to
            
        Returns:
            Download status
        """
        if not self.is_connected():
            logger.warning("Not connected to HuggingFace, attempting to connect")
            if not self.connect():
                return {"error": "Not connected to HuggingFace", "success": False}
        
        try:
            # Import here to avoid dependency issues if not using HuggingFace
            from datasets import load_dataset
            
            # Set download path
            download_path = path or os.path.join(self.cache_dir, dataset_id.replace("/", "_"))
            
            # Load and download the dataset
            dataset = load_dataset(dataset_id, cache_dir=download_path)
            
            # Get basic information about the dataset
            dataset_info = {
                "id": dataset_id,
                "splits": list(dataset.keys()),
                "num_rows": {split: dataset[split].num_rows for split in dataset},
                "features": str(next(iter(dataset.values())).features),
                "path": download_path
            }
            
            return {"data": dataset_info, "success": True}
        except Exception as e:
            error_msg = f"Error downloading HuggingFace dataset: {e}"
            logger.error(error_msg)
            return {"error": error_msg, "success": False}
    
    def get_dataset_info(self, dataset_id: str) -> Dict:
        """Get information about a HuggingFace dataset.
        
        Args:
            dataset_id: ID of the dataset
            
        Returns:
            Dataset information
        """
        if not self.is_connected():
            logger.warning("Not connected to HuggingFace, attempting to connect")
            if not self.connect():
                return {"error": "Not connected to HuggingFace", "success": False}
        
        try:
            # Import here to avoid dependency issues if not using HuggingFace
            from huggingface_hub import hf_hub_download
            from datasets import get_dataset_config_info
            
            # Get dataset info
            try:
                info = get_dataset_config_info(dataset_id)
                
                result = {
                    "id": dataset_id,
                    "description": info.description,
                    "citation": info.citation,
                    "homepage": info.homepage,
                    "license": info.license,
                    "features": str(info.features),
                    "size_in_bytes": info.size_in_bytes,
                    "version": info.version
                }
            except Exception:
                # Fallback to metadata from the hub
                repo_info = self.api.dataset_info(dataset_id)
                
                result = {
                    "id": dataset_id,
                    "author": repo_info.author,
                    "tags": repo_info.tags,
                    "downloads": repo_info.downloads,
                    "likes": repo_info.likes
                }
            
            return {"data": result, "success": True}
        except Exception as e:
            error_msg = f"Error getting HuggingFace dataset info: {e}"
            logger.error(error_msg)
            return {"error": error_msg, "success": False}

class DataGovConnector(BaseDatasetConnector):
    """Connector for Data.gov datasets."""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.api_endpoint = "https://catalog.data.gov/api/3/action/package_search"
        self.resource_endpoint = "https://catalog.data.gov/api/3/action/resource_show"
        self.api_key = self.config.get("api_key", os.getenv("DATA_GOV_API_KEY"))
        self.cache_dir = self.config.get("cache_dir", "./data/datagov")
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def connect(self) -> bool:
        """Connect to Data.gov.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Test connection with a simple request
            params = {
                "q": "test",
                "rows": 1
            }
            
            headers = {}
            if self.api_key:
                headers["X-API-KEY"] = self.api_key
            
            response = requests.get(self.api_endpoint, params=params, headers=headers)
            
            if response.status_code == 200:
                self.connected = True
                logger.info("Connected to Data.gov successfully")
                return True
            else:
                logger.error(f"Failed to connect to Data.gov: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error connecting to Data.gov: {e}")
            return False
    
    def search(self, query: str, limit: int = 10) -> Dict:
        """Search for datasets on Data.gov.
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            
        Returns:
            Search results
        """
        if not self.is_connected():
            logger.warning("Not connected to Data.gov, attempting to connect")
            if not self.connect():
                return {"error": "Not connected to Data.gov", "success": False}
        
        try:
            # Set up search parameters
            params = {
                "q": query,
                "rows": limit
            }
            
            headers = {}
            if self.api_key:
                headers["X-API-KEY"] = self.api_key
            
            # Execute search
            response = requests.get(self.api_endpoint, params=params, headers=headers)
            
            if response.status_code != 200:
                return {"error": f"Search failed: {response.status_code} - {response.text}", "success": False}
            
            results_data = response.json()
            
            # Format results
            results = []
            for item in results_data.get("result", {}).get("results", []):
                resources = []
                for resource in item.get("resources", []):
                    resources.append({
                        "id": resource.get("id"),
                        "name": resource.get("name"),
                        "format": resource.get("format"),
                        "size": resource.get("size"),
                        "url": resource.get("url")
                    })
                
                results.append({
                    "id": item.get("id"),
                    "title": item.get("title"),
                    "name": item.get("name"),
                    "notes": item.get("notes"),
                    "organization": item.get("organization", {}).get("title"),
                    "resources": resources,
                    "tags": [tag.get("name") for tag in item.get("tags", [])],
                    "url": f"https://catalog.data.gov/dataset/{item.get('name')}"
                })
            
            return {"data": results, "success": True}
        except Exception as e:
            error_msg = f"Error searching Data.gov: {e}"
            logger.error(error_msg)
            return {"error": error_msg, "success": False}
    
    def download(self, dataset_id: str, path: str = None) -> Dict:
        """Download a dataset from Data.gov.
        
        Args:
            dataset_id: Resource ID to download
            path: Path to save the dataset to
            
        Returns:
            Download status
        """
        if not self.is_connected():
            logger.warning("Not connected to Data.gov, attempting to connect")
            if not self.connect():
                return {"error": "Not connected to Data.gov", "success": False}
        
        try:
            # Get resource information
            params = {
                "id": dataset_id
            }
            
            headers = {}
            if self.api_key:
                headers["X-API-KEY"] = self.api_key
            
            response = requests.get(self.resource_endpoint, params=params, headers=headers)
            
            if response.status_code != 200:
                return {"error": f"Failed to get resource information: {response.status_code}", "success": False}
            
            resource_info = response.json().get("result", {})
            resource_url = resource_info.get("url")
            
            if not resource_url:
                return {"error": "Resource URL not found", "success": False}
            
            # Create download directory
            download_path = path or os.path.join(self.cache_dir, dataset_id)
            os.makedirs(download_path, exist_ok=True)
            
            # Determine filename from URL or resource name
            resource_name = resource_info.get("name", "").replace(" ", "_").lower()
            resource_format = resource_info.get("format", "").lower()
            
            if resource_name and resource_format:
                filename = f"{resource_name}.{resource_format}"
            else:
                # Extract filename from URL
                url_path = resource_url.split("?")[0]
                filename = url_path.split("/")[-1] or "dataset"
            
            file_path = os.path.join(download_path, filename)
            
            # Download the resource
            resource_response = requests.get(resource_url, stream=True)
            
            if resource_response.status_code != 200:
                return {"error": f"Failed to download resource: {resource_response.status_code}", "success": False}
            
            with open(file_path, 'wb') as f:
                for chunk in resource_response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return {
                "success": True, 
                "path": file_path, 
                "filename": filename,
                "resource_id": dataset_id,
                "resource_name": resource_info.get("name"),
                "format": resource_info.get("format")
            }
        except Exception as e:
            error_msg = f"Error downloading Data.gov resource: {e}"
            logger.error(error_msg)
            return {"error": error_msg, "success": False}

class ZenodoConnector(BaseDatasetConnector):
    """Connector for Zenodo datasets."""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.api_endpoint = "https://zenodo.org/api"
        self.api_key = self.config.get("api_key", os.getenv("ZENODO_API_KEY"))
        self.cache_dir = self.config.get("cache_dir", "./data/zenodo")
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def connect(self) -> bool:
        """Connect to Zenodo.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Test connection with a simple request
            params = {
                "q": "test",
                "size": 1
            }
            
            headers = {}
            if self.api_key:
                params["access_token"] = self.api_key
            
            response = requests.get(f"{self.api_endpoint}/records", params=params, headers=headers)
            
            if response.status_code == 200:
                self.connected = True
                logger.info("Connected to Zenodo successfully")
                return True
            else:
                logger.error(f"Failed to connect to Zenodo: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error connecting to Zenodo: {e}")
            return False
    
    def search(self, query: str, limit: int = 10) -> Dict:
        """Search for datasets on Zenodo.
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            
        Returns:
            Search results
        """
        if not self.is_connected():
            logger.warning("Not connected to Zenodo, attempting to connect")
            if not self.connect():
                return {"error": "Not connected to Zenodo", "success": False}
        
        try:
            # Set up search parameters
            params = {
                "q": query,
                "size": limit,
                "type": "dataset"  # Filter to datasets only
            }
            
            if self.api_key:
                params["access_token"] = self.api_key
            
            # Execute search
            response = requests.get(f"{self.api_endpoint}/records", params=params)
            
            if response.status_code != 200:
                return {"error": f"Search failed: {response.status_code} - {response.text}", "success": False}
            
            results_data = response.json()
            
            # Format results
            results = []
            for item in results_data.get("hits", {}).get("hits", []):
                files = []
                for file_info in item.get("files", []):
                    files.append({
                        "key": file_info.get("key"),
                        "size": file_info.get("size"),
                        "type": file_info.get("type"),
                        "url": file_info.get("links", {}).get("self")
                    })
                
                metadata = item.get("metadata", {})
                results.append({
                    "id": item.get("id"),
                    "title": metadata.get("title"),
                    "description": metadata.get("description"),
                    "creators": [creator.get("name") for creator in metadata.get("creators", [])],
                    "doi": metadata.get("doi"),
                    "keywords": metadata.get("keywords", []),
                    "license": metadata.get("license", {}).get("id"),
                    "files": files,
                    "url": item.get("links", {}).get("html")
                })
            
            return {"data": results, "success": True}
        except Exception as e:
            error_msg = f"Error searching Zenodo: {e}"
            logger.error(error_msg)
            return {"error": error_msg, "success": False}
    
    def download(self, dataset_id: str, path: str = None) -> Dict:
        """Download a dataset from Zenodo.
        
        Args:
            dataset_id: Record ID to download (or file URL if it contains a slash)
            path: Path to save the dataset to
            
        Returns:
            Download status
        """
        if not self.is_connected():
            logger.warning("Not connected to Zenodo, attempting to connect")
            if not self.connect():
                return {"error": "Not connected to Zenodo", "success": False}
        
        try:
            # Create download directory
            download_path = path or os.path.join(self.cache_dir, dataset_id)
            os.makedirs(download_path, exist_ok=True)
            
            # Check if dataset_id is a record ID or file URL
            if "/" in dataset_id and dataset_id.startswith("http"):
                # Direct file URL
                file_url = dataset_id
                filename = file_url.split("/")[-1]
                
                file_response = requests.get(file_url, stream=True)
                
                if file_response.status_code != 200:
                    return {"error": f"Failed to download file: {file_response.status_code}", "success": False}
                
                file_path = os.path.join(download_path, filename)
                with open(file_path, 'wb') as f:
                    for chunk in file_response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                return {
                    "success": True, 
                    "path": file_path, 
                    "filename": filename,
                    "file_url": file_url
                }
            else:
                # Record ID - get record details first
                params = {}
                if self.api_key:
                    params["access_token"] = self.api_key
                
                response = requests.get(f"{self.api_endpoint}/records/{dataset_id}", params=params)
                
                if response.status_code != 200:
                    return {"error": f"Failed to get record information: {response.status_code}", "success": False}
                
                record_data = response.json()
                files = record_data.get("files", [])
                
                if not files:
                    return {"error": "No files found in the record", "success": False}
                
                # Download all files
                downloaded_files = []
                for file_info in files:
                    file_url = file_info.get("links", {}).get("self")
                    if not file_url:
                        continue
                    
                    # Add access token if available
                    if self.api_key and "?" not in file_url:
                        file_url += f"?access_token={self.api_key}"
                    elif self.api_key:
                        file_url += f"&access_token={self.api_key}"
                    
                    filename = file_info.get("key")
                    file_path = os.path.join(download_path, filename)
                    
                    # Download file
                    file_response = requests.get(file_url, stream=True)
                    
                    if file_response.status_code == 200:
                        with open(file_path, 'wb') as f:
                            for chunk in file_response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        
                        downloaded_files.append(filename)
                    else:
                        logger.warning(f"Failed to download file {filename}: {file_response.status_code}")
                
                return {
                    "success": True, 
                    "path": download_path, 
                    "files": downloaded_files,
                    "record_id": dataset_id,
                    "title": record_data.get("metadata", {}).get("title"),
                    "total_files": len(files),
                    "downloaded_files": len(downloaded_files)
                }
        except Exception as e:
            error_msg = f"Error downloading Zenodo dataset: {e}"
            logger.error(error_msg)
            return {"error": error_msg, "success": False}

class ArXivDatasetConnector(BaseDatasetConnector):
    """Connector for arXiv dataset papers."""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.api_endpoint = "http://export.arxiv.org/api/query"
        self.cache_dir = self.config.get("cache_dir", "./data/arxiv")
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def connect(self) -> bool:
        """Connect to arXiv API.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Test connection with a simple request
            params = {
                "search_query": "cat:cs.AI",
                "max_results": 1
            }
            
            response = requests.get(self.api_endpoint, params=params)
            
            if response.status_code == 200:
                self.connected = True
                logger.info("Connected to arXiv API successfully")
                return True
            else:
                logger.error(f"Failed to connect to arXiv API: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error connecting to arXiv API: {e}")
            return False
    
    def search(self, query: str, limit: int = 10) -> Dict:
        """Search for dataset papers on arXiv.
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            
        Returns:
            Search results
        """
        if not self.is_connected():
            logger.warning("Not connected to arXiv API, attempting to connect")
            if not self.connect():
                return {"error": "Not connected to arXiv API", "success": False}
        
        try:
            # Add dataset search terms to the query
            search_query = f"all:{query} AND (all:dataset OR all:corpus OR all:benchmark)"
            
            # Set up search parameters
            params = {
                "search_query": search_query,
                "max_results": limit,
                "sortBy": "relevance",
                "sortOrder": "descending"
            }
            
            # Execute search
            response = requests.get(self.api_endpoint, params=params)
            
            if response.status_code != 200:
                return {"error": f"Search failed: {response.status_code} - {response.text}", "success": False}
            
            # Parse XML response
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.text)
            
            # Define namespaces used in arXiv API response
            ns = {
                "atom": "http://www.w3.org/2005/Atom",
                "arxiv": "http://arxiv.org/schemas/atom"
            }
            
            # Extract results
            results = []
            for entry in root.findall(".//atom:entry", ns):
                # Extract arXiv ID from the ID URL
                id_url = entry.find("atom:id", ns).text
                arxiv_id = id_url.split("/")[-1]
                
                # Get links
                links = {}
                for link in entry.findall("atom:link", ns):
                    link_type = link.get("title") or link.get("rel")
                    if link_type:
                        links[link_type] = link.get("href")
                
                # Get categories/tags
                categories = []
                for category in entry.findall("atom:category", ns):
                    categories.append(category.get("term"))
                
                result = {
                    "id": arxiv_id,
                    "title": entry.find("atom:title", ns).text.strip(),
                    "summary": entry.find("atom:summary", ns).text.strip(),
                    "published": entry.find("atom:published", ns).text,
                    "updated": entry.find("atom:updated", ns).text,
                    "authors": [author.find("atom:name", ns).text for author in entry.findall("atom:author", ns)],
                    "categories": categories,
                    "links": links,
                    "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                    "abstract_url": f"https://arxiv.org/abs/{arxiv_id}"
                }
                
                results.append(result)
            
            return {"data": results, "success": True}
        except Exception as e:
            error_msg = f"Error searching arXiv: {e}"
            logger.error(error_msg)
            return {"error": error_msg, "success": False}
    
    def download(self, dataset_id: str, path: str = None) -> Dict:
        """Download a paper from arXiv.
        
        Args:
            dataset_id: arXiv ID (e.g., 2101.12345)
            path: Path to save the paper to
            
        Returns:
            Download status
        """
        if not self.is_connected():
            logger.warning("Not connected to arXiv API, attempting to connect")
            if not self.connect():
                return {"error": "Not connected to arXiv API", "success": False}
        
        try:
            # Create download directory
            download_path = path or os.path.join(self.cache_dir, dataset_id)
            os.makedirs(download_path, exist_ok=True)
            
            # Download PDF
            pdf_url = f"https://arxiv.org/pdf/{dataset_id}.pdf"
            pdf_path = os.path.join(download_path, f"{dataset_id}.pdf")
            
            response = requests.get(pdf_url, stream=True)
            
            if response.status_code != 200:
                return {"error": f"Failed to download PDF: {response.status_code}", "success": False}
            
            with open(pdf_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Also get metadata
            params = {
                "id_list": dataset_id,
                "max_results": 1
            }
            
            meta_response = requests.get(self.api_endpoint, params=params)
            
            if meta_response.status_code == 200:
                # Save metadata as JSON
                import xml.etree.ElementTree as ET
                root = ET.fromstring(meta_response.text)
                
                ns = {
                    "atom": "http://www.w3.org/2005/Atom",
                    "arxiv": "http://arxiv.org/schemas/atom"
                }
                
                entry = root.find(".//atom:entry", ns)
                if entry:
                    metadata = {
                        "id": dataset_id,
                        "title": entry.find("atom:title", ns).text.strip(),
                        "summary": entry.find("atom:summary", ns).text.strip(),
                        "published": entry.find("atom:published", ns).text,
                        "updated": entry.find("atom:updated", ns).text,
                        "authors": [author.find("atom:name", ns).text for author in entry.findall("atom:author", ns)],
                        "categories": [category.get("term") for category in entry.findall("atom:category", ns)]
                    }
                    
                    metadata_path = os.path.join(download_path, f"{dataset_id}_metadata.json")
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
            
            return {
                "success": True, 
                "path": download_path, 
                "files": [f"{dataset_id}.pdf", f"{dataset_id}_metadata.json"],
                "arxiv_id": dataset_id,
                "pdf_url": pdf_url
            }
        except Exception as e:
            error_msg = f"Error downloading arXiv paper: {e}"
            logger.error(error_msg)
            return {"error": error_msg, "success": False}

# Update the factory function with all connectors
def create_dataset_connector(dataset_type: str, config: Dict = None) -> BaseDatasetConnector:
    """Create a dataset connector based on the specified type.
    
    Args:
        dataset_type: Type of dataset connector to create
        config: Configuration for the connector
        
    Returns:
        Dataset connector instance
    """
    connectors = {
        "kaggle": KaggleConnector,
        "google_dataset_search": GoogleDatasetSearchConnector,
        "uci_ml": UCIMLRepositoryConnector,
        "imagenet": ImageNetConnector,
        "common_crawl": CommonCrawlConnector,
        "huggingface": HuggingFaceDatasetConnector,
        "datagov": DataGovConnector,
        "zenodo": ZenodoConnector,
        "arxiv": ArXivDatasetConnector
    }
    
    connector_class = connectors.get(dataset_type.lower())
    if not connector_class:
        logger.warning(f"Unknown dataset type: {dataset_type}")
        return None
    
    return connector_class(config) 