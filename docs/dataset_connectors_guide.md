# Dataset Connectors Guide

This guide provides information on how to use the dataset connectors in the Lucky Train AI project to access various public datasets and knowledge bases.

## Available Dataset Connectors

The following dataset connectors are available:

1. **Kaggle** (`kaggle`) - Access datasets from [Kaggle](https://www.kaggle.com/datasets)
2. **Google Dataset Search** (`google_dataset_search`) - Search datasets using [Google Dataset Search](https://datasetsearch.research.google.com/)
3. **UCI Machine Learning Repository** (`uci_ml`) - Access datasets from [UCI ML Repository](https://archive.ics.uci.edu/ml/)
4. **ImageNet** (`imagenet`) - Access image datasets from [ImageNet](https://image-net.org/)
5. **Common Crawl** (`common_crawl`) - Access web crawl data from [Common Crawl](https://commoncrawl.org/)
6. **HuggingFace Datasets** (`huggingface`) - Access datasets from [HuggingFace](https://huggingface.co/datasets)
7. **Data.gov** (`datagov`) - Access open government datasets from [Data.gov](https://data.gov/)
8. **Zenodo** (`zenodo`) - Access research datasets from [Zenodo](https://zenodo.org/)
9. **arXiv Dataset** (`arxiv`) - Access dataset papers from [arXiv](https://arxiv.org/)

## Installation

Ensure you have the required dependencies installed:

```bash
pip install -r requirements.txt
```

## Credentials

Most dataset connectors require API keys or credentials. Set these as environment variables or create a `.env` file in your project root with the following:

```
# Kaggle
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_key

# HuggingFace
HF_API_KEY=your_huggingface_api_key

# Google (for Google Dataset Search)
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CSE_ID=your_custom_search_engine_id

# ImageNet
IMAGENET_USERNAME=your_imagenet_username
IMAGENET_ACCESS_KEY=your_imagenet_access_key

# Data.gov
DATA_GOV_API_KEY=your_data_gov_api_key

# Zenodo
ZENODO_API_KEY=your_zenodo_api_key
```

## Basic Usage

Here's a basic example of how to use a dataset connector:

```python
from dataset_connectors import create_dataset_connector

# Create a connector
kaggle_connector = create_dataset_connector("kaggle")

# Connect to the service
if kaggle_connector.connect():
    # Search for datasets
    results = kaggle_connector.search("climate change", limit=5)
    
    if results["success"]:
        # Print results
        for dataset in results["data"]:
            print(f"Dataset: {dataset['title']}")
            print(f"URL: {dataset['url']}")
            print("---")
        
        # Download a dataset
        if results["data"]:
            dataset_id = results["data"][0]["id"]
            download = kaggle_connector.download(dataset_id)
            
            if download["success"]:
                print(f"Downloaded to: {download['path']}")
```

## Using from Configuration

You can also initialize dataset connectors from your configuration file:

```python
import json
from dataset_connectors import create_dataset_connector

# Load config
with open("config/config.json", "r") as f:
    config = json.load(f)

# Get dataset settings
dataset_configs = config.get("data_source_settings", {})

# Initialize connectors
connectors = {}
for dataset_type, settings in dataset_configs.items():
    if settings.get("enabled", False):
        connector = create_dataset_connector(dataset_type, settings)
        if connector and connector.connect():
            connectors[dataset_type] = connector
```

## Common Methods

All dataset connectors provide these common methods:

### connect()

Connect to the dataset service.

```python
connector.connect()
```

### search(query, limit)

Search for datasets matching the query.

```python
results = connector.search("machine learning", limit=10)
```

### download(dataset_id, path)

Download a specific dataset.

```python
result = connector.download("dataset_id", path="./data/downloaded_dataset")
```

## Examples

For more detailed examples, see `src/dataset_usage_example.py`.

## Adding a New Connector

To add a new dataset connector:

1. Create a new class in `src/dataset_connectors.py` that inherits from `BaseDatasetConnector`
2. Implement the required methods: `connect()`, `search()`, and `download()`
3. Add your connector to the `create_dataset_connector()` factory function

## Troubleshooting

If you encounter issues:

1. **Connection errors**: Check your API keys and internet connection
2. **Missing dependencies**: Ensure all required packages are installed
3. **Rate limiting**: Some services have API rate limits, add delay between requests if needed
4. **Download failures**: Check if you have write permissions to the download directory

## Further Resources

- Kaggle API: https://github.com/Kaggle/kaggle-api
- HuggingFace Hub: https://huggingface.co/docs/huggingface_hub/
- UCI ML Repository: https://archive.ics.uci.edu/ml/datasets.php
- arXiv API: https://arxiv.org/help/api/ 