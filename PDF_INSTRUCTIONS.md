# Lucky Train AI - PDF Knowledge Base Integration

This document provides instructions on how to integrate PDF files into the Lucky Train AI knowledge base for enhanced question answering capabilities.

## Overview

The Lucky Train AI system can now use the content from PDF files as an additional knowledge source. The system processes PDF files, extracts their content, and adds it to a vector database for semantic search and retrieval.

## Required Files

The following PDF files have been integrated:

1. `blockchain.pdf`
2. `golosarii.pdf`
3. `LT Brand Essentials.pdf`
4. `Lucky Train- Документация.pdf`
5. `Prodv.pdf`
6. `TON.pdf`
7. `White Paper (черновик).pdf`
8. `Документация.pdf`

## Prerequisites

Before using the PDF knowledge base functionality, make sure you have the following:

1. Python 3.8 or higher
2. Required Python packages (added to `requirements.txt`):
   - pypdf>=3.17.2
   - pdfminer.six>=20221105
   - pdf2image>=1.16.3
   - pytesseract>=0.3.10
   - fitz>=0.0.1.dev2
   - pymupdf>=1.23.8

3. API keys (add to your `.env` file):
   - OpenAI API key (for embeddings and retrieval)
   - Optional: Vector database API keys (if using cloud-hosted vector databases)

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Create or update your `.env` file with the necessary API keys:

```
OPENAI_API_KEY=your_openai_api_key
# Optional: Vector database API keys
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
```

## Usage

### Processing PDF Files

To process PDF files and add them to the knowledge base, use the provided scripts:

1. To process specific PDF files:

```bash
python src/pdf_to_knowledge_base.py --pdf /path/to/file1.pdf /path/to/file2.pdf
```

2. To process all PDF files in a directory:

```bash
python src/pdf_to_knowledge_base.py --directory /path/to/pdf/directory
```

3. To process the default PDF files provided in the system:

```bash
python src/process_pdfs.py
```

4. To initialize the system with the PDF knowledge base:

```bash
python src/init_pdf_knowledge_base.py
```

### Advanced Options

When processing PDFs, you can customize the following parameters:

- `--chunk-size`: The maximum number of characters per text chunk (default: 1000)
- `--overlap`: The number of characters to overlap between chunks (default: 200)
- `--config`: Path to a custom configuration file

Example:

```bash
python src/pdf_to_knowledge_base.py --pdf /path/to/file.pdf --chunk-size 1500 --overlap 300
```

### Using in the Main Application

Once PDFs are processed, the main Lucky Train AI system will automatically use the knowledge base for answering questions. No additional steps are required.

Start the system as usual:

```bash
python src/main.py console
```

## Configuration

The system uses a vector database to store and retrieve PDF content. By default, it uses a local Qdrant database, but you can configure it to use other vector databases like Pinecone or Weaviate.

Configuration settings are stored in the `config/config.json` file under the `vector_db_settings` section:

```json
"vector_db_settings": {
    "db_type": "qdrant",  // Options: qdrant, pinecone, weaviate
    "qdrant_url": null,   // For cloud Qdrant
    "qdrant_api_key": null,
    "qdrant_path": "./data/qdrant_data",  // For local Qdrant
    "collection_name": "lucky_train_kb",
    "embedding_model": {
        "type": "openai",  // Options: openai, sentence_transformers
        "name": "text-embedding-3-small"
    }
}
```

## Troubleshooting

### PDF Processing Issues

If you encounter issues when processing PDFs:

1. Make sure the PDF files exist and are accessible
2. Check that you have sufficient disk space
3. Verify that your PDF files are not password-protected
4. For large PDFs, consider increasing the available memory

### Vector Database Issues

If you encounter issues with the vector database:

1. Check that your API keys are correctly set in the `.env` file
2. Verify that the vector database service is running and accessible
3. For local Qdrant, ensure the data directory exists and is writable

## Advanced: Adding Custom PDF Processors

You can extend the PDF processing capabilities by creating custom processor classes. See the `PDFProcessor` class in `src/pdf_to_knowledge_base.py` for an example of how to implement a custom processor.

## Resources

For more information, refer to the following resources:

- [Lucky Train AI Documentation](./README.md)
- [PyPDF Documentation](https://pypdf2.readthedocs.io/en/latest/)
- [PyMuPDF Documentation](https://pymupdf.readthedocs.io/en/latest/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [OpenAI Embeddings Documentation](https://platform.openai.com/docs/guides/embeddings) 