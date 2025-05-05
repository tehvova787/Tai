"""
PDF to Knowledge Base Converter for Lucky Train AI

This script processes PDF files and loads their content into the knowledge base
for semantic search and retrieval by the Lucky Train AI assistant.
"""

import os
import sys
import json
import logging
from typing import Dict, List, Optional, Union, Any
import argparse
import re
import time
from tqdm import tqdm
from dotenv import load_dotenv

# Try importing fitz from PyMuPDF with error handling
try:
    import fitz  # PyMuPDF
except ImportError:
    try:
        # Alternative import method
        import pymupdf
        fitz = pymupdf
    except ImportError:
        logging.error("Neither 'fitz' nor 'pymupdf' could be imported. Please install PyMuPDF: pip install pymupdf")
        fitz = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)

# Import vector database handler
try:
    from vector_db import VectorDBHandler
except ImportError:
    logger.error("Could not import VectorDBHandler. Make sure the vector_db.py file exists.")
    VectorDBHandler = None

# Load environment variables
load_dotenv()

class PDFProcessor:
    """Process PDF files and extract text for knowledge base integration."""
    
    def __init__(self, config_path: str = "../config/config.json"):
        """Initialize the PDF processor.
        
        Args:
            config_path: Path to the configuration file.
        """
        # Check if PDF libraries are available
        if fitz is None:
            logger.error("PDF processing libraries not available. Please install required dependencies.")
            raise ImportError("Required PDF processing library (PyMuPDF) is not installed")
            
        self.config_path = config_path
        
        # Initialize vector database handler with error handling
        try:
            if VectorDBHandler is None:
                raise ImportError("VectorDBHandler module not available")
                
            self.vector_db = VectorDBHandler(config_path)
            self.vector_db_available = True
            logger.info("PDF Processor initialized with vector database")
        except Exception as e:
            logger.error(f"Failed to initialize vector database: {e}")
            self.vector_db = None
            self.vector_db_available = False
            logger.warning("PDF Processor initialized WITHOUT vector database - only text extraction will be available")
    
    def process_pdf(self, pdf_path: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
        """Process a PDF file and extract text chunks.
        
        Args:
            pdf_path: Path to the PDF file.
            chunk_size: Maximum number of characters per chunk.
            overlap: Number of characters to overlap between chunks.
            
        Returns:
            List of document chunks with text and metadata.
        """
        logger.info(f"Processing PDF file: {pdf_path}")
        
        try:
            # Open the PDF file
            pdf_document = fitz.open(pdf_path)
            filename = os.path.basename(pdf_path)
            
            # Extract document information
            metadata = {
                "title": pdf_document.metadata.get("title", filename),
                "author": pdf_document.metadata.get("author", "Unknown"),
                "subject": pdf_document.metadata.get("subject", ""),
                "keywords": pdf_document.metadata.get("keywords", ""),
                "creator": pdf_document.metadata.get("creator", ""),
                "producer": pdf_document.metadata.get("producer", ""),
                "file_path": pdf_path,
                "filename": filename,
                "page_count": len(pdf_document),
                "created_at": int(time.time())
            }
            
            # Extract all text from the PDF
            all_text = ""
            for page_num, page in enumerate(pdf_document):
                text = page.get_text()
                all_text += f"Page {page_num + 1}:\n{text}\n\n"
            
            # Clean the text
            all_text = self._clean_text(all_text)
            
            # Split the text into chunks with overlap
            chunks = self._split_text_with_overlap(all_text, chunk_size, overlap)
            
            # Create document chunks
            documents = []
            for i, chunk_text in enumerate(chunks):
                # Create a unique ID for the chunk
                chunk_id = f"{filename.replace('.', '_')}_{i}"
                
                # Add chunk metadata
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "id": chunk_id,
                    "chunk_index": i,
                    "chunk_count": len(chunks),
                    "chunk_text_start": chunk_text[:100] + "...",  # For debugging
                })
                
                # Add the document chunk
                documents.append({
                    "text": chunk_text,
                    "metadata": chunk_metadata,
                    "source": pdf_path
                })
            
            logger.info(f"Extracted {len(documents)} chunks from {pdf_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing PDF file {pdf_path}: {e}")
            return []
    
    def _clean_text(self, text: str) -> str:
        """Clean the extracted text.
        
        Args:
            text: The text to clean.
            
        Returns:
            The cleaned text.
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Replace Unicode characters
        text = text.replace('\\u', ' ')
        # Remove other special characters
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}\"\'\`]', '', text)
        # Remove multiple periods
        text = re.sub(r'\.{2,}', '...', text)
        # Fix spacing around punctuation
        text = re.sub(r'\s+([\.,:;!?)])', r'\1', text)
        text = re.sub(r'([([{])\s+', r'\1', text)
        
        return text.strip()
    
    def _split_text_with_overlap(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split the text into chunks with overlap.
        
        Args:
            text: The text to split.
            chunk_size: Maximum number of characters per chunk.
            overlap: Number of characters to overlap between chunks.
            
        Returns:
            List of text chunks.
        """
        # Split by newlines first to avoid breaking in the middle of paragraphs
        paragraphs = text.split('\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed the chunk size, 
            # start a new chunk (but only if we already have some content)
            if current_chunk and len(current_chunk) + len(paragraph) > chunk_size:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap from previous chunk
                if len(current_chunk) > overlap:
                    words = current_chunk.split()
                    overlap_word_count = len(' '.join(words[-100:]).strip())
                    current_chunk = current_chunk[-overlap_word_count:] + "\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add the last chunk if there's anything left
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def add_pdf_to_knowledge_base(self, pdf_path: str, chunk_size: int = 1000, overlap: int = 200) -> bool:
        """Process a PDF file and add it to the knowledge base.
        
        Args:
            pdf_path: Path to the PDF file.
            chunk_size: Maximum number of characters per chunk.
            overlap: Number of characters to overlap between chunks.
            
        Returns:
            True if successful, False otherwise.
        """
        # Check if vector database is available
        if not self.vector_db_available:
            logger.error("Vector database not available. Cannot add PDF to knowledge base.")
            return False
            
        # Process the PDF file
        documents = self.process_pdf(pdf_path, chunk_size, overlap)
        
        if not documents:
            logger.error(f"No content extracted from {pdf_path}")
            return False
        
        # Add documents to vector database
        try:
            success = self.vector_db.add_documents(documents)
            
            if success:
                logger.info(f"Successfully added {len(documents)} chunks from {pdf_path} to knowledge base")
            else:
                logger.error(f"Failed to add documents from {pdf_path} to knowledge base")
            
            return success
        except Exception as e:
            logger.error(f"Error adding documents to vector database: {e}")
            return False
    
    def process_directory(self, directory: str, chunk_size: int = 1000, overlap: int = 200) -> Dict:
        """Process all PDF files in a directory.
        
        Args:
            directory: Path to directory containing PDF files.
            chunk_size: Maximum number of characters per chunk.
            overlap: Number of characters to overlap between chunks.
            
        Returns:
            Dictionary with processing statistics.
        """
        logger.info(f"Processing PDF files in directory: {directory}")
        
        pdf_files = [f for f in os.listdir(directory) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {directory}")
            return {"success": False, "error": "No PDF files found"}
        
        processed_count = 0
        failed_count = 0
        failed_files = []
        
        for pdf_file in tqdm(pdf_files, desc="Processing PDF files"):
            pdf_path = os.path.join(directory, pdf_file)
            
            try:
                success = self.add_pdf_to_knowledge_base(pdf_path, chunk_size, overlap)
                
                if success:
                    processed_count += 1
                else:
                    failed_count += 1
                    failed_files.append(pdf_file)
                    
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {e}")
                failed_count += 1
                failed_files.append(pdf_file)
        
        result = {
            "success": processed_count > 0,
            "processed_count": processed_count,
            "failed_count": failed_count,
            "total_files": len(pdf_files),
            "failed_files": failed_files
        }
        
        logger.info(f"Processed {processed_count} of {len(pdf_files)} PDF files")
        
        return result
    
    def process_specific_files(self, file_paths: List[str], chunk_size: int = 1000, overlap: int = 200) -> Dict:
        """Process specific PDF files.
        
        Args:
            file_paths: List of paths to PDF files.
            chunk_size: Maximum number of characters per chunk.
            overlap: Number of characters to overlap between chunks.
            
        Returns:
            Dictionary with processing statistics.
        """
        logger.info(f"Processing specific PDF files: {len(file_paths)} files")
        
        processed_count = 0
        failed_count = 0
        failed_files = []
        
        for pdf_path in tqdm(file_paths, desc="Processing PDF files"):
            # Проверяем, существует ли файл по указанному пути
            if not os.path.exists(pdf_path):
                # Проверка абсолютного пути для Windows
                if os.name == 'nt' and pdf_path.startswith('C:'):
                    # Используем абсолютный путь как есть
                    abs_path = pdf_path
                    if os.path.exists(abs_path):
                        logger.info(f"Using absolute path: {abs_path}")
                    else:
                        logger.error(f"File not found at absolute path: {abs_path}")
                        failed_count += 1
                        failed_files.append(pdf_path)
                        continue
                else:
                    logger.error(f"File not found: {pdf_path}")
                    failed_count += 1
                    failed_files.append(pdf_path)
                    continue
            else:
                abs_path = pdf_path
                
            if not abs_path.lower().endswith('.pdf'):
                logger.error(f"Not a PDF file: {abs_path}")
                failed_count += 1
                failed_files.append(abs_path)
                continue
            
            try:
                success = self.add_pdf_to_knowledge_base(abs_path, chunk_size, overlap)
                
                if success:
                    processed_count += 1
                else:
                    failed_count += 1
                    failed_files.append(abs_path)
                    
            except Exception as e:
                logger.error(f"Error processing {abs_path}: {e}")
                failed_count += 1
                failed_files.append(abs_path)
        
        result = {
            "success": processed_count > 0,
            "processed_count": processed_count,
            "failed_count": failed_count,
            "total_files": len(file_paths),
            "failed_files": failed_files
        }
        
        logger.info(f"Processed {processed_count} of {len(file_paths)} PDF files")
        
        return result

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Process PDF files and add them to the knowledge base.')
    
    # Add arguments
    parser.add_argument(
        '--pdf',
        nargs='+',
        help='Path to one or more PDF files'
    )
    
    parser.add_argument(
        '--directory',
        help='Path to directory containing PDF files'
    )
    
    parser.add_argument(
        '--config',
        default='../config/config.json',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=1000,
        help='Maximum number of characters per chunk'
    )
    
    parser.add_argument(
        '--overlap',
        type=int,
        default=200,
        help='Number of characters to overlap between chunks'
    )
    
    return parser.parse_args()

def main():
    """Main function to run the PDF processor."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Initialize the PDF processor
    processor = PDFProcessor(args.config)
    
    # Process PDF files
    if args.pdf:
        result = processor.process_specific_files(args.pdf, args.chunk_size, args.overlap)
    elif args.directory:
        result = processor.process_directory(args.directory, args.chunk_size, args.overlap)
    else:
        logger.error("No PDF files or directory specified")
        print("Error: Please specify either --pdf or --directory")
        return
    
    # Print results
    if result["success"]:
        print(f"Successfully processed {result['processed_count']} of {result['total_files']} PDF files")
    else:
        print(f"Failed to process PDF files. {result['failed_count']} failures out of {result['total_files']} files")
    
    if result["failed_files"]:
        print("Failed files:")
        for file in result["failed_files"]:
            print(f"  - {file}")

if __name__ == "__main__":
    main() 