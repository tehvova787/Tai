"""
Initialize Lucky Train AI System with PDF Knowledge Base

This script initializes the Lucky Train AI system and loads the PDF files
into the knowledge base for semantic search and retrieval.
"""

import os
import sys
import argparse
import logging
from dotenv import load_dotenv

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)

from system_init import init_system, get_system
from pdf_to_knowledge_base import PDFProcessor

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Use base directory for portable paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# List of PDF files to process
default_pdf_files = [
    os.path.join(base_dir, "blockchain.pdf"),
    os.path.join(base_dir, "golosarii.pdf"),
    os.path.join(base_dir, "LT Brand Essentials.pdf"),
    os.path.join(base_dir, "Lucky Train- Документация.pdf"),
    os.path.join(base_dir, "Prodv.pdf"),
    os.path.join(base_dir, "TON.pdf"),
    os.path.join(base_dir, "White Paper (черновик).pdf"),
    os.path.join(base_dir, "Документация.pdf"),
    os.path.join(base_dir, "inform.pdf")
]

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Initialize Lucky Train AI system with PDF knowledge base.')
    
    parser.add_argument(
        '--config',
        default=os.path.join(parent_dir, 'config', 'config.json'),
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--pdf',
        nargs='+',
        help='Path to one or more PDF files (override default list)'
    )
    
    parser.add_argument(
        '--directory',
        help='Path to directory containing PDF files (override default list)'
    )
    
    parser.add_argument(
        '--skip-pdfs',
        action='store_true',
        help='Skip processing PDF files'
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
    """Initialize the Lucky Train AI system and process PDF files."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Initialize the system
    logger.info("Initializing Lucky Train AI system")
    system = init_system(args.config)
    
    if args.skip_pdfs:
        logger.info("Skipping PDF processing as requested")
        return
    
    # Initialize the PDF processor
    processor = PDFProcessor(args.config)
    
    # Определить полный путь к inform.pdf
    inform_pdf_path = "C:\\Users\\User\\Desktop\\LuckyTrainAI\\inform.pdf"
    
    # Determine which PDF files to process
    if args.pdf:
        pdf_files = args.pdf
        # Добавляем inform.pdf, если его нет в списке
        if inform_pdf_path not in pdf_files:
            pdf_files.append(inform_pdf_path)
            logger.info(f"Added external inform.pdf file: {inform_pdf_path}")
        logger.info(f"Using {len(pdf_files)} PDF files from command line")
    elif args.directory:
        # Get all PDF files in the directory
        pdf_files = [
            os.path.join(args.directory, f) 
            for f in os.listdir(args.directory) 
            if f.lower().endswith('.pdf')
        ]
        # Добавляем inform.pdf, если его нет в директории
        if inform_pdf_path not in pdf_files:
            pdf_files.append(inform_pdf_path)
            logger.info(f"Added external inform.pdf file: {inform_pdf_path}")
        logger.info(f"Found {len(pdf_files)} PDF files in directory {args.directory}")
    else:
        pdf_files = default_pdf_files
        # Проверяем, есть ли inform.pdf в списке по умолчанию
        if inform_pdf_path not in pdf_files:
            pdf_files.append(inform_pdf_path)
            logger.info(f"Added external inform.pdf file: {inform_pdf_path}")
        logger.info(f"Using {len(pdf_files)} default PDF files")
    
    # Verify files exist
    existing_files = []
    for pdf_path in pdf_files:
        if os.path.exists(pdf_path):
            existing_files.append(pdf_path)
        else:
            logger.error(f"File not found: {pdf_path}")
    
    if not existing_files:
        logger.error("No PDF files found. Please check the file paths.")
        return
    
    # Process the files
    logger.info(f"Processing {len(existing_files)} PDF files...")
    result = processor.process_specific_files(existing_files, args.chunk_size, args.overlap)
    
    # Print results
    if result["success"]:
        logger.info(f"Successfully processed {result['processed_count']} of {result['total_files']} PDF files")
    else:
        logger.error(f"Failed to process PDF files. {result['failed_count']} failures out of {result['total_files']} files")
    
    if result["failed_files"]:
        logger.error("Failed files:")
        for file in result["failed_files"]:
            logger.error(f"  - {file}")
    
    logger.info("Lucky Train AI system initialization with PDF knowledge base complete")

if __name__ == "__main__":
    main() 