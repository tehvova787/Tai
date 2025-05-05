"""
Process PDF Files Script

This script processes the specified PDF files and adds them to the Lucky Train AI knowledge base.
"""

import os
import sys
import logging
from pdf_to_knowledge_base import PDFProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Use os.path.join для кроссплатформенной совместимости путей
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# List of PDF files to process
pdf_files = [
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

def main():
    """Process the specified PDF files and add them to the knowledge base."""
    # Initialize the PDF processor
    processor = PDFProcessor("../config/config.json")
    
    # Добавляем абсолютный путь к inform.pdf
    inform_pdf_path = "C:\\Users\\User\\Desktop\\LuckyTrainAI\\inform.pdf"
    if inform_pdf_path not in pdf_files:
        pdf_files.append(inform_pdf_path)
        logger.info(f"Added external inform.pdf file: {inform_pdf_path}")
    
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
    result = processor.process_specific_files(existing_files)
    
    # Print results
    if result["success"]:
        logger.info(f"Successfully processed {result['processed_count']} of {result['total_files']} PDF files")
    else:
        logger.error(f"Failed to process PDF files. {result['failed_count']} failures out of {result['total_files']} files")
    
    if result["failed_files"]:
        logger.error("Failed files:")
        for file in result["failed_files"]:
            logger.error(f"  - {file}")

if __name__ == "__main__":
    main() 