"""
Process Inform PDF File Script

This script specifically processes the inform.pdf file and adds it to the Lucky Train AI knowledge base.
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

def main():
    """Process the inform.pdf file and add it to the knowledge base."""
    # Initialize the PDF processor
    processor = PDFProcessor("../config/config.json")
    
    # Абсолютный путь к inform.pdf
    inform_pdf_path = "C:\\Users\\User\\Desktop\\LuckyTrainAI\\inform.pdf"
    
    # Verify file exists
    if not os.path.exists(inform_pdf_path):
        logger.error(f"inform.pdf file not found at: {inform_pdf_path}")
        return
    
    logger.info(f"Processing inform.pdf file from: {inform_pdf_path}")
    
    # Process the file
    result = processor.process_specific_files([inform_pdf_path])
    
    # Print results
    if result["success"]:
        logger.info(f"Successfully processed inform.pdf")
    else:
        logger.error(f"Failed to process inform.pdf")
    
    if result["failed_files"]:
        logger.error("Failed files:")
        for file in result["failed_files"]:
            logger.error(f"  - {file}")

if __name__ == "__main__":
    main() 