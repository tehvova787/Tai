"""
Check Dependencies for Lucky Train AI

This script checks if all required dependencies are installed
and provides guidance on installing missing dependencies.
"""

import importlib
import subprocess
import sys
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Required dependencies for PDF processing
PDF_DEPENDENCIES = [
    "pymupdf",        # PyMuPDF
    "pypdf",          # PyPDF
    "pdfminer.six",   # PDFMiner
    "pdf2image",      # PDF2Image
    "pytesseract",    # PyTesseract
]

# Required dependencies for vector database
VECTOR_DB_DEPENDENCIES = [
    "openai",           # OpenAI for embeddings
    "numpy",            # NumPy for vector operations
    "sentence-transformers",  # For embeddings
    "qdrant-client",    # Qdrant vector database
]

# Other essential dependencies
ESSENTIAL_DEPENDENCIES = [
    "requests",        # HTTP requests
    "python-dotenv",   # Environment variables
    "tqdm",            # Progress bars
    "tenacity",        # Retry logic
]

def check_dependency(module_name):
    """Check if a dependency is installed.
    
    Args:
        module_name: Name of the module to check.
        
    Returns:
        True if installed, False otherwise.
    """
    try:
        # Some modules have different import names
        if module_name == "pdfminer.six":
            importlib.import_module("pdfminer")
        elif module_name == "pymupdf":
            # Try both import options
            try:
                importlib.import_module("fitz")
            except ImportError:
                importlib.import_module("pymupdf")
        else:
            importlib.import_module(module_name)
        return True
    except ImportError:
        return False

def install_dependency(module_name):
    """Attempt to install a dependency.
    
    Args:
        module_name: Name of the module to install.
        
    Returns:
        True if installation successful, False otherwise.
    """
    print(f"Attempting to install {module_name}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", module_name])
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing {module_name}: {e}")
        return False

def check_and_install_dependencies(dependencies, auto_install=False):
    """Check and optionally install dependencies.
    
    Args:
        dependencies: List of dependencies to check.
        auto_install: Whether to automatically install missing dependencies.
        
    Returns:
        Tuple of (all_installed, missing_dependencies)
    """
    missing_dependencies = []
    
    for dependency in dependencies:
        if check_dependency(dependency):
            print(f"✓ {dependency} is installed")
        else:
            print(f"✗ {dependency} is NOT installed")
            missing_dependencies.append(dependency)
    
    # If auto_install is True, try to install missing dependencies
    if auto_install and missing_dependencies:
        print("\nAttempting to install missing dependencies...")
        still_missing = []
        
        for dependency in missing_dependencies:
            if install_dependency(dependency):
                print(f"✓ Successfully installed {dependency}")
            else:
                still_missing.append(dependency)
                print(f"✗ Failed to install {dependency}")
        
        missing_dependencies = still_missing
    
    all_installed = len(missing_dependencies) == 0
    return all_installed, missing_dependencies

def main():
    """Main function to check dependencies."""
    print("Checking Lucky Train AI dependencies...\n")
    
    # Determine if automatic installation should be attempted
    auto_install = "--install" in sys.argv
    
    # Check essential dependencies
    print("Checking essential dependencies:")
    essential_ok, missing_essential = check_and_install_dependencies(ESSENTIAL_DEPENDENCIES, auto_install)
    print()
    
    # Check PDF dependencies
    print("Checking PDF processing dependencies:")
    pdf_ok, missing_pdf = check_and_install_dependencies(PDF_DEPENDENCIES, auto_install)
    print()
    
    # Check vector database dependencies
    print("Checking vector database dependencies:")
    vector_ok, missing_vector = check_and_install_dependencies(VECTOR_DB_DEPENDENCIES, auto_install)
    print()
    
    # Summary
    print("\n--- Dependency Check Summary ---")
    
    if essential_ok and pdf_ok and vector_ok:
        print("✓ All dependencies are installed!")
        print("You can proceed with processing PDF files:\n")
        print("  python src/init_pdf_knowledge_base.py")
        return 0
    else:
        print("✗ Some dependencies are missing.")
        
        all_missing = missing_essential + missing_pdf + missing_vector
        
        print("\nTo install missing dependencies, run:")
        print(f"  pip install {' '.join(all_missing)}")
        print("\nOr run this script with the --install flag:")
        print("  python src/check_dependencies.py --install")
        
        if missing_pdf and not missing_essential and not missing_vector:
            print("\nPDF processing will not work without the required dependencies.")
        
        if missing_vector and not missing_essential:
            print("\nVector database functionality will not work without the required dependencies.")
            print("You can still extract text from PDFs but cannot create a searchable knowledge base.")
        
        return 1

if __name__ == "__main__":
    sys.exit(main()) 