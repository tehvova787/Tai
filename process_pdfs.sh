#!/bin/bash

# Process PDFs script for Lucky Train AI
# This script processes PDF files and adds them to the knowledge base

# Activate virtual environment if available
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Ensure required directories exist
mkdir -p logs
mkdir -p data
mkdir -p config

# Check if config.json exists, create if not
if [ ! -f "config/config.json" ]; then
    echo "Creating default config file..."
    python src/system_init.py
fi

# Check dependencies first
echo "Checking dependencies..."
python src/check_dependencies.py
if [ $? -ne 0 ]; then
    echo
    echo "Dependencies check failed. Install missing dependencies and try again."
    echo "You can automatically install dependencies with: python src/check_dependencies.py --install"
    echo
    read -p "Press any key to exit..." -n1 -s
    exit 1
fi

# Process PDF files
echo "Processing PDF files..."
python src/init_pdf_knowledge_base.py
if [ $? -ne 0 ]; then
    echo
    echo "PDF processing failed. Check the error messages above."
    echo
    read -p "Press any key to exit..." -n1 -s
    exit 1
fi

# Report completion
echo "PDF processing complete."
echo "You can now run the system with: python src/main.py console"

# Keep terminal open
read -p "Press any key to exit..." -n1 -s
echo 