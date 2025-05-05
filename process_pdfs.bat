@echo off
REM Process PDFs script for Lucky Train AI (Batch file version)
REM This script processes PDF files and adds them to the knowledge base

echo === Lucky Train AI - PDF Processing ===

REM Activate virtual environment if available
if exist .venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
)

REM Ensure required directories exist
if not exist logs mkdir logs
if not exist data mkdir data
if not exist config mkdir config

REM Check if config.json exists, create if not
if not exist config\config.json (
    echo Creating default config file...
    python src/system_init.py
)

REM Check dependencies first
echo Checking dependencies...
python src/check_dependencies.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Dependencies check failed. Install missing dependencies and try again.
    echo You can automatically install dependencies with: python src/check_dependencies.py --install
    echo.
    pause
    exit /b 1
)

REM Process PDF files
echo Processing PDF files...
python src/init_pdf_knowledge_base.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo PDF processing failed. Check the error messages above.
    echo.
    pause
    exit /b 1
)

REM Report completion
echo.
echo PDF processing complete.
echo You can now run the system with: python src/main.py console

REM Keep console open
echo.
pause 