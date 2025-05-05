# Process PDFs script for Lucky Train AI (PowerShell version)
# This script processes PDF files and adds them to the knowledge base

# Check if running as administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "NOTE: Some operations may require administrator privileges." -ForegroundColor Yellow
}

# Activate virtual environment if available
if (Test-Path ".venv") {
    Write-Host "Activating virtual environment..."
    .\.venv\Scripts\Activate.ps1
}

# Ensure required directories exist
if (-not (Test-Path "logs")) {
    New-Item -ItemType Directory -Path "logs" | Out-Null
}
if (-not (Test-Path "data")) {
    New-Item -ItemType Directory -Path "data" | Out-Null
}
if (-not (Test-Path "config")) {
    New-Item -ItemType Directory -Path "config" | Out-Null
}

# Check if config.json exists, create if not
if (-not (Test-Path "config\config.json")) {
    Write-Host "Creating default config file..."
    python src/system_init.py
}

# Check dependencies first
Write-Host "Checking dependencies..." -ForegroundColor Cyan
python src/check_dependencies.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "`nDependencies check failed. Install missing dependencies and try again." -ForegroundColor Red
    Write-Host "You can automatically install dependencies with: python src/check_dependencies.py --install" -ForegroundColor Yellow
    Write-Host "`nPress any key to exit..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

# Process PDF files
Write-Host "Processing PDF files..." -ForegroundColor Cyan
python src/init_pdf_knowledge_base.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "`nPDF processing failed. Check the error messages above." -ForegroundColor Red
    Write-Host "`nPress any key to exit..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

# Report completion
Write-Host "PDF processing complete." -ForegroundColor Green
Write-Host "You can now run the system with: python src/main.py console" -ForegroundColor Cyan

# Keep console open
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") 