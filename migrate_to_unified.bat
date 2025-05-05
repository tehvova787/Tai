@echo off
echo Lucky Train AI - Migration to Unified System
echo ============================================
echo.

:: Check if Python is installed
python --version > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher and try again
    exit /b 1
)

:: Create backup of important files
echo Creating backups...
mkdir backup 2> nul
copy src\main.py backup\main.py.bak
if exist src\improved_system_init.py copy src\improved_system_init.py backup\improved_system_init.py.bak
if exist src\system_improvements.py copy src\system_improvements.py backup\system_improvements.py.bak
echo Backups created in backup directory

:: Copy new files
echo.
echo Installing new files...

:: Create improvements configuration
echo Creating improvements configuration...
if not exist config mkdir config
echo {> config\improvements.json
echo   "security": {>> config\improvements.json
echo     "enabled": true,>> config\improvements.json
echo     "secrets": {>> config\improvements.json
echo       "auto_create": true,>> config\improvements.json
echo       "refresh_interval": 300>> config\improvements.json
echo     },>> config\improvements.json
echo     "jwt": {>> config\improvements.json
echo       "auto_rotate": true,>> config\improvements.json
echo       "rotation_days": 30>> config\improvements.json
echo     }>> config\improvements.json
echo   },>> config\improvements.json
echo   "memory_management": {>> config\improvements.json
echo     "enabled": true,>> config\improvements.json
echo     "memory_limit_mb": 1024,>> config\improvements.json
echo     "warning_threshold": 0.8,>> config\improvements.json
echo     "critical_threshold": 0.95>> config\improvements.json
echo   },>> config\improvements.json
echo   "logging": {>> config\improvements.json
echo     "enabled": true,>> config\improvements.json
echo     "max_log_age_days": 30,>> config\improvements.json
echo     "max_bytes": 10485760,>> config\improvements.json
echo     "backup_count": 10,>> config\improvements.json
echo     "compress_logs": true,>> config\improvements.json
echo     "filter_sensitive": true>> config\improvements.json
echo   },>> config\improvements.json
echo   "database": {>> config\improvements.json
echo     "enabled": true,>> config\improvements.json
echo     "min_connections": 2,>> config\improvements.json
echo     "max_connections": 10,>> config\improvements.json
echo     "connection_lifetime": 3600,>> config\improvements.json
echo     "idle_timeout": 600>> config\improvements.json
echo   },>> config\improvements.json
echo   "thread_safety": {>> config\improvements.json
echo     "enabled": true>> config\improvements.json
echo   }>> config\improvements.json
echo }>> config\improvements.json

:: Install dependencies
echo.
echo Installing dependencies...
pip install psutil cryptography

:: Create shell script for launching the unified system
echo.
echo Creating start scripts...
echo @echo off> run_unified.bat
echo python src\main.py.unified %*>> run_unified.bat

:: Create script for running all components
echo @echo off> run_all_components.bat
echo python src\main.py.unified --all %*>> run_all_components.bat

:: Create script for console mode
echo @echo off> run_console.bat
echo python src\main.py.unified --console %*>> run_console.bat

:: Create script for web interface
echo @echo off> run_web.bat
echo python src\main.py.unified --web %*>> run_web.bat

:: Create script for Telegram bot
echo @echo off> run_telegram.bat
echo python src\main.py.unified --bot %*>> run_telegram.bat

echo.
echo Migration completed successfully!
echo.
echo You can now run the unified system using:
echo   run_unified.bat       - Run with default settings
echo   run_all_components.bat - Run all components
echo   run_console.bat       - Run in console mode
echo   run_web.bat           - Run web interface
echo   run_telegram.bat      - Run Telegram bot
echo.
echo For more information, see UNIFIED_SYSTEM_README.md
echo. 