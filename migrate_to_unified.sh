#!/bin/bash

echo "Lucky Train AI - Migration to Unified System"
echo "============================================"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8 or higher and try again"
    exit 1
fi

# Create backup of important files
echo "Creating backups..."
mkdir -p backup
cp src/main.py backup/main.py.bak 2>/dev/null || true
[ -f src/improved_system_init.py ] && cp src/improved_system_init.py backup/improved_system_init.py.bak
[ -f src/system_improvements.py ] && cp src/system_improvements.py backup/system_improvements.py.bak
echo "Backups created in backup directory"

# Copy new files
echo
echo "Installing new files..."

# Create improvements configuration
echo "Creating improvements configuration..."
mkdir -p config
cat > config/improvements.json << 'EOF'
{
  "security": {
    "enabled": true,
    "secrets": {
      "auto_create": true,
      "refresh_interval": 300
    },
    "jwt": {
      "auto_rotate": true,
      "rotation_days": 30
    }
  },
  "memory_management": {
    "enabled": true,
    "memory_limit_mb": 1024,
    "warning_threshold": 0.8,
    "critical_threshold": 0.95
  },
  "logging": {
    "enabled": true,
    "max_log_age_days": 30,
    "max_bytes": 10485760,
    "backup_count": 10,
    "compress_logs": true,
    "filter_sensitive": true
  },
  "database": {
    "enabled": true,
    "min_connections": 2,
    "max_connections": 10,
    "connection_lifetime": 3600,
    "idle_timeout": 600
  },
  "thread_safety": {
    "enabled": true
  }
}
EOF

# Install dependencies
echo
echo "Installing dependencies..."
pip3 install psutil cryptography

# Create shell scripts for launching the unified system
echo
echo "Creating start scripts..."

# Create script for running the unified system
cat > run_unified.sh << 'EOF'
#!/bin/bash
python3 src/main.py.unified "$@"
EOF
chmod +x run_unified.sh

# Create script for running all components
cat > run_all_components.sh << 'EOF'
#!/bin/bash
python3 src/main.py.unified --all "$@"
EOF
chmod +x run_all_components.sh

# Create script for console mode
cat > run_console.sh << 'EOF'
#!/bin/bash
python3 src/main.py.unified --console "$@"
EOF
chmod +x run_console.sh

# Create script for web interface
cat > run_web.sh << 'EOF'
#!/bin/bash
python3 src/main.py.unified --web "$@"
EOF
chmod +x run_web.sh

# Create script for Telegram bot
cat > run_telegram.sh << 'EOF'
#!/bin/bash
python3 src/main.py.unified --bot "$@"
EOF
chmod +x run_telegram.sh

echo
echo "Migration completed successfully!"
echo
echo "You can now run the unified system using:"
echo "  ./run_unified.sh        - Run with default settings"
echo "  ./run_all_components.sh - Run all components"
echo "  ./run_console.sh        - Run in console mode"
echo "  ./run_web.sh            - Run web interface"
echo "  ./run_telegram.sh       - Run Telegram bot"
echo
echo "For more information, see UNIFIED_SYSTEM_README.md"
echo 