# Lucky Train AI Unified System

This document explains how the Lucky Train AI system has been integrated into a unified architecture that combines all existing components using a microservices design pattern.

## Overview

The unified system integrates all components of the Lucky Train AI ecosystem into a cohesive architecture while maintaining compatibility with existing code. This integration provides several benefits:

- **Centralized Management**: All components are managed through a single interface
- **Improved Resource Utilization**: Shared resources and connection pooling
- **Enhanced Security**: Centralized security management with improved input validation
- **Better Performance**: Optimized memory usage and database connections
- **Improved Reliability**: Central monitoring and error handling

## Architecture

The unified system follows a microservices architecture with the following components:

1. **API Gateway Service**: Central entry point for all clients
2. **Assistant Core Service**: Core dialog processing
3. **Knowledge Base Service**: Vector database and knowledge retrieval
4. **AI Model Service**: Manages AI model interactions
5. **Blockchain Integration Service**: TON blockchain integration
6. **User Management Service**: User profiles and preferences
7. **Metaverse Connector Service**: Metaverse integration

## How to Use

### Running the Unified System

To run the Lucky Train AI with the unified system architecture, use the new main entry point:

```bash
# Basic usage (starts the web interface)
python src/main.py.unified

# Run with the Telegram bot
python src/main.py.unified --bot

# Run all components
python src/main.py.unified --all

# Run in console mode
python src/main.py.unified --console

# Specify which components to run
python src/main.py.unified --components web_interface,telegram_bot

# Specify a different configuration file
python src/main.py.unified --config path/to/config.json
```

### Command Line Options

The unified system supports the following command line options:

- `--config`: Path to configuration file
- `--log-level`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `--components`: Comma-separated list of components to start
- `--console`: Run in interactive console mode
- `--web`: Run web interface
- `--bot`: Run Telegram bot
- `--all`: Run all components
- `--host`: Web interface host (default: 0.0.0.0)
- `--port`: Web interface port (default: 5000)
- `--demo`: Run in demo mode
- `--security-level`: Security level (standard or enhanced)

### Available Components

You can specify any of the following components to start:

- `api_gateway`: API Gateway Service
- `assistant_core`: Assistant Core Service
- `knowledge_base`: Knowledge Base Service
- `ai_model`: AI Model Service
- `blockchain`: Blockchain Integration Service
- `user_management`: User Management Service
- `metaverse`: Metaverse Connector Service
- `web_interface`: Web Interface
- `telegram_bot`: Telegram Bot
- `multimodal`: Multimodal Interface

## Migration Guide

To migrate from the previous system to the unified architecture:

1. Install the new dependencies:

```bash
pip install -r requirements.txt
```

2. Copy the new files to your project:

- `src/unified_system_integrator.py`
- `src/main.py.unified`

3. Run the system using the new main entry point:

```bash
python src/main.py.unified
```

## Implementation Details

The unified system is implemented as a façade that provides a simple interface to the complex subsystem:

- **UnifiedSystem**: Main class that integrates all components
- **Microservices**: Each component is implemented as a microservice
- **Threading**: Each service runs in its own thread
- **API Gateway**: Central entry point for all client requests
- **Improved Security**: SecretManager and InputValidator
- **Memory Management**: MemoryMonitor and optimized caching
- **Logging Enhancements**: Improved logging with rotation
- **Connection Pooling**: Database connection pool

## Configuration

The unified system uses the same configuration files as the existing system, with additional settings for the improvements.

Create a file at `config/improvements.json` with the following content:

```json
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
```

## API Usage

The unified system provides a simple API for handling requests:

```python
from unified_system_integrator import create_unified_system

# Create the unified system
unified_system = create_unified_system("path/to/config.json")

# Start specific services
unified_system.start_services(["web_interface", "telegram_bot"])

# Handle a request
response = unified_system.handle_request({
    "message": "What is Lucky Train?",
    "user_id": "user123",
    "platform": "api"
})

# Shutdown the system
unified_system.shutdown()
```

## Troubleshooting

If you encounter issues when using the unified system:

1. **Check logs**: Look at the logs for error messages
2. **Memory issues**: Adjust memory limits in the configuration
3. **Database connection issues**: Check database settings and connection limits
4. **Service initialization failures**: Ensure all required dependencies are installed

## Technical Support

For questions or assistance implementing the unified system, contact the maintainer team at support@luckytrainai.com. 