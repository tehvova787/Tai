#!/bin/bash

# Build script for Lucky Train AI - generates installation files for all platforms

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "Error: Node.js is not installed. Please install Node.js before running this script."
    exit 1
fi

# Set up environment for Android building if needed
if [ -z "$ANDROID_HOME" ]; then
    # Try to find Android SDK location
    if [ -d "$HOME/Library/Android/sdk" ]; then
        export ANDROID_HOME="$HOME/Library/Android/sdk"
    elif [ -d "$HOME/Android/Sdk" ]; then
        export ANDROID_HOME="$HOME/Android/Sdk"
    else
        echo "Warning: ANDROID_HOME environment variable not set. Android builds may fail."
    fi
fi

# Print info about the build environment
echo "🚂 Lucky Train AI - Build Environment"
echo "=================================="
echo "Node.js version: $(node --version)"
echo "NPM version: $(npm --version)"
echo "OS: $(uname -s)"
if [ -n "$ANDROID_HOME" ]; then
    echo "Android SDK: $ANDROID_HOME"
fi

# Run the build script
echo -e "\nStarting build process..."
node build_all_platforms.js

# Inform about build results
if [ $? -eq 0 ]; then
    echo -e "\n✅ Build completed successfully!"
    echo "Check the builds/ directory for installation files."
else
    echo -e "\n❌ Build failed!"
fi 