@echo off
SETLOCAL

:: Build script for Lucky Train AI - generates installation files for all platforms

:: Check if Node.js is installed
WHERE node >nul 2>nul
IF %ERRORLEVEL% NEQ 0 (
    echo Error: Node.js is not installed. Please install Node.js before running this script.
    EXIT /B 1
)

:: Set up environment for Android building if needed
IF "%ANDROID_HOME%"=="" (
    IF EXIST "%USERPROFILE%\AppData\Local\Android\Sdk" (
        SET ANDROID_HOME=%USERPROFILE%\AppData\Local\Android\Sdk
    ) ELSE (
        echo Warning: ANDROID_HOME environment variable not set. Android builds may fail.
    )
)

:: Print info about the build environment
echo 🚂 Lucky Train AI - Build Environment
echo ==================================
echo Node.js version: 
node --version
echo NPM version: 
npm --version
echo OS: Windows
IF NOT "%ANDROID_HOME%"=="" (
    echo Android SDK: %ANDROID_HOME%
)

:: Run the build script
echo.
echo Starting build process...
node build_all_platforms.js

:: Inform about build results
IF %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ Build completed successfully!
    echo Check the builds/ directory for installation files.
) ELSE (
    echo.
    echo ❌ Build failed!
)

ENDLOCAL 