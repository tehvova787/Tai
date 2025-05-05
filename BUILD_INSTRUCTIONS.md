# Lucky Train AI - Build Instructions

This document provides instructions on how to build installation files for all supported platforms:

- iOS (`.ipa`)
- Android (`.apk`)
- macOS (`.dmg`)
- Windows (`.exe` installer)
- Linux (`.deb` and `.rpm` packages)

## Prerequisites

### For all platforms

- Node.js 16+ and npm

### For mobile platforms

- React Native development environment setup
  - Follow the [React Native Environment Setup Guide](https://reactnative.dev/docs/environment-setup)

### For iOS

- macOS computer
- Xcode 13+
- iOS Developer account
- Valid signing certificates and provisioning profiles

### For Android

- Android Studio
- JDK 11+
- Android SDK
- Gradle

### For desktop platforms

- For Windows builds: Windows 10+ or WSL
- For macOS builds: macOS 10.15+
- For Linux builds: Ubuntu 20.04+ or other supported distro

## Build Configuration

1. **iOS Signing**:
   - Open `src/mobile/ios/ExportOptions.plist`
   - Replace `YOUR_TEAM_ID` with your actual Apple Developer Team ID
   - Ensure your signing certificates and provisioning profiles are set up in Xcode

2. **Android Signing**:
   - Create a keystore file in `src/mobile/android/app/my-release-key.keystore`
   - Update `src/mobile/android/gradle.properties` with your keystore information

## Building Installation Files

### Option 1: Automated build for all platforms

Run the appropriate script for your operating system:

**On Windows:**

```bash
build_all.bat
```

**On macOS/Linux:**

```bash
chmod +x build_all.sh
./build_all.sh
```

Or use npm:

```bash
npm run build:all
```

### Option 2: Build individual platforms

#### iOS (macOS only)

```bash
npm run ios
cd src/mobile/ios
xcodebuild -workspace LuckyTrainAI.xcworkspace -scheme LuckyTrainAI -sdk iphoneos -configuration Release archive -archivePath ./build/LuckyTrainAI.xcarchive
xcodebuild -exportArchive -archivePath ./build/LuckyTrainAI.xcarchive -exportOptionsPlist ExportOptions.plist -exportPath ./build
```

#### Android

```bash
npm run android -- --variant=release
```

#### macOS (macOS only)

```bash
npm run make:macos
```

#### Windows

```bash
npm run make:windows
```

#### Linux

```bash
npm run make:linux
```

## Build Output

All installation files will be generated in the `builds/` directory in the project root:

- `LuckyTrainAI.ipa` - iOS app
- `LuckyTrainAI.apk` - Android app
- `LuckyTrainAI.dmg` - macOS installer
- `LuckyTrainAI-Setup.exe` - Windows installer
- `LuckyTrainAI.deb` - Linux Debian package
- `LuckyTrainAI.rpm` - Linux RPM package

## Troubleshooting

### Common Issues

1. **iOS Build Failing**
   - Make sure you have proper signing certificates set up
   - Verify your Team ID in ExportOptions.plist
   - Check Xcode for any build errors

2. **Android Build Failing**
   - Make sure ANDROID_HOME is set to your Android SDK location
   - Check gradle build logs for specific errors
   - Verify keystore configuration

3. **Desktop Build Failing**
   - Check electron-forge logs for specific errors
   - Make sure you have correct permissions to write to output directories
