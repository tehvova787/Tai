const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');

// Configuration
const BUILD_DIR = path.join(__dirname, 'builds');
const ANDROID_RELEASE_DIR = path.join(__dirname, 'src/mobile/android/app/build/outputs/apk/release');
const IOS_RELEASE_DIR = path.join(__dirname, 'src/mobile/ios/build/Build/Products/Release-iphoneos');

// Create builds directory if it doesn't exist
if (!fs.existsSync(BUILD_DIR)) {
  fs.mkdirSync(BUILD_DIR, { recursive: true });
}

// Helper function to run commands and handle errors
function runCommand(command, errorMessage) {
  try {
    console.log(`Running: ${command}`);
    execSync(command, { stdio: 'inherit' });
  } catch (error) {
    console.error(errorMessage || `Error executing command: ${command}`);
    console.error(error);
    process.exit(1);
  }
}

// Copy output files to builds directory
function copyBuildOutput(sourcePath, destinationName) {
  if (fs.existsSync(sourcePath)) {
    const destPath = path.join(BUILD_DIR, destinationName);
    fs.copyFileSync(sourcePath, destPath);
    console.log(`Copied to: ${destPath}`);
  } else {
    console.warn(`Warning: Source file not found at ${sourcePath}`);
  }
}

// Build iOS app (requires macOS)
function buildIOS() {
  if (os.platform() !== 'darwin') {
    console.log('Skipping iOS build: can only be built on macOS');
    return;
  }

  console.log('\n📱 Building iOS app...');
  // Build React Native iOS app in release mode
  runCommand('cd src/mobile && npx react-native run-ios --configuration Release --no-packager', 'Failed to build iOS app');
  
  // Create IPA archive (requires manually configured provisioning profiles)
  try {
    runCommand('cd src/mobile/ios && xcodebuild -workspace LuckyTrainAI.xcworkspace -scheme LuckyTrainAI -sdk iphoneos -configuration Release archive -archivePath ./build/LuckyTrainAI.xcarchive', 'Failed to archive iOS app');
    runCommand('cd src/mobile/ios && xcodebuild -exportArchive -archivePath ./build/LuckyTrainAI.xcarchive -exportOptionsPlist ExportOptions.plist -exportPath ./build', 'Failed to export iOS app');
    
    // Copy IPA to builds directory
    copyBuildOutput(path.join(__dirname, 'src/mobile/ios/build/LuckyTrainAI.ipa'), 'LuckyTrainAI.ipa');
  } catch (error) {
    console.warn('Warning: IPA generation requires proper code signing setup. See iOS build logs for details.');
  }
}

// Build Android app
function buildAndroid() {
  console.log('\n🤖 Building Android app...');
  // Build React Native Android app in release mode
  runCommand('cd src/mobile && npx react-native run-android --variant=release --no-packager', 'Failed to build Android app');
  
  // Copy APK to builds directory
  copyBuildOutput(path.join(ANDROID_RELEASE_DIR, 'app-release.apk'), 'LuckyTrainAI.apk');
}

// Build desktop apps using electron-forge
function buildDesktop() {
  console.log('\n💻 Building desktop apps...');
  
  // Build for macOS
  if (os.platform() === 'darwin') {
    console.log('\nBuilding for macOS...');
    runCommand('npm run make:macos', 'Failed to build macOS app');
    // Copy DMG to builds directory
    const macDmgPath = path.join(__dirname, 'out/make/LuckyTrainAI.dmg');
    copyBuildOutput(macDmgPath, 'LuckyTrainAI.dmg');
  } else {
    console.log('Skipping macOS build: can only be built on macOS');
  }
  
  // Build for Windows
  console.log('\nBuilding for Windows...');
  runCommand('npm run make:windows', 'Failed to build Windows app');
  // Copy installer to builds directory
  const winInstallerPath = path.join(__dirname, 'out/make/squirrel.windows/x64/LuckyTrainAI-1.0.0 Setup.exe');
  copyBuildOutput(winInstallerPath, 'LuckyTrainAI-Setup.exe');
  
  // Build for Linux
  if (os.platform() !== 'win32') {
    console.log('\nBuilding for Linux...');
    runCommand('npm run make:linux', 'Failed to build Linux app');
    // Copy deb and rpm to builds directory
    const debPath = path.join(__dirname, 'out/make/deb/x64/lucky-train-ai_1.0.0_amd64.deb');
    const rpmPath = path.join(__dirname, 'out/make/rpm/x64/lucky-train-ai-1.0.0-1.x86_64.rpm');
    copyBuildOutput(debPath, 'LuckyTrainAI.deb');
    copyBuildOutput(rpmPath, 'LuckyTrainAI.rpm');
  } else {
    console.log('Skipping Linux build: better results when built on Linux');
  }
}

// Main build process
async function buildAll() {
  console.log('🚂 Lucky Train AI - Building installation files for all platforms');
  console.log('==================================================================');
  
  // Build for all platforms
  buildIOS();
  buildAndroid();
  buildDesktop();
  
  console.log('\n✅ Build process completed!');
  console.log(`All installation files have been placed in: ${BUILD_DIR}`);
  console.log('\nAvailable installation files:');
  
  // List all generated files
  const files = fs.readdirSync(BUILD_DIR);
  files.forEach(file => {
    console.log(`- ${file}`);
  });
}

// Run the build process
buildAll().catch(err => {
  console.error('Build process failed:', err);
  process.exit(1); 