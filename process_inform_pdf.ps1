# Lucky Train AI - Process Inform PDF

Write-Host "====================================" -ForegroundColor Cyan
Write-Host "Lucky Train AI - Process Inform PDF" -ForegroundColor Cyan 
Write-Host "====================================" -ForegroundColor Cyan
Write-Host ""

# Проверка наличия Python
try {
    $pythonVersion = python --version
    Write-Host "Используется $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "Python не найден. Пожалуйста, установите Python." -ForegroundColor Red
    Write-Host "Нажмите любую клавишу для выхода..."
    $host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") | Out-Null
    exit
}

# Активация виртуального окружения, если оно существует
if (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Host "Активация виртуального окружения..." -ForegroundColor Yellow
    . .\.venv\Scripts\Activate.ps1
} else {
    Write-Host "Виртуальное окружение не найдено, используем системный Python..." -ForegroundColor Yellow
}

# Проверяем наличие файла inform.pdf
if (-not (Test-Path "inform.pdf")) {
    Write-Host "ВНИМАНИЕ: файл inform.pdf не найден в текущей директории." -ForegroundColor Yellow
    Write-Host "Проверяем наличие файла по абсолютному пути..." -ForegroundColor Yellow
}

# Проверка абсолютного пути
$informPdfPath = "C:\Users\User\Desktop\LuckyTrainAI\inform.pdf"
if (-not (Test-Path $informPdfPath)) {
    Write-Host "ОШИБКА: файл inform.pdf не найден по пути $informPdfPath" -ForegroundColor Red
    Write-Host "Нажмите любую клавишу для выхода..."
    $host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") | Out-Null
    exit
} else {
    Write-Host "Найден файл inform.pdf" -ForegroundColor Green
}

# Запуск скрипта обработки PDF
Write-Host ""
Write-Host "Запуск обработки файла inform.pdf..." -ForegroundColor Cyan
try {
    python src/process_inform_pdf.py
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Обработка завершена успешно." -ForegroundColor Green
    } else {
        Write-Host "ОШИБКА: скрипт обработки PDF завершился с ошибкой (код $LASTEXITCODE)." -ForegroundColor Red
    }
} catch {
    Write-Host "ОШИБКА при выполнении скрипта: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "Нажмите любую клавишу для выхода..."
$host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") | Out-Null 