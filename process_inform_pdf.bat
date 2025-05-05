@echo off
SETLOCAL

echo ===================================
echo Lucky Train AI - Process Inform PDF
echo ===================================

REM Проверка наличия Python
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Python не найден. Пожалуйста, установите Python.
    goto :end
)

REM Активация виртуального окружения, если оно существует
if exist .venv\Scripts\activate.bat (
    echo Активация виртуального окружения...
    call .venv\Scripts\activate.bat
) else (
    echo Виртуальное окружение не найдено, используем системный Python...
)

REM Проверяем наличие файла inform.pdf
if not exist "inform.pdf" (
    echo ВНИМАНИЕ: файл inform.pdf не найден в текущей директории.
    echo Проверяем наличие файла по абсолютному пути...
)

REM Проверка абсолютного пути
if not exist "C:\Users\User\Desktop\LuckyTrainAI\inform.pdf" (
    echo ОШИБКА: файл inform.pdf не найден по пути C:\Users\User\Desktop\LuckyTrainAI\inform.pdf
    goto :end
) else (
    echo Найден файл inform.pdf
)

REM Запуск скрипта обработки PDF
echo.
echo Запуск обработки файла inform.pdf...
python src/process_inform_pdf.py
if %ERRORLEVEL% neq 0 (
    echo ОШИБКА: скрипт обработки PDF завершился с ошибкой.
) else (
    echo Обработка завершена успешно.
)

:end
echo.
echo Нажмите любую клавишу для выхода...
pause >nul
ENDLOCAL 