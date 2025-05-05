# Интеграция файла inform.pdf в систему Lucky Train AI

Этот документ описывает, как в системе Lucky Train AI настроена интеграция файла `inform.pdf` по абсолютному пути `C:\Users\User\Desktop\LuckyTrainAI\inform.pdf`.

## Внесенные изменения

1. **Добавлен файл в список PDF-файлов для обработки**:
   - Файл `inform.pdf` добавлен в список PDF-файлов в `src/init_pdf_knowledge_base.py`
   - Файл `inform.pdf` добавлен в список PDF-файлов в `src/process_pdfs.py`

2. **Поддержка абсолютных путей**:
   - Метод `process_specific_files` в `src/pdf_to_knowledge_base.py` модифицирован для поддержки абсолютных путей к файлам
   - Добавлена проверка существования файла по абсолютному пути для Windows

3. **Дополнительные инструменты**:
   - Создан специальный скрипт `src/process_inform_pdf.py` для обработки только файла `inform.pdf`
   - Создан batch-файл `process_inform_pdf.bat` для Windows для запуска обработки
   - Создан PowerShell скрипт `process_inform_pdf.ps1` для запуска обработки

## Как использовать

Существует несколько способов обработать файл `inform.pdf`:

### 1. Использовать стандартный скрипт инициализации

```bash
python src/init_pdf_knowledge_base.py
```

Скрипт автоматически включит файл `inform.pdf` в обработку.

### 2. Использовать специальный скрипт для process_inform_pdf.py

```bash
python src/process_inform_pdf.py
```

Этот скрипт обрабатывает только файл `inform.pdf`.

### 3. Использовать batch-файл (Windows)

```
.\process_inform_pdf.bat
```

### 4. Использовать PowerShell скрипт (Windows)

```powershell
.\process_inform_pdf.ps1
```

## Проверка

После обработки файла `inform.pdf` его содержимое должно быть доступно в базе знаний системы Lucky Train AI. Это можно проверить, задав вопрос системе, который требует информации из этого файла.

## Примечания

- Абсолютный путь к файлу `inform.pdf` жестко закодирован как `C:\Users\User\Desktop\LuckyTrainAI\inform.pdf`
- Если путь к файлу изменится, необходимо обновить все соответствующие файлы
- Система добавляет файл `inform.pdf` к стандартному списку PDF-файлов, так что при обработке всех файлов он также будет обработан 