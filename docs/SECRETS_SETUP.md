# Настройка секретов и переменных окружения

## Обзор

В этом документе описано, как правильно настроить секретные данные и переменные окружения для проекта LuckyTrainAI.

## Переменные окружения

1. Создайте файл `.env` в корневой директории проекта на основе `.env.example`:

```bash
cp .env.example .env
```

2. Заполните следующие переменные в файле `.env`:

```env
# OpenAI
OPENAI_API_KEY=your_openai_api_key_here

# Flask
FLASK_SECRET_KEY=your_flask_secret_key_here

# Безопасность
SECURITY_SECRET_KEY=your_security_secret_key_here
JWT_SECRET_KEY=your_jwt_secret_key_here
PASSWORD_PEPPER=your_password_pepper_here

# API ключи
WEB_API_KEY=your_web_api_key_here
TON_API_KEY=your_ton_api_key_here
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here

# Базы данных
QDRANT_API_KEY=your_qdrant_api_key
PINECONE_API_KEY=your_pinecone_api_key
WEAVIATE_API_KEY=your_weaviate_api_key
```

## Конфигурационные файлы

1. Создайте файл `config/config.json` на основе `config/config.example.json`:

```bash
cp config/config.example.json config/config.json
```

2. Заполните необходимые поля в `config/config.json`:

```json
{
    "qdrant_api_key": "your_qdrant_api_key",
    "pinecone_api_key": "your_pinecone_api_key",
    "weaviate_api_key": "your_weaviate_api_key",
    "api_key": "your_api_key"
}
```

## Безопасность

1. Никогда не коммитьте файлы с секретами в репозиторий
2. Используйте `.gitignore` для исключения файлов с секретами
3. Регулярно обновляйте секретные данные
4. Используйте разные секреты для разработки и продакшена

## GitHub Secrets

Для CI/CD используйте GitHub Secrets:

1. Перейдите в настройки репозитория
2. Выберите "Secrets and variables" -> "Actions"
3. Добавьте необходимые секреты:
   - `OPENAI_API_KEY`
   - `FLASK_SECRET_KEY`
   - `SECURITY_SECRET_KEY`
   - и другие

## Проверка безопасности

Перед коммитом проверьте:

1. Нет ли секретов в коде
2. Все ли секреты перемещены в переменные окружения
3. Правильно ли настроен `.gitignore`
4. Нет ли секретов в истории коммитов

## Восстановление после утечки

Если секреты были случайно закоммичены:

1. Немедленно отозвать все скомпрометированные ключи
2. Сгенерировать новые ключи
3. Обновить все места использования
4. Очистить историю коммитов от секретов 