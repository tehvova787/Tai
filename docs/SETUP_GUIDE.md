# Руководство по настройке Lucky Train AI

## Содержание

1. [Установка и требования](#установка-и-требования)
2. [Переменные окружения](#переменные-окружения)
3. [Настройка компонентов](#настройка-компонентов)
   - [Основная система](#основная-система)
   - [Telegram бот](#telegram-бот)
   - [Веб-интерфейс](#веб-интерфейс)
   - [Интеграция с блокчейном](#интеграция-с-блокчейном)
   - [Интеграция с метавселенной](#интеграция-с-метавселенной)
4. [Запуск системы](#запуск-системы)
5. [Мониторинг и обслуживание](#мониторинг-и-обслуживание)

## Установка и требования

### Системные требования

- Python 3.8 или выше
- Node.js 16 или выше (для разработки клиентских приложений)
- Подключение к интернету
- 4 ГБ RAM минимум (8 ГБ рекомендуется)
- 10 ГБ свободного места на диске

### Установка

1. **Клонирование репозитория:**

```bash
git clone https://github.com/your-username/lucky-train-ai.git
cd lucky-train-ai
```

2. **Создание и активация виртуального окружения:**

```bash
# Linux/macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

3. **Установка зависимостей:**

```bash
pip install -r requirements.txt
```

Для разработки клиентских приложений:

```bash
npm install
```

## Переменные окружения

Создайте файл `.env` в корневой директории проекта. Используйте `env.example` как шаблон:

```bash
cp env.example .env
```

### Обязательные переменные окружения

| Переменная | Описание | Пример |
|------------|----------|--------|
| OPENAI_API_KEY | API-ключ OpenAI для интеграции языковой модели | sk-abcdefg123456789 |
| TELEGRAM_BOT_TOKEN | Токен Telegram бота от BotFather | 1234567890:ABCDEFGhijklmnop |
| TON_API_KEY | API-ключ TON для доступа к функциям блокчейна | ton_api_1234567890 |
| FLASK_SECRET_KEY | Секретный ключ для безопасности веб-интерфейса | any-random-string-here |
| WEB_API_KEY | API-ключ для аутентификации веб-запросов | web_api_1234567890 |
| ADMIN_USERNAME | Имя администратора | admin |
| ADMIN_PASSWORD | Пароль администратора | secure_password |

### Опциональные переменные окружения

| Переменная | Описание | Пример |
|------------|----------|--------|
| ELEVENLABS_API_KEY | API-ключ ElevenLabs для расширенных голосовых функций | elevenlabs_api_key |
| QDRANT_URL | URL для облачного сервиса Qdrant | https://your-qdrant-instance.cloud |
| QDRANT_API_KEY | API-ключ для Qdrant | qdrant_api_key_123 |
| PINECONE_API_KEY | API-ключ для Pinecone | pinecone_api_key_123 |
| PINECONE_ENVIRONMENT | Окружение Pinecone | us-west1-gcp |
| WEAVIATE_URL | URL для облачного сервиса Weaviate | https://your-weaviate-instance.cloud |
| WEAVIATE_API_KEY | API-ключ для Weaviate | weaviate_api_key_123 |

## Настройка компонентов

### Основная система

1. **Настройка конфигурации:**

Отредактируйте файл `config/config.json` для настройки основных параметров системы:

```json
{
  "language": "ru",
  "max_tokens": 1000,
  "temperature": 0.7,
  "knowledge_base_path": "./knowledge_base",
  "supported_platforms": ["telegram", "website", "metaverse", "console"],
  "default_ai_model": "ani",
  "current_ai_model": "ani",
  ...
}
```

2. **Инициализация базы знаний:**

Подготовьте JSON-файлы в директории `knowledge_base/`:

- `lucky_train.json` - информация о проекте
- `ton_blockchain.json` - информация о блокчейне TON
- `metaverse.json` - информация о метавселенной

3. **Проверка зависимостей:**

```bash
python src/check_dependencies.py
```

### Telegram бот

1. **Создание Telegram бота:**

- Откройте @BotFather в Telegram
- Создайте нового бота командой `/newbot`
- Следуйте инструкциям для получения токена
- Добавьте токен в переменную `TELEGRAM_BOT_TOKEN` в файле `.env`

2. **Настройка команд бота:**

В файле `config/config.json` отредактируйте раздел `telegram_settings`:

```json
"telegram_settings": {
  "welcome_message": "Привет! Я официальный AI-ассистент проекта Lucky Train на блокчейне TON. Я могу рассказать вам о проекте, токене LTT, метавселенной и многом другом.",
  "commands": {
    "start": "Начать общение с ботом",
    "help": "Показать список доступных команд",
    "about": "Информация о проекте Lucky Train",
    "token": "Информация о токене LTT",
    "metaverse": "Информация о метавселенной Lucky Train"
  }
}
```

3. **Запуск Telegram бота:**

```bash
python src/main.py telegram
```

### Веб-интерфейс

1. **Настройка веб-интерфейса:**

В файле `config/config.json` отредактируйте раздел `web_interface_settings`:

```json
"web_interface_settings": {
  "title": "Lucky Train AI Assistant",
  "theme": "light",
  "welcome_message": "Привет! Я официальный AI-ассистент проекта Lucky Train. Чем я могу вам помочь?"
}
```

2. **Безопасность веб-интерфейса:**

Настройте параметры безопасности в разделе `security_settings`:

```json
"security_settings": {
  "rate_limit": {
    "enabled": true,
    "max_requests_per_minute": 60
  },
  "input_validation": true,
  "api_key_required": true
}
```

3. **Запуск веб-интерфейса:**

```bash
python src/main.py web --host 0.0.0.0 --port 5000
```

### Интеграция с блокчейном

1. **Получение API-ключа TON:**

- Зарегистрируйтесь на сервисе TON API (например, toncenter.com)
- Получите API-ключ
- Добавьте ключ в переменную `TON_API_KEY` в файле `.env`

2. **Настройка параметров блокчейна:**

В файле `config/config.json` отредактируйте раздел `blockchain_settings`:

```json
"blockchain_settings": {
  "network": "mainnet",
  "rpc_endpoint": "https://toncenter.com/api/v2/jsonRPC",
  "explorer_url": "https://tonscan.org"
}
```

3. **Проверка интеграции с блокчейном:**

```bash
python src/main.py blockchain
```

### Интеграция с метавселенной

1. **Настройка параметров метавселенной:**

В файле `config/config.json` отредактируйте раздел `metaverse_settings`:

```json
"metaverse_settings": {
  "avatar_model": "default_assistant",
  "voice_enabled": true,
  "welcome_message": "Приветствую в метавселенной Lucky Train! Я ваш виртуальный ассистент. Чем я могу помочь?",
  "engine": "unity",
  "server_endpoint": "wss://metaverse.luckytrain.io"
}
```

2. **Настройка мультимодальных возможностей:**

В файле `config/config.json` отредактируйте раздел `multimodal_settings`:

```json
"multimodal_settings": {
  "image_generation": {
    "enabled": true,
    "provider": "openai"
  },
  "text_to_speech": {
    "enabled": true,
    "provider": "elevenlabs"
  },
  "speech_to_text": {
    "enabled": true,
    "provider": "whisper"
  }
}
```

3. **Проверка интеграции с метавселенной:**

```bash
python src/main.py metaverse
```

## Запуск системы

### Запуск отдельных компонентов

1. **Консольный режим (для тестирования):**

```bash
python src/main.py console
```

2. **Telegram бот:**

```bash
python src/main.py telegram
```

3. **Веб-интерфейс:**

```bash
python src/main.py web --host 0.0.0.0 --port 5000
```

4. **Демонстрация блокчейна:**

```bash
python src/main.py blockchain
```

5. **Демонстрация мультимодальности:**

```bash
python src/main.py multimodal
```

### Запуск всех компонентов

Для запуска всех компонентов одновременно:

```bash
python src/main.py all
```

### Запуск в фоновом режиме

Для запуска в фоновом режиме с использованием systemd (Linux):

1. Создайте файл сервиса:

```bash
sudo nano /etc/systemd/system/luckytrain-ai.service
```

2. Добавьте следующее содержимое:

```ini
[Unit]
Description=Lucky Train AI Assistant
After=network.target

[Service]
User=username
WorkingDirectory=/path/to/lucky-train-ai
ExecStart=/path/to/lucky-train-ai/venv/bin/python src/main.py all
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

3. Включите и запустите сервис:

```bash
sudo systemctl enable luckytrain-ai.service
sudo systemctl start luckytrain-ai.service
```

## Мониторинг и обслуживание

### Логи

Логи системы хранятся в директории `logs/`:

- `logs/lucky_train_ai.log` - основной лог
- `logs/error.log` - ошибки
- `logs/access.log` - доступ
- `logs/security.log` - безопасность
- `logs/analytics.log` - аналитика

### Резервное копирование

Рекомендуется регулярно создавать резервные копии следующих директорий:

- `knowledge_base/` - база знаний
- `data/` - данные системы
- `config/` - конфигурация

### Обновление системы

Для обновления системы:

1. Остановите все компоненты:

```bash
sudo systemctl stop luckytrain-ai.service  # Для systemd
```

2. Обновите код:

```bash
git pull
```

3. Обновите зависимости:

```bash
pip install -r requirements.txt
```

4. Запустите систему:

```bash
sudo systemctl start luckytrain-ai.service  # Для systemd
```

### Мониторинг производительности

Для мониторинга производительности можно использовать административный интерфейс, доступный по адресу:

`http://your-server:5000/admin`

Для доступа используйте учетные данные администратора, указанные в переменных окружения `ADMIN_USERNAME` и `ADMIN_PASSWORD`. 