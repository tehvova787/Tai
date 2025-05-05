# Lucky Train AI Assistant

Интеллектуальный AI-ассистент для проекта Lucky Train на блокчейне TON. Этот ассистент предоставляет информацию о проекте Lucky Train, его токене, блокчейне TON и метавселенной.

## Возможности

- **Кроссплатформенное приложение**: Работает на iOS, Android, MacOS, Windows и Linux
- **Telegram бот**: Взаимодействие с пользователями через Telegram
- **Веб-интерфейс**: Встраиваемый чат-интерфейс для веб-сайта проекта
- **Интеграция в метавселенную**: Возможность работы ассистента в виртуальном пространстве
- **Консольный режим**: Локальный интерфейс для тестирования и разработки
- **Интеграция с OpenAI**: Использование современных языковых моделей для генерации ответов
- **Retrieval Augmented Generation (RAG)**: Извлечение релевантной информации из базы знаний для улучшения ответов

## Платформы и установка

### Мобильные приложения (iOS и Android)

Мобильные приложения разработаны с использованием React Native и доступны в официальных магазинах приложений:

- **iOS**: [App Store](https://apps.apple.com/app/lucky-train-ai/id1234567890)
- **Android**: [Google Play](https://play.google.com/store/apps/details?id=io.luckytrain.app)

#### Системные требования:

- **iOS**: iOS 12.0 или выше
- **Android**: Android 6.0 (API уровень 23) или выше

### Десктопные приложения (MacOS, Windows, Linux)

Десктопные приложения разработаны с использованием Electron и доступны для скачивания:

- **MacOS**: [Скачать DMG](https://luckytrain.io/downloads/lucky-train-ai-macos.dmg)
- **Windows**: [Скачать EXE](https://luckytrain.io/downloads/lucky-train-ai-windows.exe)
- **Linux**: [Скачать AppImage](https://luckytrain.io/downloads/lucky-train-ai-linux.AppImage) или [DEB пакет](https://luckytrain.io/downloads/lucky-train-ai-linux.deb)

#### Системные требования:

- **MacOS**: macOS 10.13 (High Sierra) или выше
- **Windows**: Windows 10 или выше
- **Linux**: Ubuntu 18.04 или другие основные дистрибутивы Linux

### Серверная часть (для разработчиков)

#### Требования

- Python 3.8 или выше
- pip (менеджер пакетов Python)
- Node.js 16 или выше (для разработки клиентских приложений)
- Доступ в интернет для установки зависимостей
- API ключ OpenAI (опционально, для использования LLM)

#### Шаги установки

1. Клонируйте репозиторий:

```bash
git clone https://github.com/your-username/lucky-train-ai.git
cd lucky-train-ai
```

1. Создайте и активируйте виртуальное окружение:

```bash
# Linux/macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

1. Установите зависимости:

```bash
pip install -r requirements.txt
```

1. Настройте переменные окружения:

```bash
# Скопируйте пример файла переменных окружения
cp env.example .env

# Отредактируйте файл .env и добавьте свои ключи и настройки
nano .env
```

1. Настройте конфигурацию:

```bash
# Убедитесь, что файл config/config.json существует и содержит правильные настройки
```

## Tailwind CSS Setup

This project uses Tailwind CSS for styling. To build Tailwind CSS:

1. Make sure you have the required dependencies:
   ```bash
   npm install
   ```

2. Run the Tailwind build script:
   ```bash
   npm run build:tailwind
   ```

This will generate the Tailwind CSS files in the appropriate locations:
- `src/web/static/css/tailwind.css` - For the web interface
- `src/web/browser_extension/css/tailwind.css` - For the browser extension

For production usage, we DO NOT use the Tailwind CDN as recommended by the Tailwind team. Instead, we build and include a local version of Tailwind CSS.

## Разработка кроссплатформенных клиентов

### Необходимые инструменты

- Node.js 16 или выше
- npm или Yarn
- Xcode (для разработки под iOS)
- Android Studio (для разработки под Android)

### Запуск в режиме разработки

#### Мобильное приложение (React Native)

```bash
# Установка зависимостей
npm install

# Запуск для iOS
npm run ios

# Запуск для Android
npm run android
```

#### Десктопное приложение (Electron)

```bash
# Установка зависимостей
npm install

# Запуск
npm run start:desktop
```

### Сборка приложений

#### Сборка мобильных приложений

```bash
# Сборка для iOS
cd src/mobile/ios
pod install
cd ../../..
npm run build:ios

# Сборка для Android
npm run build:android
```

#### Сборка десктопных приложений

```bash
# Сборка для всех платформ
npm run make:all

# Сборка для MacOS
npm run make:macos

# Сборка для Windows
npm run make:windows

# Сборка для Linux
npm run make:linux
```

## Использование серверной части

### Консольный режим

Для запуска ассистента в консольном режиме выполните:

```bash
python src/main.py console
```

### Telegram бот

Для запуска Telegram бота выполните:

```bash
python src/main.py telegram
```

Токен Telegram бота должен быть указан в файле `.env`:

```env
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
```

### Веб-интерфейс

Для запуска веб-интерфейса выполните:

```bash
python src/main.py web --host 0.0.0.0 --port 5000
```

После запуска веб-интерфейс будет доступен по адресу `http://localhost:5000`.

### Интеграция в метавселенную

Для запуска интеграции в метавселенную выполните:

```bash
python src/main.py metaverse
```

### Запуск всех сервисов одновременно

Для запуска всех компонентов одновременно выполните:

```bash
python src/main.py all
```

## Структура проекта

```text
lucky-train-ai/
├── src/                  # Исходный код
│   ├── shared/           # Общий код для всех платформ
│   │   ├── api/          # API для взаимодействия с бэкендом
│   │   ├── components/   # UI компоненты
│   │   ├── hooks/        # React hooks
│   │   ├── store/        # Глобальное состояние
│   │   └── utils/        # Утилиты
│   ├── mobile/           # Код для React Native (iOS/Android)
│   │   ├── App.tsx       # Точка входа для мобильного приложения
│   │   ├── android/      # Специфичный код для Android
│   │   └── ios/          # Специфичный код для iOS
│   ├── desktop/          # Код для Electron (MacOS/Windows/Linux)
│   │   ├── main.js       # Основной процесс Electron
│   │   └── preload.js    # Preload скрипт Electron
│   ├── bot/              # Код ассистента и Telegram бота
│   │   ├── assistant.py  # Основной класс ассистента
│   │   └── telegram_bot.py # Telegram интеграция
│   ├── web/              # HTML шаблоны для веб-интерфейса
│   ├── main.py           # Главный скрипт запуска серверной части
│   ├── web_interface.py  # Веб-интерфейс
│   └── metaverse_integration.py # Интеграция в метавселенную
├── knowledge_base/       # База знаний в формате JSON
│   ├── lucky_train.json  # Информация о проекте
│   ├── ton_blockchain.json # Информация о блокчейне TON
│   └── metaverse.json    # Информация о метавселенной
├── config/               # Конфигурационные файлы
├── docs/                 # Документация
├── .env                  # Переменные окружения (создается из env.example)
└── requirements.txt      # Зависимости серверной части
```

## Настройка

Основная конфигурация находится в файле `config/config.json`. Вы можете настроить следующие параметры:

- `language`: Язык ассистента (по умолчанию "ru")
- `max_tokens`: Максимальное количество токенов для ответов
- `temperature`: Температура генерации ответов (влияет на творческость)
- `supported_platforms`: Поддерживаемые платформы
- `llm_settings`: Настройки языковой модели
- `response_templates`: Шаблоны ответов
- `telegram_settings`: Настройки Telegram бота
- `web_interface_settings`: Настройки веб-интерфейса
- `metaverse_settings`: Настройки интеграции с метавселенной

## Интеграция с OpenAI

Ассистент может использовать языковые модели OpenAI для генерации ответов. Для включения этой функции:

1. Укажите ваш API ключ OpenAI в файле `.env`:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

2. В файле `config/config.json` настройте параметры LLM:

```json
"llm_settings": {
  "llm_model": "gpt-3.5-turbo",
  "fallback_to_tfidf": true
}
```

Если API ключ не указан или недоступен, система автоматически переключится на алгоритм TF-IDF для генерации ответов.

## OpenAI Integration

The Lucky Train AI project now supports integration with the OpenAI API to provide advanced AI assistance through the web interface.

### Setup OpenAI Integration

1. Install the required dependencies:
   ```
   pip install -r requirements_openai.txt
   ```

2. Set your OpenAI API key (optional, a test key is provided by default):
   ```
   # On Windows
   set OPENAI_API_KEY=your_openai_api_key_here

   # On Linux/Mac
   export OPENAI_API_KEY=your_openai_api_key_here
   ```

3. Set your OpenAI Organization ID (optional):
   ```
   # On Windows
   set OPENAI_ORGANIZATION_ID=your_organization_id_here

   # On Linux/Mac
   export OPENAI_ORGANIZATION_ID=your_organization_id_here
   ```

4. Run the web server with OpenAI integration:
   ```
   python run_openai_server.py
   ```

5. Open your browser and navigate to:
   ```
   http://localhost:5000/
   ```

### Features

- The web interface now uses the OpenAI API to generate responses
- Chat history is preserved between sessions
- Responses are tailored to the Lucky Train project context
- Supports multiple languages (Ukrainian, Russian, English)
- Powered by GPT-4o-mini for fast and accurate responses

## Лицензия

Этот проект распространяется под лицензией [MIT](LICENSE).

## Связь

Для вопросов и предложений обращайтесь по адресу [info@luckytrain.io](mailto:info@luckytrain.io). 