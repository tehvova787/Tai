# Переменные окружения для Lucky Train AI

В этом документе описаны все переменные окружения, используемые системой Lucky Train AI. Настройте эти переменные перед запуском системы для обеспечения корректной работы.

## Основные переменные

| Переменная | Описание | Пример | Обязательная |
|------------|----------|--------|--------------|
| `OPENAI_API_KEY` | API ключ OpenAI для доступа к языковым моделям | `sk-abcdefg123456789` | Да |
| `FLASK_SECRET_KEY` | Секретный ключ для Flask-сессий | `random_secure_string` | Да |
| `WEB_API_KEY` | API ключ для веб-интерфейса | `web_api_key_123` | Да |

## Переменные безопасности

| Переменная | Описание | Пример | Обязательная |
|------------|----------|--------|--------------|
| `SECURITY_SECRET_KEY` | Главный секретный ключ для модуля безопасности | `very_secure_random_string` | Да |
| `JWT_SECRET_KEY` | Секретный ключ для подписи JWT-токенов | `jwt_secret_key_string` | Да |
| `PASSWORD_PEPPER` | "Перец" для дополнительного хеширования паролей | `random_pepper_string` | Нет |
| `ENCRYPTION_KEY` | Ключ для шифрования данных (в base64) | `base64_encoded_key` | Нет |

## Административные учетные данные

| Переменная | Описание | Пример | Обязательная |
|------------|----------|--------|--------------|
| `ADMIN_USERNAME` | Имя пользователя для административного доступа | `admin` | Да |
| `ADMIN_PASSWORD` | Пароль для административного доступа | `secure_admin_password` | Да |

## Интеграция с блокчейном

| Переменная | Описание | Пример | Обязательная |
|------------|----------|--------|--------------|
| `TON_API_KEY` | API ключ для доступа к блокчейну TON | `ton_api_key_123` | Да* |
| `TON_ENDPOINT` | Адрес RPC-сервера TON | `https://toncenter.com/api/v2/jsonRPC` | Нет |

## Интеграция с Telegram

| Переменная | Описание | Пример | Обязательная |
|------------|----------|--------|--------------|
| `TELEGRAM_BOT_TOKEN` | Токен Telegram бота от BotFather | `1234567890:ABCDEFGhijklmnop` | Да* |

## Векторная база данных

| Переменная | Описание | Пример | Обязательная |
|------------|----------|--------|--------------|
| `QDRANT_URL` | URL облачного сервиса Qdrant | `https://your-qdrant-instance.cloud` | Нет |
| `QDRANT_API_KEY` | API ключ для Qdrant | `qdrant_api_key_123` | Нет |
| `PINECONE_API_KEY` | API ключ для Pinecone | `pinecone_api_key_123` | Нет |
| `PINECONE_ENVIRONMENT` | Окружение Pinecone | `us-west1-gcp` | Нет |
| `WEAVIATE_URL` | URL облачного сервиса Weaviate | `https://your-weaviate-instance.cloud` | Нет |
| `WEAVIATE_API_KEY` | API ключ для Weaviate | `weaviate_api_key_123` | Нет |

## Мультимодальные возможности

| Переменная | Описание | Пример | Обязательная |
|------------|----------|--------|--------------|
| `ELEVENLABS_API_KEY` | API ключ ElevenLabs для синтеза речи | `elevenlabs_api_key_123` | Нет |

## Примечания

- Переменные, отмеченные как "Да*", обязательны только при использовании соответствующей функциональности (например, `TELEGRAM_BOT_TOKEN` обязателен только если вы запускаете Telegram бота).
- Для локальной разработки создайте файл `.env` в корневой директории проекта и укажите в нем все необходимые переменные.
- В производственной среде настройте переменные окружения через системные средства вашего сервера или контейнера.

## Пример .env файла

```env
OPENAI_API_KEY=sk-abcdefg123456789
FLASK_SECRET_KEY=random_secure_string
WEB_API_KEY=web_api_key_123
SECURITY_SECRET_KEY=very_secure_random_string
JWT_SECRET_KEY=jwt_secret_key_string
ADMIN_USERNAME=admin
ADMIN_PASSWORD=secure_admin_password
TON_API_KEY=ton_api_key_123
TELEGRAM_BOT_TOKEN=1234567890:ABCDEFGhijklmnop
```
