# Настройка мониторинга Lucky Train AI

В этом руководстве описывается настройка системы мониторинга для Lucky Train AI с использованием Prometheus и Grafana.

## Архитектура мониторинга

Система мониторинга состоит из следующих компонентов:

1. **Prometheus** - система сбора и хранения метрик
2. **Grafana** - система визуализации и анализа метрик
3. **Node Exporter** - экспортер метрик с хоста
4. **Lucky Train Metrics** - встроенный экспортер метрик приложения

![Архитектура мониторинга](../assets/monitoring_architecture.png)

## Требования

- Docker и Docker Compose
- Доступ к серверам Lucky Train AI
- Минимум 2 ГБ RAM для Prometheus и Grafana
- 10+ ГБ дискового пространства для хранения метрик

## Быстрая установка с Docker Compose

Создайте файл `docker-compose.yml` с следующим содержимым:

```yaml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    restart: unless-stopped
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'
    ports:
      - "9090:9090"
    networks:
      - monitoring
    
  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    restart: unless-stopped
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=lucky-train-admin
      - GF_USERS_ALLOW_SIGN_UP=false
      - GF_SERVER_DOMAIN=localhost
      - GF_SMTP_ENABLED=false
    ports:
      - "3000:3000"
    networks:
      - monitoring
    depends_on:
      - prometheus
  
  node-exporter:
    image: prom/node-exporter:latest
    container_name: node-exporter
    restart: unless-stopped
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - '--path.procfs=/host/proc'
      - '--path.sysfs=/host/sys'
      - '--collector.filesystem.ignored-mount-points=^/(sys|proc|dev|host|etc)($$|/)'
    ports:
      - "9100:9100"
    networks:
      - monitoring

networks:
  monitoring:
    driver: bridge

volumes:
  prometheus_data:
  grafana_data:
```

Создайте файл конфигурации Prometheus `prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  scrape_timeout: 10s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'node_exporter'
    static_configs:
      - targets: ['node-exporter:9100']
    
  - job_name: 'lucky_train_ai'
    static_configs:
      - targets: ['host.docker.internal:9091']  # Порт экспортера метрик Lucky Train AI
    metrics_path: '/metrics'
    scheme: 'http'
```

Запустите контейнеры:

```bash
docker-compose up -d
```

## Настройка Lucky Train AI для экспорта метрик

1. Обновите конфигурацию в файле `config/config.json`, добавив раздел мониторинга:

```json
{
  "monitoring_settings": {
    "enabled": true,
    "metrics_port": 9091,
    "collection_interval": 15
  }
}
```

2. Убедитесь, что в `requirements.txt` добавлены зависимости:

```
prometheus-client>=0.14.1
psutil>=5.9.0
```

3. Включите мониторинг в основном приложении:

```python
from monitoring_system import get_monitoring_service

def main():
    # Инициализация системы
    system = init_system()
    
    # Запуск мониторинга
    monitoring = get_monitoring_service(system.config.get("monitoring_settings", {}))
    monitoring.start()
    
    # ... остальной код ...
    
    # При завершении
    monitoring.stop()
```

4. Для веб-интерфейса добавьте middleware:

```python
from monitoring_system import get_metrics, MetricsMiddleware

app = Flask(__name__)

# Добавление middleware для мониторинга
metrics_middleware = MetricsMiddleware(get_metrics())
metrics_middleware.flask_register(app)
```

## Настройка дашбордов в Grafana

1. Откройте Grafana в браузере по адресу http://localhost:3000
2. Войдите с учетными данными (по умолчанию admin/lucky-train-admin)
3. Добавьте Prometheus как источник данных:
   - Перейдите в Configuration > Data Sources
   - Нажмите Add data source
   - Выберите тип Prometheus
   - В поле URL введите http://prometheus:9090
   - Нажмите Save & Test

### Импорт готовых дашбордов

В Grafana перейдите в меню "+" и выберите "Import".

#### Системный дашборд Node Exporter

Импортируйте дашборд с ID 1860 для мониторинга системных ресурсов.

#### Lucky Train AI дашборд

Скачайте файл [lucky_train_dashboard.json](../grafana/dashboards/lucky_train_dashboard.json) и импортируйте его.

## Настройка алертов

### Prometheus Alertmanager

Добавьте в `docker-compose.yml` сервис Alertmanager:

```yaml
alertmanager:
  image: prom/alertmanager:latest
  container_name: alertmanager
  restart: unless-stopped
  volumes:
    - ./alertmanager.yml:/etc/alertmanager/config.yml
  ports:
    - "9093:9093"
  networks:
    - monitoring
```

Создайте файл конфигурации `alertmanager.yml`:

```yaml
global:
  resolve_timeout: 5m
  smtp_smarthost: 'smtp.example.com:587'
  smtp_from: 'alerts@example.com'
  smtp_auth_username: 'username'
  smtp_auth_password: 'password'

route:
  group_by: ['alertname', 'job']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h
  receiver: 'email-alerts'

receivers:
- name: 'email-alerts'
  email_configs:
  - to: 'team@example.com'
```

### Правила алертов в Prometheus

Создайте файл `prometheus_rules.yml`:

```yaml
groups:
- name: lucky_train_alerts
  rules:
  - alert: HighCPUUsage
    expr: lucky_train_cpu_usage_percent > 80
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Высокая загрузка CPU"
      description: "CPU usage превышает 80% в течение 5 минут."
      
  - alert: HighMemoryUsage
    expr: lucky_train_memory_usage_percent > 85
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Высокое использование памяти"
      description: "Использование памяти превышает 85% в течение 5 минут."
      
  - alert: HighErrorRate
    expr: sum(rate(lucky_train_error_count_total[5m])) / sum(rate(lucky_train_request_count_total[5m])) > 0.05
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "Высокий процент ошибок"
      description: "Более 5% запросов завершаются с ошибкой."
```

Обновите `prometheus.yml`, добавив секцию:

```yaml
rule_files:
  - "prometheus_rules.yml"

alerting:
  alertmanagers:
  - static_configs:
    - targets:
      - alertmanager:9093
```

## Мониторинг микросервисов

Для мониторинга микросервисной архитектуры Lucky Train AI добавьте в `prometheus.yml` динамическое обнаружение сервисов:

```yaml
scrape_configs:
  # ... другие job_name ...
  
  - job_name: 'lucky_train_microservices'
    dns_sd_configs:
      - names:
          - 'tasks.api-gateway'
          - 'tasks.assistant-core'
          - 'tasks.knowledge-base'
          - 'tasks.ai-model'
          - 'tasks.blockchain-integration'
        type: 'A'
        port: 9091
```

## Безопасность

### Настройка аутентификации

Для защиты Prometheus и Grafana настройте базовую аутентификацию, добавив в `docker-compose.yml`:

```yaml
prometheus:
  # ... другие настройки ...
  environment:
    - PROMETHEUS_WEB_ROUTE_PREFIX=/
    - PROMETHEUS_WEB_EXTERNAL_URL=https://prometheus.example.com
    - PROMETHEUS_ADMIN_PASSWORD=secure-password

grafana:
  # ... другие настройки ...
  environment:
    - GF_SECURITY_ADMIN_PASSWORD=secure-password
    - GF_SERVER_ROOT_URL=https://grafana.example.com
```

### TLS/SSL

Рекомендуется использовать HTTPS для защиты метрик. Настройте NGINX как обратный прокси:

```nginx
server {
    listen 443 ssl;
    server_name prometheus.example.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://localhost:9090;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        auth_basic "Restricted";
        auth_basic_user_file /etc/nginx/.htpasswd;
    }
}
```

## Часто задаваемые вопросы

### Как проверить, что экспорт метрик работает?

Откройте в браузере URL `http://your-server-ip:9091/metrics` - вы должны увидеть текстовые метрики в формате Prometheus.

### Как добавить новые метрики?

В файле `src/monitoring_system.py` добавьте новые метрики в класс `MetricsCollector` и обновите методы для их сбора.

### Сколько места потребуется для хранения метрик?

По умолчанию Prometheus хранит данные 15 дней. При интервале сбора 15 секунд и 100 метриках потребуется примерно 2-5 ГБ.

## Дополнительные ресурсы

- [Документация Prometheus](https://prometheus.io/docs/introduction/overview/)
- [Документация Grafana](https://grafana.com/docs/)
- [Prometheus Python Client](https://github.com/prometheus/client_python)
- [Alertmanager Documentation](https://prometheus.io/docs/alerting/latest/alertmanager/) 