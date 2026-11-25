# Docker: запуск и обслуживание

Пошаговый первый запуск, переход с Python-сервиса и диагностика — [QUICKSTART.md](QUICKSTART.md). Из корня клонированного репозитория: `docker compose up --build -d --wait --wait-timeout 300`. Требуются Docker с Linux containers и Compose v2. Python 3.14.6 и зависимости устанавливаются внутри образа; модели OCR входят в Git. Windows runtime ZIP, Cambria и `.venv` для Docker не требуются. Первая сборка требует доступ к Docker Hub, Debian и PyPI. Обработка использует локальные модели и CPU.

## Сервисы и данные

| Сервис | Назначение |
|---|---|
| `db` | PostgreSQL 18, внутренний порт 5432 |
| `setup` | Одноразовая инициализация схем и привязки хранилища; exit 0 означает успех |
| `web` | Интерфейс и HTTP API на http://127.0.0.1:8000 |
| `worker` | Последовательная обработка чертежей |
| `corrections` | Текстовые/инженерные правки |
| `tables` | Табличные правки |

Том `documents` смонтирован в `/app/artifacts/profile-service-v1`, оригиналы и результаты — в его подкаталоге `storage`. Этот путь находится под корнем приложения: алгоритмы проверяют принадлежность промежуточных файлов рабочему каталогу. Том `database` смонтирован в `/var/lib/postgresql` (расположение данных PostgreSQL 18). Имена томов имеют префикс Compose-проекта `scanned-drawings-to-cad`. Windows-файлы из `artifacts/profile-service-v1/storage` и локальная БД не используются контейнерами.

Приложение работает от UID 10001. Код, зависимости и модели находятся внутри образа; исходники хоста не монтируются. Поэтому `git pull` применяется после пересборки. В контекст сборки не попадают пользовательские документы, `.venv`, базы, `.update-backups` и release ZIP.

Доступ к Web опубликован только на localhost; пароль БД в Compose предназначен для этой локальной установки. Внешний порт БД не опубликован. Для общего сетевого/публичного сервера нужна отдельная конфигурация доступа.

В `web` включён `CAD_DOCKER_LOCAL_OPERATOR=1`: обращения через опубликованный localhost-порт сохраняют права локального оператора, несмотря на подмену адреса клиента Docker NAT. Это позволяет применять правки и таблицы. Не публикуйте этот режим на сетевом интерфейсе; для сетевого размещения отключите его и настройте предусмотренные приложением ключи доступа. Заголовки forwarded client IP не используются.

Сервисы выделенной Compose-сети также имеют эти права при обращении к `web:8000`; подключайте к ней только доверенные контейнеры. Проверки Origin для изменяющих запросов сохраняются.

## Проверки и логи

```bash
docker compose ps -a
docker compose logs --tail 100 db setup web worker corrections tables
```

Необязательные проверки после запуска; два теста выполняйте по порядку:

```bash
docker compose run --rm --no-deps web check
docker compose exec -T web python /app/docker/smoke_test.py
docker compose exec -T web python /app/docker/smoke_edits.py
```

Тест создаёт синтетический чертёж с кириллицей, отправляет его через API в профиль по умолчанию, ждёт публикации, скачивает DXF, проверяет SHA и открывает его через ezdxf. Результат записывается в `/app/artifacts/profile-service-v1/docker-smoke-result.json`. Тестовый документ остаётся в списке документов.

`smoke_edits.py` затем проверяет подготовку и публикацию размеченной области и правки названия через очереди `tables` и `corrections`, создавая новые версии тестового документа. Перед повторным запуском этого теста создайте новый тестовый документ командой `smoke_test.py`. В текущем Web V31 табличный редактор не принимает версии с `_manual_correction`: сначала выполняйте табличные правки, затем ручные правки метаданных. Это ограничение проверки исходного профиля приложения, независимо от Docker.

В `.env` можно указать `CAD_HTTP_PORT=8001` и `CAD_SERVICE_TIMEOUT_SECONDS=900`. Лимит времени относится к обработке одного задания; ожидание очереди добавляется отдельно. Для Docker Desktop выделите не менее 6 ГБ RAM. Каждый worker ограничен 4 ГБ; одновременная обработка документов и правок требует больше общей памяти.

## Остановка и обновление

После завершения текущих задач остановите сервис командой `docker compose down`. Для следующего запуска уже собранной версии выполните `docker compose up -d --wait --wait-timeout 300`.

Обновление после завершения текущей обработки:

```bash
git pull --ff-only
docker compose up --build -d --wait --wait-timeout 300
```

`docker compose down` сохраняет данные. `docker compose down -v` удаляет тома вместе с документами и БД; эта команда для обычной остановки не нужна.

## Резервная копия

Остановите приём/обработку, затем создайте согласованную копию базы и хранилища. Бинарные файлы копируются командами Docker, без перенаправления бинарного вывода PowerShell:

```bash
docker compose stop web worker corrections tables
docker compose exec db pg_dump -U cad_demo -d cad_recovery -Fc -f /tmp/cad-db.dump
docker compose cp db:/tmp/cad-db.dump ./cad-db.dump
docker compose run --rm --no-deps --entrypoint tar web -czf /app/artifacts/profile-service-v1/cad-storage.tar.gz -C /app/artifacts/profile-service-v1 storage
docker compose run --name cad-backup-copy --no-deps --entrypoint true web
docker cp cad-backup-copy:/app/artifacts/profile-service-v1/cad-storage.tar.gz ./cad-storage.tar.gz
docker rm cad-backup-copy
docker compose start web worker corrections tables
```

Храните обе копии вместе: база привязана к идентификатору файлового хранилища. Для восстановления остановите приложение, восстановите дамп в БД и `storage` в томе `documents` с владельцем UID/GID 10001, затем запустите приложение. Не подменяйте работающие тома частичной копией.

## Linux runtime

Сборщик `docker/prepare_runtime.py` проверяет исходные SHA из Windows compact index, переносит только исполняемые исходники/модели и устанавливает Linux-пакеты версий из `requirements.runtime.lock`. Windows DLL и PostgreSQL.exe в образ не попадают. Затем формируется отдельный Linux-инвентарь с SHA всех файлов и новым trust anchor. Проверки не отключаются.

В Linux применяется свободный DejaVu Sans с проверкой кириллицы и инженерных символов. В DXF записывается `DejaVuSans.ttf`; Cambria не распространяется. Ширина и внешний вид текста могут отличаться. Linux-профили получают отдельные идентификаторы; перенос Windows-базы с её bindings не поддерживается автоматически. Геометрические алгоритмы и ONNX-модели сохраняются, но полное равенство численных результатов между платформами не заявляется.
