# Запуск ScannedDrawings2CAD после клонирования

## Рекомендуемый запуск: Docker Compose

Установите Docker Desktop с Linux containers (Windows/macOS) или Docker Engine с Compose v2 (Linux). Python на компьютере и runtime ZIP не нужны. Для Docker Desktop выделите не менее 6 ГБ памяти; обработка работает на CPU. Образ собирается для `linux/amd64`.

```bash
git clone https://github.com/pudoff/ScannedDrawings2CAD.git
cd ScannedDrawings2CAD
docker compose up --build -d
docker compose ps
```

Первый запуск скачивает Python, PostgreSQL и библиотеки и может занять несколько минут. Когда `web` станет `healthy`, откройте **http://127.0.0.1:8000**. `setup` должен завершиться с кодом 0: это одноразовая подготовка БД. Compose запускает Web V31, обработчик чертежей и отдельные обработчики текстовых/табличных правок.

```bash
docker compose logs -f web worker corrections tables
docker compose down
# Следующий запуск с сохранёнными документами
docker compose up -d
```

База и документы сохраняются в Docker volumes `database` и `documents`. `docker compose down` их сохраняет; **не добавляйте `-v`**, если документы нужны. Локальная Windows-БД автоматически не переносится. Если порт 8000 занят, создайте `.env` в корне со строкой `CAD_HTTP_PORT=8001` и откройте http://127.0.0.1:8001.

Docker использует **DejaVu Sans** вместо Cambria; метрики и внешний вид CAD-текста могут отличаться от Windows-сборки. Для одинакового отображения DXF установите DejaVu Sans и на компьютере с CAD. Проверки исходников/моделей сохраняются; при сборке создаётся отдельный Linux-индекс и идентификаторы профилей.

Проверка компонентов и полный тест преобразования (создаёт тестовый документ):

```bash
docker compose run --rm --no-deps web check
docker compose exec web python /app/docker/smoke_test.py
```

Детали томов, обновления и резервного копирования — [DOCKER.md](DOCKER.md).

## Альтернатива: локальный Windows runtime

Проверенная платформа: **Windows x64, CPython 3.14**. Текущая поставка — чистая сборка на базе **Web V31**, профиль по умолчанию — **«CAD · целые объекты и рамка (эксперимент)»**. GPU для этого конвейера не требуется.

## 1. Установить Python

Установите Windows x64 CPython 3.14 с [python.org](https://www.python.org/downloads/windows/), включая Python Launcher (`py`). В новом PowerShell проверьте:

```powershell
py -3.14 --version
```

Команды ниже выполняются **из корня клонированного ScannedDrawings2CAD**. Не нужно использовать путь к ProjectPractice44Labs.

## 2. Получить runtime-архив

После `git clone` каталогов `.venv`, библиотек и PostgreSQL **ещё нет**: это генерируемое окружение и бинарные зависимости, исключённые из Git. Одного `pip install -e .` недостаточно.

Нужен архив **`runtime-win-cp314-v31-20261125.zip`**, соответствующий файлу `runtime-distribution.json` из этой версии репозитория. Он содержит:

- точные сервисные библиотеки, NumPy/OpenCV/ezdxf/RapidOCR/ONNX Runtime и прочие зависимости;
- CV-библиотеки `scikit-image`/SciPy;
- три ONNX-модели PP-OCRv5;
- готовый PostgreSQL runtime, без пользовательской базы данных.

Получите архив у владельца проекта или скачайте вложение из **GitHub → Releases**. Если владелец ещё не приложил архив к Release, в GitHub его нет: передайте владельцу имя архива выше. Не ищите `.venv` в репозитории и не копируйте чужую `.venv`.

## 3. Создать `.venv` и установить комплект

Если архив находится в Downloads:

```powershell
py -3.14 setup_repository.py --runtime-archive "$env:USERPROFILE\Downloads\runtime-win-cp314-v31-20261125.zip"
```

Установщик проверит SHA-256/размер архива и каждого файла, восстановит библиотеки/модели/PostgreSQL, создаст окружение **`.venv`** стандартным модулем `venv`, затем выполнит `--check`. Системные Python-пакеты менять не требуется: конвейер использует закреплённые библиотеки из runtime. Интернет для установки из локального архива не нужен.

Если владелец дал прямую HTTPS-ссылку на нужный архив, вместо `--runtime-archive` можно указать `--runtime-url "https://.../runtime-win-cp314-v31-20261125.zip"`. SHA берётся из репозитория, а не из скачанного файла. Сам архив не запускается как программа.

Эквивалент создания окружения вручную:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe setup_repository.py --runtime-archive "$env:USERPROFILE\Downloads\runtime-win-cp314-v31-20261125.zip"
```

Активация `.venv` через `Activate.ps1` не нужна. Не выполняйте `pip install -r requirements.runtime.lock` поверх каталогов runtime: lock здесь справочный список версий; точные файлы определяет `runtime/compact/index.json`.

## 4. Проверить и запустить

```powershell
# Проверка компонентов, моделей и версий; сервер и БД не запускаются
.\.venv\Scripts\python.exe run_service.py --check

# Web, три worker и локальная БД; открыть браузер
.\.venv\Scripts\python.exe run_service.py --open
```

Адрес: **http://127.0.0.1:8000**. Загрузите PNG/JPEG/скан-PDF, дождитесь результата и скачайте DXF. Новый default уже выбран. `run_service.py` запускает проверенный `scripts/start_clean_demo_v1.py`: это компактная упаковка исправлений V28–V31; старые команды `start_profiles_v27.py` больше не использовать.

Остановка — **Ctrl+C** в терминале запуска; дождитесь штатного завершения PostgreSQL. Либо во втором PowerShell:

```powershell
.\.venv\Scripts\python.exe run_service.py --stop
```

При занятых портах:

```powershell
.\.venv\Scripts\python.exe run_service.py --port 8001 --db-port 55446 --open
```

Первый запуск создаёт пустую локальную БД; PostgreSQL может запускаться/восстанавливаться до 300 секунд. Документы сохраняются при остановке. Не запускайте второй экземпляр поверх уже работающего.

## Если запуск не получается

| Сообщение | Действие |
|---|---|
| `py` не найден / Python 3.14 отсутствует | Установите CPython 3.14 x64 и Launcher; откройте новый терминал |
| `.venv\Scripts\python.exe` не найден | Выполните шаг 3; `.venv` создаётся на вашей машине |
| Runtime отсутствует | Получите указанный ZIP и выполните шаг 3; файлы исходного исследовательского проекта не нужны |
| SHA/размер не совпадает | Архив от другой версии или повреждён; получите архив именно для текущего `runtime-distribution.json` |
| `Unexpected runtime file` | Не смешивайте две версии src/scripts/runtime; сохраните документы и используйте чистый checkout этой версии |
| Ошибка `cambria.ttc` | Сборка проверяет установленный CAD-шрифт Cambria по SHA. Windows может содержать другую версию. Сообщите владельцу фактический SHA; не заменяйте шрифт случайным файлом. Совместимость другой версии шрифта должна быть отдельно проверена |
| Ошибка загрузки DLL | Проверьте Windows x64/CPython 3.14 x64; при требовании системного VC++ Runtime установите его с сайта Microsoft, не копируйте случайные DLL |
| PostgreSQL не готов | Смотрите `postgres.log` в каталоге логов; не удаляйте существующую БД для повторной попытки |

Для CAD используется установленный в Windows **Cambria**. Шрифт не включён в архив и не скачивается установщиком. Ожидаемый SHA: `84e70ccc1664482f4a960442c7a166c91a1b2cf98ff88c33cb73f79403f66d7b`. Команда проверки:

```powershell
Get-FileHash "$env:WINDIR\Fonts\cambria.ttc" -Algorithm SHA256
```

Логи: `artifacts/profile-service-clean-v1/logs/`. Документы: `artifacts/profile-service-v1/storage/`. БД: `artifacts/demo-runtime-v1/postgres/data/`. Эти каталоги не публикуются в Git. Проверка на другом ПК и ручная приёмка CAD остаются отдельными этапами; качество распознавания геометрии/текста не стало выше от обновления установщика.
