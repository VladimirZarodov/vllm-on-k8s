# vLLM CPU в Kubernetes (Minikube) с безопасным логированием

Локальное CPU-развёртывание vLLM 0.32.0 с моделью `Qwen/Qwen2.5-0.5B-Instruct`:
модель хранится на PVC, скачивается с Hugging Face один раз init-контейнером,
токены лежат в Kubernetes Secret, параметры запуска — в ConfigMap, а логи
запросов фильтруются так, чтобы промпты, ответы и токены в них не попадали.

## Архитектура

```
Minikube (driver=docker)
└── namespace vllm
    ├── PVC model-cache (8Gi, storageClass standard, hostpath) → /models
    ├── Secret vllm-secrets (HF_TOKEN, VLLM_API_KEY)   
    ├── ConfigMap vllm-config (параметры vLLM)          
    ├── Deployment vllm (1 реплика, strategy Recreate)
    │   ├── initContainer model-downloader: python /opt/scripts/download_model.py    
    │   │     
    │   └── container vllm: vllm serve /models/model (локальный путь, HF_HUB_OFFLINE=1)
    └── Service vllm (ClusterIP :8000) → kubectl port-forward
```

Ключевые механизмы:

- **Повторное использование модели.** Init-контейнер после успешной загрузки
  и проверки (`config.json`, веса по `model.safetensors.index.json`, tokenizer)
  пишет маркер `.model_ready.json`. При следующих стартах Pod, если маркер и
  файлы валидны, загрузчик завершается **без обращения к Hugging Face**.
  Основной контейнер получает `HF_HUB_OFFLINE=1` и модель по локальному пути
  `/models/model` — случайное повторное скачивание исключено.
- **Rootless.** Образ работает под UID/GID 1000 (`USER 1000:1000`), Pod —
  `runAsNonRoot: true`, `capabilities: drop ALL`, `allowPrivilegeEscalation: false`,
  том PVC доступен на запись за счёт `fsGroup: 1000`.
- **Аутентификация.** vLLM 0.32.0 включает Bearer-auth из переменной окружения
  `VLLM_API_KEY` (см. `vllm/entrypoints/serve/middleware/register.py`). Она
  приезжает из Secret и **не** попадает в `args` контейнера — так ключ не виден
  в `kubectl describe pod`, в cmdline процесса и в служебных логах запуска.
  Защищены префиксы `/v1`, `/v2`, `/inference`, `/cohere`; `/health` открыт —
  на него настроены пробы.
- **Приватное логирование.** `--logging-config.pylogging_config_file`
  подключает `logging_config.json` (dictConfig), где handler `vllm` оснащён
  фильтром `vllm_privacy.privacy_logging.PrivacyFilter`:
  - из `request_logger` остаётся только «Received request <id>: request
    accepted» (параметры, промпты, тела, ответы — отбрасываются);
  - access-лог uvicorn сохраняет метод, путь, статус и длительность — факт
    запроса виден, содержимое нет;
  - логгеры приложения (`vllm.*`) пишутся как обычно.
  Флаг `--enable-log-requests` по умолчанию **не включён** (vLLM 0.32.0 по
  умолчанию его не логирует содержимое запросов вовсе; фильтр — дополнительная
  защита на случай его включения).

## Требования

- Windows 11 + PowerShell;
- Docker Desktop (WSL2), 8+ ГБ RAM хоста, **память Docker-VM увеличена до 6 ГБ**
  (Settings → Resources → Memory, либо `%USERPROFILE%\.wslconfig`:
  `[wsl2] memory=6GB` + `wsl --shutdown`). Это требование данной сборки:
  vLLM CPU с 0.5B-моделью занимает ~4.2 GiB steady-state, плюс ~0.7 GiB на
  контрол-плейн Kubernetes;
- `minikube`, `kubectl` в `PATH`;
- Python 3.12 на хосте — только для запуска тестов;
- Образ `vllm-cpu:dev` собирается локально (требует ~6 ГБ свободного места).

## 1. Сборка образа и запуск Minikube

Все команды выполняются в корне репозитория.

```powershell
# сборка rootless-образа (pip-слой кешируется, повторная сборка быстрая)
docker build -f docker/Dockerfile -t vllm-cpu:dev .

# Minikube должен иметь не меньше 5400 MiB памяти (см. Требования):
minikube config set memory 5400
minikube start --driver=docker --cpus=4 --disk-size=20g
# если кластер уже был создан раньше с 4608 MiB и пересоздавать его нельзя
# (на PVC лежит модель) — поднять лимит живого контейнера можно так:
docker update --memory 5400m --memory-swap 6100m minikube

# загрузка образа в кластер (иначе Pod будет тянуть его из registry и не найдёт)
minikube image load vllm-cpu:dev
minikube image ls | Select-String vllm-cpu    # должно показать docker.io/library/vllm-cpu:dev
```

## 2. Namespace и Secret (обязательная подготовка)

`kubectl apply -k` применит Deployment, который ссылается на Secret
`vllm-secrets`. Секрет **намеренно не включён в Kustomize** — его значения
не должны попадать в репозиторий. Порядок действий:

```powershell
# 1) namespace (идемпотентно)
kubectl apply -f k8s/namespace.yaml

# 2) токен Hugging Face (нужен только для gated-моделей; для Qwen2.5-0.5B-Instruct
#    можно оставить пустым — загрузка пойдёт анонимно).
#    Задайте значения в переменные, чтобы они не попадали в историю буквально.
$env:HF_TOKEN = ""                            # или реальный hf_... токен
# 3) ключ доступа к нашему API — сгенерируйте свой:
$env:VLLM_API_KEY = [guid]::NewGuid().ToString("N")

# 4) создание Secret из переменных окружения PowerShell:
kubectl create secret generic vllm-secrets -n vllm `
  --from-literal=HF_TOKEN="$env:HF_TOKEN" `
  --from-literal=VLLM_API_KEY="$env:VLLM_API_KEY" `
  --dry-run=client -o yaml | kubectl apply -f -
```

Вариант, при котором значения не появляются даже в командной строке процесса:
сохранить их в файлы (вне репозитория) и использовать
`--from-file=HF_TOKEN=путь --from-file=VLLM_API_KEY=путь`. Так сделано в
автоматизированной проверке этого репозитория: `.checks/` в `.gitignore`.

## 3. Развёртывание

```powershell
kubectl apply -k k8s/
kubectl get pvc -n vllm          # model-cache -> Bound
kubectl get pods -n vllm -w      # дождаться Ready 1/1
```

Первый запуск включает скачивание ~950 МБ с Hugging Face — на медленной сети
это 10–20 минут. Прогресс init-контейнера:

```powershell
kubectl logs -n vllm deploy/vllm -c model-downloader -f
```

После завершения появятся `completion marker written` и старт vLLM. Готовность:

```powershell
kubectl get deploy vllm -n vllm              # READY 1/1
kubectl logs -n vllm deploy/vllm -c vllm | Select-String "Application startup complete"
```

## 4. Доступ к API и проверка авторизации

```powershell
# в отдельном окне PowerShell (или оставьте запущенным в фоне):
kubectl port-forward -n vllm svc/vllm 8000:8000
```

Без токена — ожидаем отказ:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/v1/chat/completions -Method Post `
  -ContentType "application/json" `
  -Body '{"model":"qwen2.5-0.5b-instruct","messages":[{"role":"user","content":"hi"}]}'
# ошибка HTTP 401 (Unauthorized) — так и должно быть
```

С токеном (`VLLM_API_KEY` из шага 2 или из файла, если сохраняли):

```powershell
$headers = @{ Authorization = "Bearer $env:VLLM_API_KEY" }
$body = @{
  model      = "qwen2.5-0.5b-instruct"
  messages   = @(@{ role = "user"; content = "Привет! Один короткий абзац." })
  max_tokens = 32
  temperature = 0
} | ConvertTo-Json -Depth 5
Invoke-RestMethod -Uri http://127.0.0.1:8000/v1/chat/completions -Method Post `
  -Headers $headers -ContentType "application/json" -Body $body
```

Внутри кластера адрес сервиса: `http://vllm.vllm.svc.cluster.local:8000`.

## 5. Проверка повторного запуска без скачивания

```powershell
kubectl rollout restart deployment vllm -n vllm
kubectl -n vllm wait --for=condition=Available deploy/vllm --timeout=600s
kubectl logs -n vllm deploy/vllm -c model-downloader
# ожидаем: "model 'Qwen/Qwen2.5-0.5B-Instruct' is already complete (marker + files valid)"
#         "skipping download: no network access required"
```

## 6. Тесты

Локально, без кластера (чистая stdlib):

```powershell
python tests/test_privacy_filter.py
```

Интеграционный тест на живом сервере — шлёт запрос с уникальным маркером,
проверяет 401 без токена, 200 с токеном и отсутствие маркера/ключа в логах:

```powershell
$env:VLLM_TEST_API_KEY = $env:VLLM_API_KEY   # значение из Secret
python tests/test_logging_privacy_live.py
```

Ручная проверка логов после тестового запроса:

```powershell
kubectl logs -n vllm deploy/vllm --all-containers --since=10m |
  Select-String "chat/completions"       # факт запроса виден (access log, статус)
kubectl logs -n vllm deploy/vllm --all-containers --since=10m |
  Select-String "UNSEEN-PROMPT"          # эта строка должна остаться ПУСТОЙ
```

## 7. Диагностика

```powershell
kubectl describe pod -n vllm -l app=vllm
kubectl logs -n vllm deploy/vllm -c model-downloader --previous
kubectl top pod -n vllm                            # потребление памяти
kubectl exec -n vllm deploy/vllm -c vllm -- ls -la /models/model   # содержимое PVC
kubectl get events -n vllm --sort-by=.lastTimestamp | Select-Object -Last 20
```

Частые проблемы:

- `CreateContainerConfigError` — нет Secret `vllm-secrets` (шаг 2);
- `ErrImagePull`/`ImagePullBackOff` — образ не загружен в Minikube
  (`minikube image load vllm-cpu:dev`);
- Pod в `Pending` с `Insufficient memory` — уменьшите requests/limits в
  `k8s/deployment.yaml` или добавьте памяти Minikube;
- init-контейнер падает с `PVC is not writable` — не сработал `fsGroup`
  (проверьте, что PV создан provisioner `standard` и не пересоздан вручную).

## 8. Очистка

```powershell
kubectl delete -k k8s/                       # ресурсы приложения (PVC удалится вместе)
# данные модели (hostpath) останутся на диске Minikube — для полного удаления:
kubectl delete pvc model-cache -n vllm
kubectl delete secret vllm-secrets -n vllm
kubectl delete namespace vllm
minikube image rm docker.io/library/vllm-cpu:dev
```

## Известные ограничения

- **Память.** Хост 8 ГБ → Docker-VM 6 ГБ → Minikube 5400 MiB. Ровно одна
  реплика. Замеренный профиль vLLM 0.32.0 CPU (три процесса: APIServer ~0.8,
  EngineCore ~0.5, Worker с весами 0.94 GiB ~2.4 GiB): steady-state ~4.1 GiB,
  пик при генерации ~4.3 GiB — при лимите контейнера 4500Mi и swap=0.
  Обязательно `LD_PRELOAD=libtcmalloc_minimal.so.4` (есть в образе): без него
  тот же workload требует ~4.9 GiB (glibc-арены) и не помещается.
  KV-кэш задаётся байтами через ConfigMap `KV_CACHE_MEMORY_BYTES` (в CLI как
  `--kv-cache-memory-bytes`, 256 MiB ≈ 21k токенов), **не** через
  `VLLM_CPU_KVCACHE_SPACE` — тот принимает только целые GiB и переопределяет
  CLI-флаг, поэтому переменная в контейнере намеренно не установлена.
  Отклонения от исходного `docker run`-конфига (KV 1 GiB, batched-tokens 8192
  по умолчанию): KV 256 MiB и `--max-num-batched-tokens 1024` — реального
  запроса хватает с запасом, гигабайт KV на этой машине не помещается.
  `/dev/shm` — emptyDir Memory 512 МиБ (при пересоздании Pod чистый — важно:
  пережитые контейнер-рестарты shm-сегменты накапливаются в Pod-томе).
- **Первый запуск медленный.** Инференс на 4 vCPU CPU-бэкендом: генерация
  десятков токентов занимает минуты; живые тесты рассчитаны на таймаут 900 с.
- **PVC `ReadWriteOnce` + strategy `Recreate`:** обновление последовательное,
  две реплики с общей записью на hostpath недопустимы.
- **`--api-key`/`VLLM_API_KEY` vLLM защищает только `/v1`, `/v2`, `/inference`,
  `/cohere`.** Служебные `/health`, `/metrics` открыты (это же используется
  пробами); в кластере доступ к Service имеют только внутренние клиенты —
  наружу сервис не публикуется. Для продакшена — ingress с аутентификацией на
  уровне прокси.
- **Скачивание анонимное**, если `HF_TOKEN` пуст; для gated-моделей задайте
  реальный токен в Secret.
- Сжатие памяти (`hf_xet`) и rate-limit HF могут замедлять первую загрузку;
  init-контейнер делает до 3 попыток, файлы докачиваются при рестарте.
- Логи: при `--enable-log-requests=True` фильтр скрывает содержимое, но не
  включайте DEBUG-уровень логгеров `vllm.*` без необходимости.
- `VLLM_API_KEY` ротировать можно перезапуском Secret + `kubectl rollout
  restart deployment vllm -n vllm`.

## Структура

```text
docker/Dockerfile            # rootless CPU-образ (python:3.12-slim по digest)
scripts/download_model.py    # init-контейнер: загрузка+валидация+маркер
vllm_privacy/privacy_logging.py  # PrivacyFilter
logging_config.json          # dictConfig для vLLM
requirements.txt             # фиксированные зависимости (vLLM 0.32.0 CPU)
tests/                       # unit + live-тесты приватности/авторизации
k8s/                         # kustomize: namespace, pvc, configmap, deployment, service
```
