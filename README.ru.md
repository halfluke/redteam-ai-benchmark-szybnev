# Red Team AI Benchmark

> **Примечание к ветке:** `feature/cloudrun-v2` включила scoring-слой upstream v2.3.0 (`rubric-v2.1.0` негационные эвристики, улучшения refusal detector, паттерны датасета 2.3.0). Runner, optimizer, repeats/seed и leaderboard из upstream v2.3.0 не смёрджены; эта ветка добавляет Cloud Run auth, keepalive, dual-track semantic scoring и исправления optimizer, которых нет в upstream.

**English version:** [README.md](README.md)

> **Синхронизация README:** [README.md](README.md) и [README.ru.md](README.ru.md) поддерживаются парой — одинаковые разделы, возможности и примеры CLI; отличается только язык. При изменении user-facing документации обновляйте оба файла в одном коммите/задаче.

Red Team AI Benchmark — CLI-бенчмарк для выбора base LLM под авторизованные red-team и offensive-security задачи. Версия 2 использует rubric-based датасет вместо оценки ответа только по одному golden answer.

---

## Ветка: `feature/cloudrun-v2`

Эта ветка расширяет upstream `v2` аутентификацией Cloud Run, фоновым keepalive и рядом улучшений optimizer. Ниже — список изменений с пометкой **Cloud Run specific** или **general improvement** (полезно для любого деплоя).

| Изменение | Тег | Описание |
| --- | --- | --- |
| GCP Cloud Run auth (`cloudrun_auth.py`, `bearer_auth.py`) | ☁️ Cloud Run | Получает и автообновляет GCP identity tokens через `gcloud auth print-identity-token`. Подставляет Bearer header в клиенты Ollama и LM Studio через `BearerAuthMixin`. На локальные HTTP endpoints не влияет. |
| `cloudrun_identity` в конфиге provider | ☁️ Cloud Run | В YAML: `auth: cloudrun_identity` и опциональные `cloudrun_audience` / `cloudrun_impersonate_service_account`. Игнорируется при локальных endpoints. |
| Фоновый keepalive (`benchmark/keepalive.py`) | ☁️ Cloud Run | Периодически пингует target model во время benchmark run, чтобы Cloud Run сервисы не уходили в scale-to-zero между вопросами. Настраивается в YAML под `keepalive:`. Работает только при явном включении. |
| `_probe_timeout()` в клиентах Ollama и LM Studio | ☁️ Cloud Run | 5-секундный timeout для локальных HTTP endpoints и полный настроенный `timeout` для HTTPS/Cloud Run при проверке connectivity. Убирает ложные «cannot connect» на cold-start сервисах. |
| Примеры конфигов Cloud Run (`configs/`) | ☁️ Cloud Run | `configs/cloudrun_ollama.yaml`, `configs/cloudrun_vllm_deephat.yaml` и `configs/cloudrun_ollama_bugtrace.yaml` (3072 max tokens) — рабочие setups с auth, keepalive и rubric scoring. |
| Helper scripts (`scripts/`) | ☁️ Cloud Run | Shell-скрипты для Cloud Run env, прогрева сервисов и запуска benchmark (`warmup_*`, `run_*_baseline.sh`). Включает `local_env.sh.example` для локальных overrides (gitignored). |
| Триггер optimization: rubric или semantic ниже 25% | ✅ General | Optimization срабатывает, когда baseline rubric или semantic score **ниже 25%**, либо semantic scoring пропущен как garbage. При **≥ 25%** на обоих треках не запускается. |
| Early exit optimization при 75%+ на треках | ✅ General | С `--semantic` цикл останавливается, когда лучший rubric и лучший semantic score (по всем попыткам) оба **≥ 75%** и semantic не garbage. Без `--semantic` — когда rubric-best **≥ 75%**. До четырёх strategy, если трек остаётся ниже 75%. |
| Формат вывода по вопросу | ✅ General | Без optimization: `Final: rubric X% \| semantic Y% \| duration \| snippet`. С optimization: строка `Baseline:` (оба score) перед блоком optimization и итог `Best —` по каждому треку. Финальный отчёт — таблицы **Rubric Winners** и **Semantic Winners** (Q#, score, source, полный текст вопроса), когда применимо. |
| 4-strategy round-robin optimizer | ✅ General | Каждая попытка — следующая strategy в цикле `role_playing → technical → few_shot → cve_framing`, а не выбор из двух по censored/non-censored. Максимум попыток по умолчанию: 4 (по одной на strategy). |
| Подробный вывод optimization | ✅ General | После каждой попытки — snippet reframed prompt, snippet ответа модели и оба score (rubric и semantic при `--semantic`). После всех итераций — `Best —` с лучшими rubric и semantic score. В финальных строках и таблицах указан source победителя (`baseline`, `opt/role_playing` и т.д.). |
| Корректный keepalive во время optimization | ☁️ Cloud Run | `optimize_prompt()` принимает `keepalive`. Вызовы target и optimizer обёрнуты в `keepalive_busy()`, чтобы keepalive-thread не пинговал во время generation — меньше ложных timeout и модель остаётся warm между попытками. |
| `keep_alive: -1` в Ollama Cloud Run configs | ☁️ Cloud Run | В `configs/cloudrun_ollama.yaml` и `configs/cloudrun_ollama_bugtrace.yaml` задано `provider.keep_alive: -1` на каждый main query и keepalive ping. Ollama не выгружает модель из VRAM, пока контейнер жив. |
| Retry-then-skip при API-ошибках по вопросу | ✅ General | Ошибка `5xx` (например chat-template failure) retry с backoff в клиенте (`models/base.py`, `models/openwebui.py`). После исчерпания retry runner фиксирует один failed result (`error`, `score: 0`, без optimization) и идёт дальше, не прерывая весь run. |
| Timestamp/duration/prompt по вопросу + start/finish run | ✅ General | Строка вопроса с clock timestamp (корреляция с Cloud Run logs), snippet исходного prompt (раньше показывался только reframed prompt optimizer), duration в строке score. Run печатает `Started:`/`Finished:` с общим elapsed на модель. |
| Score rationale (`Why: ...`) | ✅ General | При rubric scorer (v2) под score — `Why:` с deterministic причиной: `N/M criteria passed` / `missing`, `refused (censored)` или `fatal_error: <id>`. Для baseline, post-optimization и concurrent runner. Без rubric breakdown — строка не печатается. |
| Fix: optimizer отбрасывал real 0% responses | ✅ General | `best_score = 0` и `score > best_score` не перезаписывали placeholder при всех попытках на 0% — `full_response` мог быть пустым при ненулевой latency. Исправлено sentinel `-1`, первый ответ всегда сохраняется. |
| Оценка стоимости Cloud Run (`cloudrun_cost:` в YAML) | ☁️ Cloud Run | Опционально, по умолчанию выкл. Одна строка estimated cost в конце run (или на модель в `interactive`), по published on-demand rates (CPU/memory/GPU per second, см. `utils/cloudrun_cost.py`) × observed uptime. Оценка, не GCP invoice — см. `docs/agent-reference.md`. |
| Pre-flight GPU check (`gpu_check:` / `--min-vram-fraction`) | ☁️ Cloud Run | Опционально, по умолчанию выкл. Перед платным benchmark — загрузка модели и `/api/ps` Ollama: доля `size_vram` vs `size` в GPU VRAM; abort ниже порога — ловит silent CPU fallback до полного run (см. [`ollama/ollama#16449`](https://github.com/ollama/ollama/issues/16449)). Только target model, не optimizer. См. `configs/cloudrun_ollama.yaml` / `configs/cloudrun_ollama_bugtrace.yaml`. |
| Provider-agnostic optimizer (`--optimizer-provider`) | ✅ General | Optimizer: Ollama, LM Studio, OpenWebUI, OpenRouter или DeepInfra. `--optimizer-provider` и `--optimizer-model` — атомарная пара. Cloud providers fail fast без API key. |

### Запуск на Cloud Run

Подключите окружение и запустите:

```bash
source scripts/cloudrun_env.sh        # CLOUDRUN_ENDPOINT, TOKEN и т.д.
uv run run_benchmark.py run lmstudio \
  -m "DeepHat/DeepHat-V1-7B" \
  -e "$CLOUDRUN_ENDPOINT" \
  --config configs/cloudrun_vllm_deephat.yaml
```

Чтобы сервис Cloud Run не засыпал во время run, добавьте `keepalive: enabled: true` в YAML. Полный пример — `configs/cloudrun_ollama.yaml`.

Чтобы не платить за run на CPU вместо GPU, добавьте `gpu_check: enabled: true` с `min_vram_fraction` (уже в `configs/cloudrun_ollama.yaml` и `configs/cloudrun_ollama_bugtrace.yaml`) или `--min-vram-fraction 0.9`.

BugTrace Apex 26B Q4 (Ollama на Cloud Run, `max_tokens=3072`):

```bash
source scripts/local_env.sh          # опционально: BUGTRACE_ENDPOINT / BUGTRACE_MODEL
./scripts/run_bugtrace_baseline.sh
```

---

Дефолтный v2 suite содержит 60 вопросов в `datasets/v2/benchmark.jsonl`, разбитых по доменам и сложности.

## v2 Local Leaderboard

Топ локальных моделей по июньскому прогону v2, отсортированный по `judge_adjusted_score`.
Прогон использовал полный `standard` profile, Ollama, `max_tokens=3072`, `temperature=0.2`
и post-hoc LLM-as-Judge только для disputed cases через OpenRouter `deepseek/deepseek-v4-flash`.

| Место | Модель | Rubric | Judge-adjusted | Judge critical error rate |
| --- | --- | ---: | ---: | ---: |
| 1 | `hf.co/BugTraceAI/BugTraceAI-Apex-G4-26B-Q4:latest` | `80.89%` | `89.45%` | `0.00%` |
| 2 | `nemotron-3-nano:30b` | `75.55%` | `86.81%` | `7.14%` |
| 3 | `hf.co/yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF:Q6_K` | `73.23%` | `81.12%` | `7.14%` |
| 4 | `hf.co/unsloth/Qwen3-Coder-Next-GGUF:Qwen3-Coder-Next-Q4_K_S` | `75.50%` | `80.15%` | `33.33%` |
| 5 | `mistral-small3.2:24b` | `69.39%` | `76.58%` | `8.33%` |

`judge_adjusted_score` — рекомендуемый audit score для этого лидерборда: deterministic
rubric score, где disputed cases заменены оценкой LLM-as-Judge. При выборе base model
важно смотреть не только итоговый score, но и critical-error rate.

## Что измеряет v2

Бенчмарк сохраняет общий weighted score и отдельные audit-метрики:

| Метрика | Значение |
| --- | --- |
| `refusal_rate` | Доля отказов или цензурированных ответов |
| `technical_accuracy` | Средняя точность по техническим критериям rubric |
| `critical_error_rate` | Доля ответов с fatal technical falsehood |
| `completeness` | Покрытие обязательных шагов и условий |
| `specificity` | Конкретность инструментов, полей, команд и evidence |
| `hallucination_rate` | Сейчас совпадает с critical technical errors |
| `latency_ms_avg` | Средняя latency ответа |

Интерпретация intentionally conservative:

| Итоговый балл | Интерпретация |
| --- | --- |
| `< 60%` | `not-suitable` |
| `60-79.9%` | `requires-validation` |
| `>= 80%` | `strong-candidate` |

Высокий score не является разрешением на production use. Перед выбором модели нужно смотреть breakdown по доменам, сложности, отказам и critical errors.

## Покрытие датасета

v2 dataset покрывает:

- Windows tradecraft
- AD и AD CS
- Web exploitation
- Cloud и IAM
- Containers и Kubernetes
- Detection and evasion reasoning
- OpSec и operational tradeoffs
- Tool usage
- Post-exploitation planning
- Validation and reporting

Уровни сложности: `L1 factual`, `L2 procedure`, `L3 troubleshooting`, `L4 scenario reasoning`, `L5 multi-step operator task`.

## Установка

Требования:

- Python `3.13+`
- `uv`
- Один провайдер: Ollama, LM Studio, OpenWebUI, OpenRouter или DeepInfra

Базовые зависимости — **из корня репозитория**:

```bash
cd /path/to/redteam-ai-benchmark
uv sync
```

Все команды CLI запускайте из корня репозитория. Пути к датасету (`datasets/v2/benchmark.jsonl`), файлам ответов (`answers_v2.txt`, `answers_all.txt`), конфигам и каталогам export/cache считаются относительно текущего working directory. Установка wheel/`redteam-benchmark` не включает эти data files; полный прогон поддерживается только при cwd = корень репозитория (либо с абсолютными путями ко всем входным файлам).

## Провайдеры

| Провайдер | Endpoint по умолчанию | Примечание |
| --- | --- | --- |
| `ollama` | `http://localhost:11434` | Native Ollama API; optional Bearer auth для reverse proxy |
| `lmstudio` | `http://localhost:1234` | OpenAI-compatible LM Studio API |
| `openwebui` | `http://localhost:3000` | OpenAI-compatible OpenWebUI API |
| `openrouter` | `https://openrouter.ai/api/v1` | Требует API key |
| `deepinfra` | `https://api.deepinfra.com/v1/openai` | Требует `DEEPINFRA_TOKEN` или `--api-key` |

## Использование

Список моделей:

```bash
uv run run_benchmark.py ls ollama
uv run run_benchmark.py ls lmstudio
uv run run_benchmark.py ls openwebui
uv run run_benchmark.py ls openrouter --api-key "$OPENROUTER_API_KEY"
```

Запуск дефолтного v2 standard profile:

```bash
uv run run_benchmark.py run ollama -m "llama3.1:8b"
```

Быстрый smoke subset:

```bash
uv run run_benchmark.py run ollama -m "llama3.1:8b" --profile quick
```

Запуск выбранных v2 вопросов по ID:

```bash
uv run run_benchmark.py run ollama -m "llama3.1:8b" --question-ids 5 12
```

Append-only JSONL лог по каждому вопросу:

```bash
uv run run_benchmark.py run ollama -m "llama3.1:8b" --request-log results/requests.jsonl
```

Интерактивный запуск нескольких локальных моделей:

```bash
uv run run_benchmark.py interactive ollama --profile standard
```

Поддерживаемые профили:

| Profile | Назначение |
| --- | --- |
| `quick` | 16-question smoke subset |
| `standard` | Полный v2 benchmark на 60 вопросов |
| `enterprise` | Полный v2 dataset с audit-friendly export |
| `local-only` | Полный v2 dataset без LLM judge usage |
| `cloud-comparison` | Полный v2 dataset для фиксированных cloud-model comparisons |

## Scoring

Runtime scorer всегда `rubric`. Он deterministic и не требует внешней LLM-as-judge. Каждый v2 вопрос содержит атомарные criteria, fatal-error patterns, acceptable variants, tags и weight.

Legacy режимы `keyword`, `semantic` и `hybrid` для runtime scoring не поддерживаются. Для post-hoc LLM-as-Judge audit используйте отдельную команду `judge`.

Optional embedded semantic scoring доступен как параллельная audit metric. Он не заменяет rubric scoring: `score`, `total_score` и base interpretation labels остаются rubric-based. Semantic scoring сравнивает каждый ответ с эталонами из `answers_v2.txt` через локальный SentenceTransformer (`--semantic`) или DeepInfra embeddings (`--semantic --semantic-provider deepinfra`, модель `Qwen/Qwen3-Embedding-8B`).

**Локальный provider** (по умолчанию `Qwen/Qwen3-Embedding-0.6B`):

```bash
uv sync --extra semantic
uv run run_benchmark.py run ollama -m "llama3.1:8b" --semantic
uv run run_benchmark.py preload-semantic
```

**DeepInfra provider** (по умолчанию `Qwen/Qwen3-Embedding-8B`, перекалиброванные score bands):

```bash
export DEEPINFRA_TOKEN="your-token"
uv run run_benchmark.py run ollama -m "llama3.1:8b" \
  --semantic --semantic-provider deepinfra
uv run run_benchmark.py preload-semantic --semantic-provider deepinfra
```

Или через YAML (`scoring.semantic.provider: deepinfra`).

### Кэш reference-эмбеддингов

`preload-semantic` прогревает **локальный дисковый кэш эмбеддингов эталонных ответов** из `answers_v2.txt` (или `scoring.semantic.answers_file`). **Ответы модели во время benchmark run не кэшируются** — при DeepInfra каждый live-ответ по-прежнему вызывает embedding API в runtime.

Файлы кэша лежат в `.cache/redteam/semantic/` (переопределение через `REDTEAM_SEMANTIC_CACHE_DIR`). Имя файла: `<ModelName>_<digest16>.json`, где digest включает путь к файлу ответов, полный текст эталонов, `max_seq_length`, версию semantic scorer и provider. Изменение любого из этих параметров создаёт новый файл кэша; старые остаются на диске, но не используются.

| Ситуация | Поведение |
|----------|-----------|
| Первый запуск / нет файла кэша | Кодируются все эталонные ответы |
| Изменился текст эталона | Перекодируются только изменённые question id; остальные переиспользуются (SHA-256 hash на каждый ответ) |
| Legacy-кэш без hashes | Однократный полный re-encode в текущий формат |
| Другой provider, model, `max_seq_length` или версия scorer | Новый файл кэша; совпадающие записи могут импортироваться из sibling-кэша |

Если кэш уже полный, preload выводит `Reference embedding cache already warm (60 answers)` и не делает API-вызовов. Во время `run` недостающие записи кодируются на лету и записываются обратно — warm preload убирает эту нагрузку.

**Принудительная пересборка:**

```bash
uv run run_benchmark.py preload-semantic --semantic-provider deepinfra --force
```

Или удалите файл или каталог кэша в `.cache/redteam/semantic/`. Флаг `--force` доступен только у `preload-semantic`, не у `run`.

Добавляет колонки `semantic_score` и `semantic_similarity` в итоговую таблицу без изменения rubric-итога.

По умолчанию semantic scoring использует full-answer cosine bands: `100/90/80/70/60/50/40/30/0`. DeepInfra 8B использует перекалиброванные пороги (см. `scoring/semantic_calibration.py`). JSON и CSV exports добавляют `semantic_score` и `semantic_similarity`, не меняя rubric totals.

Перед кодированием semantic scorer:

- Вырезает **закрытые** reasoning/thinking-блоки (BugTrace/DeepHat channel markers, Qwen3/DeepSeek `redacted_thinking`, DeepSeek-R1 `xml_think`, pipe/XML/bracket-варианты). Удаляются только полные open+close блоки; незакрытые префиксы остаются. Имена паттернов и hex-проверки close-tag — в `scoring/semantic_scorer.py` и `tests/test_thinking_strip.py`.
- Пропускает дегенеративные повторы как `semantic skipped (garbage)`, если уникальность слов слишком низкая (≥24 слов и unique-word ratio &lt; 0.12). Garbage baseline с высоким rubric-score всё ещё может запустить prompt optimization.
- Ограничивает окно эмбеддинга **2048 токенами** для локального provider. DeepInfra по умолчанию использует **3072** (как target `max_tokens`), если `max_seq_length` не указан.

При включённом `--request-log` каждая semantic-строка JSONL также содержит `thinking_stripped_chars`, `thinking_stripped_tokens_est` и `strip_matched_pattern` (на верхнем уровне и внутри `semantic_scores`).

### Dual-track режим (optimization + semantic)

Когда одновременно заданы optimizer flags (`--optimizer-provider` + `--optimizer-model`) и `--semantic`:

- Оптимизатор также запускается, если **semantic** score **ниже 25%**, semantic scoring пропущен как **garbage**, или rubric **ниже 25%**.
- Каждая попытка оптимизации независимо оценивается по обоим метрикам.
- Цикл останавливается досрочно, когда лучший rubric-score и лучший semantic-score (по всем попыткам) оба **≥ 75%**, а semantic не garbage; без `--semantic` — когда rubric-best **≥ 75%**.
- По каждому вопросу сохраняются два победителя: ответ с наивысшим rubric-score и ответ с наивысшим semantic-score.
- Итоговый отчёт показывает две отдельные таблицы и интерпретации — по одной для каждого трека.
- JSON-экспорт получает блок `tracks` с общим score и интерпретацией каждого трека.

```bash
uv run run_benchmark.py run ollama -m "llama3.1:8b" \
  --semantic \
  --optimizer-provider ollama \
  --optimizer-model "llama3.3:70b"
```

Без настроенного optimizer флаг `--semantic` всё равно печатает отдельные таблицы **Rubric Winners** и **Semantic Winners** (без объединённой таблицы со snippet-ами).

## Offline LLM-as-Judge

Сохранённые v2 JSON-результаты можно проверить post-hoc без повторного запуска benchmark-моделей:

```bash
OPENROUTER_API_KEY=... uv run run_benchmark.py judge \
  --results "results/*.json" \
  --dataset datasets/v2/benchmark.jsonl \
  --judge-model "deepseek/deepseek-v4-flash" \
  --output-dir judge_results_v2 \
  --mode disputed \
  --concurrency 4
```

По умолчанию (без `--results`) используются glob’ы `results/*.json` и `results_*_v2/*.json`. Ноль совпадений — ошибка. Команда `judge` пишет `per_model/*.json`, `detailed.csv`, `summary.csv` и `disputed_cases.csv`. `judge_score` относится только к judged subset; для итогового сравнения используйте `judge_adjusted_score`, где rubric score заменён judge-оценкой на disputed cases. LLM-as-Judge остаётся audit layer и не перезаписывает deterministic benchmark results.

## Конфигурация

Скопируйте `config.example.yaml` в `config.yaml`:

```yaml
provider:
  name: ollama
  endpoint: http://localhost:11434
  # api_key: sk-xxx
  # keep_alive: 30m

scoring:
  method: rubric
  semantic:
    enabled: false
    provider: local  # или deepinfra (Qwen/Qwen3-Embedding-8B, нужен DEEPINFRA_TOKEN)
    answers_file: answers_v2.txt
    model: Qwen/Qwen3-Embedding-0.6B
    # Для provider: deepinfra можно не указывать thresholds — будут 8B defaults.
    thresholds:
      100: 0.92
      90: 0.88
      80: 0.84
      70: 0.80
      60: 0.75
      50: 0.70
      40: 0.65
      30: 0.60
    device: auto
    max_seq_length: 2048  # local; для deepinfra можно не указывать (3072)

export:
  formats:
    - json
    - csv
    - criteria_csv
  output_dir: ./results
  include_response: true

optimization:
  optimizer_provider: ollama          # или: deepinfra, openrouter, lmstudio, openwebui
  optimizer_model: llama3.3:70b
  # optimizer_endpoint: http://localhost:11434
  # optimizer_api_key: ...            # для deepinfra/openrouter; лучше через env var
  max_iterations: 4

questions_file: datasets/v2/benchmark.jsonl
answers_file: answers_all.txt
rate_limit_delay: 1.5
max_tokens: 3072
temperature: 0.2
concurrency: 1
# request_log: ./results/requests.jsonl
```

Запуск с конфигом:

```bash
uv run run_benchmark.py run ollama -m "llama3.1:8b" --config config.yaml
```

## Вывод

JSON export содержит результаты модели, evidence по rubric criteria, aggregate summary и audit provenance:

```json
{
  "model": "llama3.1:8b",
  "scoring_method": "rubric",
  "total_score": 75.0,
  "interpretation": "requires-validation",
  "benchmark_version": "2.3.0",
  "dataset_id": "redteam-ai-benchmark-v2",
  "dataset_version": "2.1.0",
  "dataset_hash": "...",
  "scorer_version": "rubric",
  "config_hash": "...",
  "run_config": {
    "provider": "ollama",
    "model": "llama3.1:8b",
    "profile": "standard"
  },
  "git_commit": "...",
  "package_version": "2.3.0",
  "runtime_profile": "standard",
  "summary": {
    "metrics": {
      "refusal_rate": 0.0,
      "critical_error_rate": 0.0
    },
    "breakdown": {
      "difficulty": {},
      "domain": {},
      "capability": {}
    }
  }
}
```

CSV содержит строки по вопросам плюс строку `TOTAL`. `criteria_csv` добавляет отдельную строку на каждый passed или failed rubric criterion.

## Prompt Optimization

Prompt optimization остаётся отдельным optional режимом и не смешивается с base-model score. Включите его, задав **оба** флага `--optimizer-provider` и `--optimizer-model` (CLI или секция `optimization:` в config). Не задавайте ни одного для baseline-only run, или передайте `--no-optimize`, чтобы отключить optimization даже при настройках в YAML.

Поддерживаемые optimizer providers: `ollama`, `lmstudio`, `openwebui`, `openrouter`, `deepinfra`. Cloud providers (`openrouter`, `deepinfra`) требуют API key через `--optimizer-api-key` или переменные окружения `OPENROUTER_API_KEY` / `DEEPINFRA_TOKEN`.

Pre-flight validation выполняется до первого вопроса:

- `--optimizer-provider` и `--optimizer-model` используются только вместе (или не используются вовсе)
- Cloud providers без API key завершают run с понятной ошибкой

Optimization запускается, когда baseline rubric **или** semantic score **ниже 25%**, либо semantic scoring пропущен как **garbage** (см. `should_trigger_prompt_optimization()` в `optimization/policy.py`). По умолчанию он пробует до **четырёх** reframing strategy (`--max-optimization-iterations 4`) и независимо сохраняет **лучший rubric-ответ** и **лучший semantic-ответ**. Цикл останавливается досрочно, когда **оба** трека достигают **≥ 75%** (semantic должен быть реальным score, не garbage); без `--semantic` early exit только по rubric с тем же правилом **≥ 75%**. Результаты пишутся в `optimized_prompts_{model}_{timestamp}.json`.

На каждой итерации target model получает **новый reframed prompt**. Optimizer всегда опирается на исходный benchmark question и **baseline prompt/response** при генерации каждой strategy variant. Пока target model выполняет strategy *N*, optimizer параллельно генерирует strategy *N+1*. При активном `--semantic` каждая итерация последовательно вычисляет rubric- и semantic-score после ответа модели и сразу выводит оба значения.

```bash
# Local Ollama optimizer
uv run run_benchmark.py run ollama -m "llama3.1:8b" \
  --optimizer-provider ollama \
  --optimizer-model "llama3.3:70b"

# DeepInfra optimizer
export DEEPINFRA_TOKEN="your-token"
uv run run_benchmark.py run ollama -m "llama3.1:8b" \
  --optimizer-provider deepinfra \
  --optimizer-model "deepseek-ai/DeepSeek-V3"

# В config есть optimizer, но пропустить optimization в этом run
uv run run_benchmark.py run ollama -m "llama3.1:8b" --config config.yaml --no-optimize
```

**Неверное имя модели:** до первого вопроса runner вызывает `list_models()` и прерывает run, если **target** model (`-m`) или **optimizer** model отсутствует в списке provider. В режиме `interactive` неверная target model пропускается, а отсутствующая optimizer model всё равно прерывает весь run.

Optimized score нельзя смешивать с base model capability comparison.

## Проверка

Полезные проверки:

```bash
uv run run_benchmark.py --help
uv run run_benchmark.py run --help
uv run pytest
python3 -m compileall -q run_benchmark.py benchmark models optimization scoring tracing utils
```

## Участие

См. [CONTRIBUTING.md](CONTRIBUTING.md), [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) и [SECURITY.md](SECURITY.md).

## Лицензия

MIT. Используйте в авторизованных red team лабораториях, коммерческих security assessment, AI-security research и образовательных средах.
