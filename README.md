# Rental Ads Analyzer — Structured Extraction Service

ДЗ 1, AI Engineering (2 потік). Сервіс витягує з «брудних» оголошень про оренду
(українська/російська, помилки, пропуски) валідний JSON за Pydantic-схемою і порівнює
модель провайдера (Gemini) з локальною моделлю (Ollama, qwen2.5:3b).

Підсумки та вибір моделі для продакшну — у [REPORT.md](REPORT.md).

## Структура

```
src/
  schema.py        Pydantic-схема RentalListing (9 полів верхнього рівня, enum'и, вкладені Price/Location, список amenities)
  extractors.py    GeminiExtractor і OllamaExtractor зі спільним інтерфейсом extract(prompt, ad)
  config.py        моделі, семплінг (t0, t07), ціни, таймаути, retry, id golden set
  run.py           один прогон: retry при помилках API, 1 спроба виправлення невалідної відповіді, лог у results/
  metrics.py       точність по полях, % валідних, latency p50/p95, вартість на 1000 документів
  check_golden.py  перевірка golden set на відповідність схемі
prompts/           v1.txt, v2.txt, CHANGELOG.md
data/ads/          30 оголошень (001–030), без персональних даних
data/golden.jsonl  10 оголошень, розмічених вручну
results/           *.jsonl за прогонами + metrics.md
main.py            запуск однією командою
```

## Встановлення

Python 3.11+ (перевірено на 3.14), Windows/Linux/macOS.

```
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env          # Linux/macOS: cp .env.example .env
```

У `.env` вказати `GEMINI_API_KEY` (ключ з Google AI Studio).

Локальна модель: встановити [Ollama](https://ollama.com), потім `ollama pull qwen2.5:3b`
(має працювати `ollama serve`).

## Запуск

```
python main.py             # Gemini v2 t0 на 30 текстах + метрики за всіма results/, ~4 хв (потрібен лише GEMINI_API_KEY)
python main.py --all       # уся матриця: 2 моделі × 2 промпти × 2 семплінги + метрики (потрібна Ollama, ~1 год на CPU)
python main.py --metrics   # лише перерахувати метрики за results/ (без API і Ollama)
```

Окремий прогон: `python src/run.py <gemini|ollama> <t0|t07> <v1|v2> <all|golden>`
→ `results/{provider}_{prompt}_{sampling}.jsonl`.

Кожен рядок результату містить модель, версію промпту, temperature/top_p, статус
(`ok` / `fixed` / `failed` / `api_error`), відповідь, помилку валідації, latency і токени.

## Обмеження

- Ollama на CPU (Ryzen 3 3200U) ≈ 50–80 с на документ, тому вона проганяється лише на golden set.
- Безкоштовний рівень Gemini: 15 запитів/хв → пауза 5 с між запитами.

---

# ДЗ 2 — RAG по документах про оренду житла + evals у CI

Питання українською → гібридний пошук по корпусу (закон, роз'яснення ДПС, статті) →
відповідь з посиланнями `[n]` на джерела або чесне «немає відповіді».
Метрики та висновки — у [REPORT.md](REPORT.md) (розділ ДЗ 2).

## Пайплайн

```
data/raw/ (22 HTML + 9 PDF, manifest.csv)
  └─ src/rag/ingest.py   очистка: trafilatura (HTML), pymupdf (PDF, в т.ч. 3-колонковий буклет),
                         видалення колонтитулів і рядків-«шапок», спільних для сторінок домену
       → data/processed/docs.jsonl        (doc_id, title, doc_type, doc_date, url, text)
  └─ src/rag/chunk.py    закон: 1 стаття ЦК = 1 чанк (поле article); інше: ≤1000 симв. по рядках, overlap 1
       → data/processed/chunks.jsonl      (168 чанків + метадані документа)
  └─ src/rag/index.py    gemini-embedding-001, 768 вимірів, title+text
       → data/processed/embeddings.npy, chunk_ids.json
  └─ src/rag/retrieve.py BM25 (стемінг обрізкою до 5 літер) + вектор → RRF → фільтри (тип, дата)
                         → LLM-rerank (gemini-3.1-flash-lite, оцінки 0–3)
  └─ src/rag/answer.py   поріг «не знаю» (max rerank < 2 → відмова без виклику моделі) →
                         контекст з бюджетом 12 000 симв. (тип, дата, стаття в заголовку джерела) →
                         відповідь за JSON-схемою {answer, citations, insufficient} → перевірка номерів цитат
  └─ src/rag/evals.py    тестсет → метрики retrieval / детерміновані / LLM-judge → results/rag/
prompts/rag/             rerank_v1.txt, answer_v1.txt, judge_v1.txt (версія в імені файлу)
data/rag_testset.jsonl   29 питань: закон, податки, пастки (18%→5%, суміжні теми), поза корпусом
.github/workflows/rag-evals.yml   evals у GitHub Actions з порогами метрик
```

## Запуск

```
python main.py rag "Яка ставка ПДФО з доходу від здачі квартири в оренду?"
python main.py rag-eval              # усі питання тестсету, ~10–15 хв на free tier
python main.py rag-eval --limit 5    # швидка перевірка
python main.py rag-eval --calibration  # згода LLM-judge з ручною розміткою (без API)
python main.py rag-build             # перезібрати корпус та індекс (не потрібно: data/processed/ у репозиторії)
```

Окремі кроки: `python -m src.rag.retrieve "питання" [--no-rerank] [--types law,dps] [--date-from 2026-01-01]`.

## Evals у CI

Workflow `rag-evals` запускається на push/PR, що змінюють `src/rag/`, `prompts/rag/`, тестсет або індекс,
і вручну (Actions → rag-evals → Run workflow). Потрібен секрет репозиторію `GEMINI_API_KEY`
(Settings → Secrets and variables → Actions). Пороги — `THRESHOLDS` у `src/rag/evals.py`;
якщо метрика нижча — job падає. Звіт `results/rag/` завантажується як artifact навіть при падінні.
