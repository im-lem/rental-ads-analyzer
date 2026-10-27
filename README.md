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
