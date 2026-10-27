import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

import config
from extractors import GeminiExtractor, OllamaExtractor


# Запуск: python src\run.py <gemini|ollama> <t0|t07> <v1|v2> <all|golden>
provider, sampling_key, prompt_version, which = sys.argv[1:5]

load_dotenv()
params = config.SAMPLING[sampling_key]
model = config.MODELS[provider]

if provider == "gemini":
    ex = GeminiExtractor(os.environ["GEMINI_API_KEY"], model, **params,
                         timeout_s=config.TIMEOUT_S["gemini"])
else:
    ex = OllamaExtractor(model, **params, timeout_s=config.TIMEOUT_S["ollama"])


prompt = Path(f"prompts/{prompt_version}.txt").read_text(encoding="utf-8")

if which == "golden":
    ids = config.GOLDEN_IDS
else:
    ids = [p.stem for p in sorted(Path("data/ads").glob("*.txt"))]   # "001" ... "030"

out_path = Path(f"results/{provider}_{prompt_version}_{sampling_key}.jsonl")


def call_with_retry(prompt_text: str, ad: str, ad_id: str):
    """Retry только для ошибок СЕТИ/API (таймаут, 429, 503).
    Возвращает (result или None, число попыток, текст последней ошибки)."""
    last_error = None
    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            return ex.extract(prompt_text, ad), attempt, None
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            print(f"  {ad_id}: попытка {attempt} не удалась ({last_error[:80]})")
            time.sleep(10 * attempt)
    return None, config.MAX_RETRIES, last_error


def fix_prompt(base_prompt: str, bad_raw: str, error: str) -> str:
    """Промпт для попытки исправления: тот же промпт + прошлый ответ + ошибка валидации.
    Модель видит, ЧТО именно не так, и может это поправить."""
    return (base_prompt
            + "\n\nYour previous answer was INVALID.\n"
            + "Previous answer:\n" + bad_raw[:2000]          # обрезаем: зацикленный ответ может быть огромным
            + "\n\nValidation error:\n" + error[:1000]
            + "\n\nReturn a corrected JSON for the same ad.")


with open(out_path, "w", encoding="utf-8") as out:
    for ad_id in ids:
        ad = Path(f"data/ads/{ad_id}.txt").read_text(encoding="utf-8")

        result, attempts, last_error = call_with_retry(prompt, ad, ad_id)

        # Лог: модель, версия промпта, параметры семплинга — в каждой строке
        record = {
            "id": ad_id,
            "provider": provider,
            "model": model,
            "prompt_version": prompt_version,
            "sampling": sampling_key,
            **params,
            "attempts": attempts,
            "fix_attempted": False,
        }

        if result is None:
            # API так и не ответил — это не ошибка модели, а инфраструктуры
            record.update({"status": "api_error", "error": last_error})
        else:
            latency = result.latency_s
            in_tok, out_tok = result.input_tokens, result.output_tokens
            status = "ok"

            # Ответ пришёл, но не прошёл Pydantic → ОДНА попытка исправления
            if not result.valid:
                record["fix_attempted"] = True
                record["first_error"] = result.error
                record["first_raw"] = result.raw
                print(f"  {ad_id}: invalid → попытка исправления")
                fixed, _, fix_api_error = call_with_retry(
                    fix_prompt(prompt, result.raw, result.error), ad, ad_id)
                if fixed is not None:
                    # Время и токены считаем за ОБА вызова — это честная цена документа
                    latency += fixed.latency_s
                    in_tok += fixed.input_tokens
                    out_tok += fixed.output_tokens
                    result = fixed
                status = "fixed" if (fixed is not None and fixed.valid) else "failed"

            record.update({
                "status": status,                     # ok | fixed | failed | api_error
                "output": result.listing.model_dump(mode="json") if result.valid else None,
                "error": result.error,
                "raw": result.raw,
                "latency_s": round(latency, 2),
                "input_tokens": in_tok,
                "output_tokens": out_tok,
            })

        out.write(json.dumps(record, ensure_ascii=False) + "\n")
        out.flush()                                    # строка на диске сразу — прогон можно прервать
        print(f"{ad_id}: {record['status']}  {record.get('latency_s', '-')} с")

        time.sleep(config.PAUSE_S[provider])

print(f"Готово → {out_path}")