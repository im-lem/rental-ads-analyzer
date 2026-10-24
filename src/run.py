import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

import config
from extractors import GeminiExtractor, OllamaExtractor


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

with open(out_path, "w", encoding="utf-8") as out:
    for ad_id in ids:
        ad = Path(f"data/ads/{ad_id}.txt").read_text(encoding="utf-8")

        # Retry: при ошибке сети/таймауте/429 ждём и пробуем снова
        result, last_error = None, None
        for attempt in range(1, config.MAX_RETRIES + 1):
            try:
                result = ex.extract(prompt, ad)
                break                                  
            except Exception as e:                     
                last_error = f"{type(e).__name__}: {e}"
                print(f"  {ad_id}: попытка {attempt} не удалась ({last_error[:80]})")
                time.sleep(10 * attempt)               

        
        record = {
            "id": ad_id,
            "provider": provider,
            "model": model,
            "prompt_version": prompt_version,
            "sampling": sampling_key,
            **params,
            "attempts": attempt,
        }
        if result is None:
            record.update({"status": "api_error", "error": last_error})
        else:
            record.update({
                "status": "ok" if result.valid else "invalid",
                "output": result.listing.model_dump(mode="json") if result.valid else None,
                "error": result.error,
                "raw": result.raw,
                "latency_s": round(result.latency_s, 2),
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
            })

        out.write(json.dumps(record, ensure_ascii=False) + "\n")
        out.flush()                                    
        print(f"{ad_id}: {record['status']}  {record.get('latency_s', '-')} с")

        time.sleep(config.PAUSE_S[provider])

print(f"Готово → {out_path}")