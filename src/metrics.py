"""Метрики по всем прогонам из results/*.jsonl.

Запуск из корня проекта:  python src\\metrics.py
Пишет таблицы в results/metrics.md и печатает их в консоль.
"""
import json
import statistics
from pathlib import Path

import config

RESULTS = Path("results")

# Поля, которые сравниваем. "Текст" — значение прямо написано в объявлении,
# "вывод" — модель должна сделать вывод (тип жилья, тип арендодателя, язык).
TEXT_FIELDS = ["rooms", "area_m2", "price.amount", "price.currency", "price.utilities",
               "location.city", "location.district", "pets_allowed", "amenities"]
INFER_FIELDS = ["property_type", "landlord_type", "text_language"]
ALL_FIELDS = TEXT_FIELDS + INFER_FIELDS            # 12 полей

# Нормализация названий: модель могла написать то же самое иначе.
# Ошибку формы (Kyiv вместо Київ) мы считаем отдельно от ошибки смысла (не тот город).
TRANSLIT = {
    "kyiv": "київ", "kiev": "київ", "киев": "київ",
    "kharkiv": "харків", "харьков": "харків",
    "rivne": "рівне", "boyarka": "боярка",
    "shevchenkivskyi": "шевченківський", "shevchenkivsky": "шевченківський",
    "dniprovskyi": "дніпровський", "dniprovsky": "дніпровський",
    "center": "центр", "centre": "центр",
}
DROP_WORDS = {"район", "р-н", "district", "м.", "місто", "city"}


def norm_place(value):
    """'Shevchenkivsky District' → 'шевченківський', 'unknown' → None."""
    if value is None:
        return None
    words = [w for w in str(value).lower().replace(",", " ").split() if w not in DROP_WORDS]
    text = " ".join(words).strip()
    if text in ("", "unknown", "null", "none"):
        return None
    return TRANSLIT.get(text, text)


def get(d, path):
    """get(listing, 'price.amount') → значение вложенного поля."""
    for key in path.split("."):
        d = d.get(key) if isinstance(d, dict) else None
    return d


def field_equal(field, got, exp, normalize):
    if field == "amenities":                          # список → множество: порядок и повторы не важны
        return set(got or []) == set(exp or [])
    if field.startswith("location.") and normalize:
        return norm_place(got) == norm_place(exp)
    if isinstance(got, (int, float)) and isinstance(exp, (int, float)):
        return float(got) == float(exp)               # 46 и 46.0 — одно и то же
    return got == exp


def load_golden():
    with open("data/golden.jsonl", encoding="utf-8") as f:
        return {r["id"]: r["expected"] for r in map(json.loads, f)}


def percentile(values, p):
    """Перцентиль с линейной интерполяцией (как numpy по умолчанию)."""
    values = sorted(values)
    if not values:
        return None
    k = (len(values) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def analyze(path, golden):
    rows = [json.loads(l) for l in path.open(encoding="utf-8")]
    provider, prompt_v, sampling = path.stem.split("_")
    n = len(rows)

    valid = [r for r in rows if r["status"] in ("ok", "fixed")]
    latencies = [r["latency_s"] for r in rows if r.get("latency_s") is not None]

    # Стоимость: средние токены на документ × цена × 1000 документов
    price = config.PRICES[provider]
    tok_rows = [r for r in rows if r.get("input_tokens") is not None]
    avg_in = statistics.mean(r["input_tokens"] for r in tok_rows) if tok_rows else 0
    avg_out = statistics.mean(r["output_tokens"] for r in tok_rows) if tok_rows else 0
    cost_1000 = (avg_in * price["input"] + avg_out * price["output"]) / 1_000_000 * 1000

    # Точность — только на golden set. Невалидный ответ = все поля неверны.
    gold_rows = [r for r in rows if r["id"] in golden]
    per_field = {f: 0 for f in ALL_FIELDS}
    raw_correct = 0                                   # без нормализации названий
    errors = []                                       # для разбора ошибок в REPORT
    am_tp = am_fp = am_fn = 0                         # amenities: попал / выдумал / пропустил
    for r in gold_rows:
        exp, got = golden[r["id"]], r.get("output") or {}
        for f in ALL_FIELDS:
            ok = bool(got) and field_equal(f, get(got, f), get(exp, f), normalize=True)
            per_field[f] += ok
            raw_correct += bool(got) and field_equal(f, get(got, f), get(exp, f), normalize=False)
            if not ok:
                errors.append((r["id"], f, get(got, f) if got else r["status"], get(exp, f)))
        g, e = set(got.get("amenities") or []), set(exp["amenities"])
        am_tp, am_fp, am_fn = am_tp + len(g & e), am_fp + len(g - e), am_fn + len(e - g)

    ng = len(gold_rows) or 1

    def acc(fields):
        return sum(per_field[f] for f in fields) / (ng * len(fields))

    return {
        "name": path.stem, "provider": provider, "model": rows[0]["model"],
        "prompt": prompt_v, "sampling": sampling,
        "temperature": rows[0]["temperature"], "top_p": rows[0]["top_p"],
        "n": n, "n_golden": len(gold_rows),
        "valid_pct": 100 * len(valid) / n,
        "fixed": sum(r["status"] == "fixed" for r in rows),
        "failed": sum(r["status"] in ("failed", "invalid") for r in rows),
        "p50": percentile(latencies, 50), "p95": percentile(latencies, 95),
        "cost_1000": cost_1000,
        "acc_all": acc(ALL_FIELDS), "acc_text": acc(TEXT_FIELDS), "acc_infer": acc(INFER_FIELDS),
        "acc_raw": raw_correct / (ng * len(ALL_FIELDS)),
        "per_field": {f: per_field[f] / ng for f in ALL_FIELDS},
        "am_precision": am_tp / (am_tp + am_fp) if am_tp + am_fp else 0,
        "am_recall": am_tp / (am_tp + am_fn) if am_tp + am_fn else 0,
        "errors": errors,
    }


def pct(x):
    return f"{100 * x:.0f}%"


def main():
    golden = load_golden()
    stats = [analyze(p, golden) for p in sorted(RESULTS.glob("*_*_*.jsonl"))]
    lines = []

    lines += ["## Модель × промпт × семплінг", "",
              "| Модель | Промпт | Семплінг (T / top_p) | Док. | Валідних | Виправлено / failed "
              "| p50, с | p95, с | $ / 1000 док. | Точність (golden) | Текст | Висновок | Без нормалізації |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in stats:
        lines.append(
            f"| {s['model']} | {s['prompt']} | {s['sampling']} ({s['temperature']} / {s['top_p']}) "
            f"| {s['n']} | {s['valid_pct']:.0f}% | {s['fixed']} / {s['failed']} "
            f"| {s['p50']:.1f} | {s['p95']:.1f} | ${s['cost_1000']:.2f} "
            f"| **{pct(s['acc_all'])}** | {pct(s['acc_text'])} | {pct(s['acc_infer'])} | {pct(s['acc_raw'])} |")

    lines += ["", "## Точність по полях (golden set, 10 оголошень)", "",
              "| Поле | " + " | ".join(s["name"] for s in stats) + " |",
              "|---|" + "---|" * len(stats)]
    for f in ALL_FIELDS:
        lines.append(f"| {f} | " + " | ".join(pct(s["per_field"][f]) for s in stats) + " |")
    lines.append("| amenities precision / recall | "
                 + " | ".join(f"{pct(s['am_precision'])} / {pct(s['am_recall'])}" for s in stats) + " |")

    lines += ["", "## Помилки на golden set (id · поле · отримано → очікувалось)", ""]
    for s in stats:
        lines += [f"### {s['name']}", ""]
        lines += [f"- {i} · {f} · `{got}` → `{exp}`" for i, f, got, exp in s["errors"]] or ["- немає"]
        lines.append("")

    text = "\n".join(lines)
    (RESULTS / "metrics.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
