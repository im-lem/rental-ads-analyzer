"""Шаг 4 RAG: evals по data/rag_testset.jsonl.

  python -m src.rag.evals                 полный прогон: поиск + ответ + LLM-judge, метрики, отчёт
  python -m src.rag.evals --limit 5       первые 5 вопросов (быстрая проверка)
  python -m src.rag.evals --ci            то же + exit 1, если метрика ниже порога (для GitHub Actions)
  python -m src.rag.evals --calibration   только сверить judge с ручной разметкой (без API)

Три слоя метрик:
  1. retrieval (детерминированные): hit@k и MRR по документам для BM25 / вектора / гибрида / гибрид+rerank;
  2. ответ, детерминированные: отказ там, где надо; ключевые факты (regex); валидность цитат; цитата на эталонный документ;
  3. ответ, LLM-judge: correctness (0–2, против эталона) и faithfulness (0/1, против процитированных источников).
"""
import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

from google.genai import types
from pydantic import BaseModel

from src.rag.answer import answer
from src.rag.retrieve import Retriever

TESTSET = Path("data/rag_testset.jsonl")
OUT_DIR = Path("results/rag")
CALIB_CSV = OUT_DIR / "judge_calibration.csv"

# judge — другая модель, чем отвечающая gemini-3.1-flash-lite (меньше «самооценки»), со своей квотой.
# Почему не flash: на free tier у gemini-3.8-flash лимит 20 запросов/сутки на модель — полный прогон
# упал на 8-м вопросе; 3.5/3.6-flash отвечали 503, 2.5-flash закрыта для новых ключей (404).
# lite-judge слабее -> его согласие с человеком проверяем калибровкой (--calibration).
JUDGE_MODELS = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite-preview"]
JUDGE_THINKING = "LOW"   # для оценки по готовому эталону долгие «размышления» не нужны, а они медленные
JUDGE_PROMPT = Path("prompts/rag/judge_v1.txt")

PAUSE_S = 4          # пауза между вопросами: ~4 запроса к API на вопрос, держимся ниже минутных лимитов free tier
RETRY_PAUSE_S = 65   # при 429 ждём, пока обнулится минутное окно
MAX_TRIES = 3

# Пороги для CI. Заданы ДО первого прогона, из смысла метрики, а не подогнаны под результат.
THRESHOLDS = {
    "refusal_accuracy": 0.80,     # отказ/ответ угадан хотя бы в 4 из 5 вопросов
    "citation_validity": 0.95,    # выдуманный номер источника — почти недопустимо
    "must_match": 0.70,           # ключевой факт (число, срок) есть в ответе
    "hit@5_rerank": 0.80,         # эталонный документ в топ-5 после rerank
    "judge_correctness": 0.70,    # средняя оценка judge, нормированная в 0..1
    "faithfulness": 0.80,         # доля ответов без неподтверждённых утверждений
}

MODES = ["bm25", "vector", "hybrid", "rerank"]


class Judgment(BaseModel):
    reason: str          # первым: пусть модель сначала рассуждает, потом ставит оценку
    correctness: int     # 0..2
    faithfulness: int    # 0/1


# ---------- вспомогательное ----------

def with_retry(fn, what: str):
    for attempt in range(1, MAX_TRIES + 1):
        try:
            return fn()
        except Exception as e:   # 429 / 503 / сеть
            # суточная квота кончилась — ждать минуту бесполезно, падаем сразу (результаты уже на диске)
            if attempt == MAX_TRIES or "PerDay" in str(e):
                raise
            print(f"   {what}: {e.__class__.__name__}: {str(e)[:120]} -> повтор через {RETRY_PAUSE_S} с")
            time.sleep(RETRY_PAUSE_S)


def doc_ranking(chunk_ids: list[str] | None) -> list[str]:
    """Ранжирование чанков -> ранжирование документов (первое вхождение doc_id)."""
    seen, docs = set(), []
    for cid in chunk_ids or []:
        d = cid.split("#")[0]
        if d not in seen:
            seen.add(d)
            docs.append(d)
    return docs


def first_gold_rank(docs: list[str], gold: set[str]) -> int | None:
    for r, d in enumerate(docs, start=1):
        if d in gold:
            return r
    return None


def judge(client, item: dict, res: dict) -> tuple[Judgment, str]:
    sources = "\n\n".join(f"[{n}] {t}" for n, t in res["sources_text"].items()) or "(немає)"
    prompt = (JUDGE_PROMPT.read_text(encoding="utf-8")
              + f"\n\nЗАПИТАННЯ: {item['question']}\n\nЕТАЛОННА ВІДПОВІДЬ: {item['reference']}"
              + f"\n\nВІДПОВІДЬ СИСТЕМИ: {res['answer']}\n\nДЖЕРЕЛА:\n{sources}")
    last_err = None
    for model in JUDGE_MODELS:
        try:
            resp = client.models.generate_content(
                model=model, contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0, response_mime_type="application/json", response_schema=Judgment,
                    thinking_config=types.ThinkingConfig(thinking_level=JUDGE_THINKING)))
        except Exception as e:      # 503/429 у этой модели — пробуем следующую
            last_err = e
            continue
        if resp.parsed is not None:
            return resp.parsed, model
        last_err = RuntimeError(f"{model}: judge вернул не JSON по схеме")
    raise last_err


# ---------- прогон одного вопроса ----------

def run_item(r: Retriever, item: dict) -> dict:
    def pipeline():
        res = answer(item["question"], r)
        if res["retrieval"].get("rerank") is None:     # rerank упал (обычно 429) — метрики были бы нечестными
            raise RuntimeError("rerank не отработал")
        return res
    res = with_retry(pipeline, item["id"])

    gold = set(item["gold_docs"])
    row = {"id": item["id"], "category": item["category"], "question": item["question"],
           "reference": item["reference"], "expect_refusal": item["expect_refusal"],
           "answer": res["answer"], "insufficient": res["insufficient"],
           "refusal_reason": res["refusal_reason"], "max_llm_score": res["max_llm_score"],
           "citations": [s["chunk_id"] for s in res["citations"]],
           "bad_citations": res["bad_citations"]}

    # 1. retrieval
    if gold:
        for m in MODES:
            row[f"rank_{m}"] = first_gold_rank(doc_ranking(res["retrieval"].get(m)), gold)

    # 2. детерминированные проверки ответа
    answered = not res["insufficient"]
    row["refusal_ok"] = res["insufficient"] == item["expect_refusal"]
    if answered:
        row["citation_valid"] = not res["bad_citations"] and len(res["citations"]) > 0
    if not item["expect_refusal"]:
        text = res["answer"].lower()
        row["must_match"] = answered and all(re.search(p, text, re.I) for p in item["must_match"])
        row["cited_gold"] = answered and any(s["doc_id"] in gold for s in res["citations"])

    # 3. LLM-judge — только для вопросов с ответом и только если система ответила
    if not item["expect_refusal"]:
        if answered:
            j, model = with_retry(lambda: judge(r.client, item, res), item["id"] + " judge")
            row.update(judge_correctness=j.correctness, judge_faithfulness=j.faithfulness,
                       judge_reason=j.reason, judge_model=model)
        else:
            row.update(judge_correctness=0, judge_faithfulness=None,
                       judge_reason="система відмовилась, хоча відповідь є")
    return row


# ---------- метрики ----------

def mean(xs) -> float | None:
    xs = [float(x) for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def summarize(rows: list[dict]) -> dict:
    s: dict = {"n_questions": len(rows)}
    ret = [x for x in rows if "rank_bm25" in x]
    for m in MODES:
        ranks = [x[f"rank_{m}"] for x in ret]
        s[f"hit@1_{m}"] = mean([rk == 1 for rk in ranks])
        s[f"hit@5_{m}"] = mean([rk is not None and rk <= 5 for rk in ranks])
        s[f"mrr_{m}"] = mean([1 / rk if rk else 0 for rk in ranks])
    s["refusal_accuracy"] = mean([x["refusal_ok"] for x in rows])
    s["citation_validity"] = mean([x.get("citation_valid") for x in rows])
    s["must_match"] = mean([x.get("must_match") for x in rows])
    s["cited_gold"] = mean([x.get("cited_gold") for x in rows])
    jc = [x.get("judge_correctness") for x in rows if "judge_correctness" in x]
    s["judge_correctness"] = round(mean(jc) / 2, 3) if jc else None
    s["faithfulness"] = mean([x.get("judge_faithfulness") for x in rows])
    return s


def report_md(s: dict, rows: list[dict]) -> str:
    lines = ["# RAG evals", "", f"Вопросов: {s['n_questions']}", "",
             "## Retrieval (по документам)", "", "| режим | hit@1 | hit@5 | MRR |", "|---|---|---|---|"]
    for m in MODES:
        lines.append(f"| {m} | {s[f'hit@1_{m}']} | {s[f'hit@5_{m}']} | {s[f'mrr_{m}']} |")
    lines += ["", "## Ответ", "", "| метрика | значение | порог |", "|---|---|---|"]
    for k in ["refusal_accuracy", "citation_validity", "must_match", "cited_gold",
              "judge_correctness", "faithfulness"]:
        lines.append(f"| {k} | {s[k]} | {THRESHOLDS.get(k, '—')} |")
    lines += ["", "## Провалы", ""]
    for x in rows:
        bad = [k for k in ("refusal_ok", "must_match", "citation_valid", "cited_gold") if x.get(k) is False]
        if x.get("judge_correctness") is not None and x["judge_correctness"] < 2:
            bad.append(f"judge={x['judge_correctness']}")
        if x.get("judge_faithfulness") == 0:
            bad.append("unfaithful")
        if bad:
            lines.append(f"- **{x['id']}** ({', '.join(bad)}): {x['answer'][:200]}")
    return "\n".join(lines) + "\n"


# ---------- калибровка judge ----------

CALIB_FIELDS = ["id", "question", "reference", "answer", "judge_correctness", "judge_faithfulness",
                "judge_reason", "human_correctness", "human_faithfulness"]


def write_calibration(rows: list[dict]) -> None:
    """Таблица для ручной разметки. Уже проставленные человеком оценки сохраняем."""
    old = {}
    if CALIB_CSV.exists():
        with CALIB_CSV.open(encoding="utf-8-sig") as f:
            old = {r["id"]: r for r in csv.DictReader(f)}
    with CALIB_CSV.open("w", encoding="utf-8-sig", newline="") as f:   # -sig: Excel откроет кириллицу
        w = csv.DictWriter(f, fieldnames=CALIB_FIELDS)
        w.writeheader()
        for x in rows:
            if x.get("judge_faithfulness") is None:      # judge не вызывался — калибровать нечего
                continue
            prev = old.get(x["id"], {})
            w.writerow({**{k: x.get(k, "") for k in CALIB_FIELDS[:7]},
                        "human_correctness": prev.get("human_correctness", ""),
                        "human_faithfulness": prev.get("human_faithfulness", "")})


def calibration_report() -> None:
    if not CALIB_CSV.exists():
        print("нет", CALIB_CSV, "— сначала полный прогон")
        return
    with CALIB_CSV.open(encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r["human_correctness"].strip()]
    if not rows:
        print(f"в {CALIB_CSV} не заполнены human_correctness / human_faithfulness")
        return
    for field in ("correctness", "faithfulness"):
        pairs = [(r[f"judge_{field}"], r[f"human_{field}"].strip()) for r in rows
                 if r[f"human_{field}"].strip()]
        agree = sum(a == b for a, b in pairs)
        print(f"{field}: согласие judge и человека {agree}/{len(pairs)} = {agree / len(pairs):.0%}")
        for (a, b), r in zip(pairs, rows):
            if a != b:
                print(f"   {r['id']}: judge={a} human={b}  ({r['judge_reason'][:120]})")


# ---------- main ----------

def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--ids", help="через запятую")
    ap.add_argument("--ci", action="store_true")
    ap.add_argument("--calibration", action="store_true")
    ap.add_argument("--resume", action="store_true", help="продолжить, пропуская готовые вопросы")
    args = ap.parse_args()

    if args.calibration:
        calibration_report()
        return

    items = [json.loads(l) for l in TESTSET.open(encoding="utf-8") if l.strip()]
    if args.ids:
        items = [it for it in items if it["id"] in set(args.ids.split(","))]
    if args.limit:
        items = items[:args.limit]
    n_reviewed = sum(it.get("reviewed", False) for it in items)
    print(f"вопросов: {len(items)} (проверено человеком: {n_reviewed})")

    # Каждый вопрос сразу дописываем в файл: падение на 25-м вопросе не стирает 24 готовых,
    # а --resume продолжает с места падения (не тратя квоту повторно).
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUT_DIR / "eval_results.jsonl"
    done: dict[str, dict] = {}
    if args.resume and results_path.exists():
        done = {x["id"]: x for x in map(json.loads, results_path.open(encoding="utf-8"))}
        print(f"--resume: уже готово {len(done)}, пропускаем их")
    elif results_path.exists():
        results_path.unlink()

    r = Retriever()
    aborted = None
    for n, item in enumerate(items, 1):
        if item["id"] in done:
            continue
        try:
            row = run_item(r, item)
        except Exception as e:
            aborted = f"{item['id']}: {e.__class__.__name__}: {str(e)[:300]}"
            print(f"[{n}/{len(items)}] ПРЕРВАНО — {aborted}\n"
                  f"   готовые результаты сохранены; продолжить: python -m src.rag.evals --resume")
            break
        done[item["id"]] = row
        with results_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        flags = " ".join(f"{k}={row[k]}" for k in ("refusal_ok", "must_match", "rank_rerank",
                                                   "judge_correctness", "judge_faithfulness") if k in row)
        print(f"[{n}/{len(items)}] {item['id']}: {flags}", flush=True)
        time.sleep(PAUSE_S)

    ids = [it["id"] for it in items]
    rows = [done[i] for i in ids if i in done]      # порядок тестсета, только выбранные вопросы
    if not rows:
        sys.exit(f"нет ни одного результата ({aborted})")
    s = summarize(rows)
    s["complete"] = aborted is None and len(rows) == len(items)
    (OUT_DIR / "summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "summary.md").write_text(report_md(s, rows), encoding="utf-8")
    write_calibration(rows)
    print("\n" + report_md(s, rows))

    if aborted:
        print(f"ВНИМАНИЕ: прогон неполный ({len(rows)}/{len(items)}), метрики выше — по готовым вопросам")
    if args.ci:
        failed = [f"{k}={s.get(k)} < {t}" for k, t in THRESHOLDS.items()
                  if s.get(k) is None or s[k] < t]
        if aborted:
            failed.append(f"прогон прерван: {aborted}")
        if failed:
            print("CI FAIL:", "; ".join(failed))
            sys.exit(1)
        print("CI OK: все метрики выше порогов")


if __name__ == "__main__":
    main()
