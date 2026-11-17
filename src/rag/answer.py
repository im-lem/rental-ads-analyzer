"""Шаг 3 RAG: поиск -> порог «не знаю» -> контекст с бюджетом -> ответ с цитатами [n].

Запуск:
  python -m src.rag.answer "Яка ставка ПДФО з доходу від здачі квартири в оренду?"
"""
import argparse
import re
from datetime import date
from pathlib import Path

from google.genai import types
from pydantic import BaseModel

from src.rag.retrieve import Retriever

ANSWER_MODEL = "gemini-3.1-flash-lite"
PROMPT_VERSION = "answer_v1"
PROMPT_PATH = Path(f"prompts/rag/{PROMPT_VERSION}.txt")
REFUSAL = "У наданих джерелах немає відповіді на це запитання."   # дословно как в промпте

MAX_CONTEXT_CHARS = 12000
MIN_SCORE = 2        # по rerank_v1: 1 = «та сама тема, але відповіді немає»
K = 8                # сколько чанков берём из поиска; реальный отбор делает бюджет

CITE_RE = re.compile(r"\[(\d+)\]")


class RagAnswer(BaseModel):
    answer: str            # текст с маркерами [1], [2]
    citations: list[int]   # те же номера списком
    insufficient: bool     # True = в источниках ответа нет


def build_context(hits: list[dict]) -> tuple[str, list[dict]]:
    blocks: list[str] = []
    sources: list[dict] = []
    used = 0
    n = 1

    for h in hits:
        date_ = h["doc_date"] or "невідома"
        art = f" | ст. {h['article']}" if h["article"] else ""
        header = f"[{n}] Тип: {h['doc_type']} | Дата: {date_} | «{h['title']}»{art}"
        block = f"{header}\n{h['text']}"

        sep_len = 2 if blocks else 0
        if used + sep_len + len(block) > MAX_CONTEXT_CHARS:
            continue

        blocks.append(block)
        used += sep_len + len(block)
        sources.append({
            "n": n,
            "chunk_id": h["chunk_id"],
            "doc_id": h["doc_id"],
            "title": h["title"],
            "article": h["article"],
            "doc_date": h["doc_date"],
            "url": h["url"],
        })
        n += 1

    return "\n\n".join(blocks), sources


def answer(question: str, retriever: Retriever, rerank: bool = True) -> dict:
    hits = retriever.search(question, k=K, rerank=rerank)
    result = {
        "question": question,
        "prompt_version": PROMPT_VERSION,
        "retrieved_chunk_ids": [h["chunk_id"] for h in hits],   # для recall@k без повторного поиска
        "max_llm_score": None,
        "answer": REFUSAL,
        "insufficient": True,
        "refusal_reason": None,
        "citations": [],
        "bad_citations": [],
    }

    # Уровень 1: порог по оценкам reranker. Если rerank упал/выключен, оценок нет (None) —
    # тогда порог не применяем и полагаемся на уровень 2 (правило в промпте).
    scores = [h["llm_score"] for h in hits]
    if scores and None not in scores:
        result["max_llm_score"] = max(scores)
    if not hits or (result["max_llm_score"] is not None and result["max_llm_score"] < MIN_SCORE):
        result["refusal_reason"] = "low_retrieval"
        return result

    # Уровень 2: модель отвечает по источникам (или сама говорит insufficient).
    context, sources = build_context(hits)
    prompt = PROMPT_PATH.read_text(encoding="utf-8").format(
        today=date.today().isoformat(), sources=context, question=question)
    resp = retriever.client.models.generate_content(
        model=ANSWER_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=RagAnswer,
        ),
    )
    parsed: RagAnswer | None = resp.parsed
    if parsed is None:
        raise RuntimeError(f"модель вернула не JSON по схеме: {resp.text!r:.200}")

    # Проверка цитат: номера из списка и из текста; всё, чего нет в sources, — выдуманное.
    by_n = {s["n"]: s for s in sources}
    cited = set(parsed.citations) | {int(m) for m in CITE_RE.findall(parsed.answer)}
    result.update({
        "answer": parsed.answer,
        "insufficient": parsed.insufficient,
        "refusal_reason": "model" if parsed.insufficient else None,
        "citations": [by_n[n] for n in sorted(cited) if n in by_n],
        "bad_citations": sorted(n for n in cited if n not in by_n),
    })
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--no-rerank", action="store_true")
    args = ap.parse_args()

    r = answer(args.question, Retriever(), rerank=not args.no_rerank)
    print(r["answer"], "\n")
    for s in r["citations"]:
        art = f", ст. {s['article']}" if s["article"] else ""
        print(f"[{s['n']}] {s['title']}{art} ({s['doc_date'] or 'без дати'})\n    {s['url']}")
    print(f"\ninsufficient={r['insufficient']}  reason={r['refusal_reason']}  "
          f"max_llm_score={r['max_llm_score']}  bad_citations={r['bad_citations']}")


if __name__ == "__main__":
    main()
