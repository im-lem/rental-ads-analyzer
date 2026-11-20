"""Сравнение нашей chunking-стратегии с фиксированным размером (как «в туториале»).

Наша: закон — 1 статья ЦК = 1 чанк; остальное — абзацы/строки, набранные до 1000 симв., overlap 1 абзац.
Fixed: окно 1000 символов, перекрытие 100, режем где придётся.

Без API (только BM25), поэтому запускается где угодно:
  python -m src.rag.chunk_compare      -> results/rag/chunking_compare.md
"""
import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.rag.chunk import ARTICLE_RE, build_chunks
from src.rag.evals import LAW_DOC, gold_chunks
from src.rag.retrieve import tokenize

DOCS_PATH = Path("data/processed/docs.jsonl")
TESTSET = Path("data/rag_testset.jsonl")
OUT = Path("results/rag/chunking_compare.md")

FIXED_SIZE, FIXED_OVERLAP = 1000, 100
# Документы разных форм: закон, статья-сайт, роз'яснення ДПС (HTML), буклет ДПС (PDF), гайд cedos
SAMPLE = [LAW_DOC, "art_7eminar_novi_stavki_pdfo", "dps_rv_orenda_subyektu_hospodaryuvannya",
          "dps_cv_pamyatka_orenda", "art_cedos_shho_potribno_dlya_togo_shhoby_zdavaty_zhytlo_yaki"]
SENT_END = re.compile(r"[.!?:;»)]\s*$")


def fixed_chunks(doc: dict) -> list[dict]:
    text, out, start = doc["text"], [], 0
    while start < len(text):
        end = min(start + FIXED_SIZE, len(text))
        out.append({"chunk_id": f"{doc['doc_id']}#f{len(out)}", "doc_id": doc["doc_id"],
                    "title": doc["title"], "text": text[start:end], "start": start, "end": end})
        if end == len(text):
            break
        start = end - FIXED_OVERLAP
    return out


def cut_stats(chunks: list[dict], doc_text: str) -> dict:
    """mid_word: граница режет слово (буква слева и справа от границы). У нашей стратегии границы —
    концы строк/абзацев, поэтому 0 по построению; для fixed считаем по позиции в тексте.
    mid_sent: чанк кончается не знаком конца предложения (в PDF-списках пункты без точки — это не обрыв)."""
    n = len(chunks)
    mid_word = sum(1 for c in chunks if "end" in c and c["end"] < len(doc_text)
                   and doc_text[c["end"] - 1].isalnum() and doc_text[c["end"]].isalnum())
    mid_sent = sum(1 for c in chunks if not SENT_END.search(c["text"].rstrip()))
    lens = sorted(len(c["text"]) for c in chunks)
    return {"n": n, "median": lens[n // 2], "mid_word": mid_word / n, "mid_sent": mid_sent / n}


def law_stats(law: dict, fixed: list[dict]) -> dict:
    """Сколько статей ЦК разрезано между чанками и сколько чанков смешивают ≥2 статьи."""
    m = list(ARTICLE_RE.finditer(law["text"]))
    spans = [(x.group(1), x.start(), m[i + 1].start() if i + 1 < len(m) else len(law["text"]))
             for i, x in enumerate(m)]
    split = sum(1 for _, s, e in spans
                if not any(c["start"] <= s and e <= c["end"] for c in fixed))
    mixed = sum(1 for c in fixed
                if sum(1 for _, s, e in spans if s < c["end"] and e > c["start"]) >= 2)
    return {"articles": len(spans), "split": split, "mixed": mixed, "spans": spans}


def bm25_rank(chunks: list[dict], query: str) -> list[str]:
    bm = BM25Okapi([tokenize(c["title"] + "\n" + c["text"]) for c in chunks])
    scores = bm.get_scores(tokenize(query))
    return [chunks[i]["chunk_id"] for i in sorted(range(len(chunks)), key=lambda i: -scores[i])]


def main() -> None:
    docs = [json.loads(l) for l in DOCS_PATH.open(encoding="utf-8")]
    by_id = {d["doc_id"]: d for d in docs}
    ours_all = build_chunks(docs)
    fixed_all = [c for d in docs for c in fixed_chunks(d)]

    lines = ["# Chunking: наша стратегія vs фіксований розмір", "",
             f"Fixed = вікно {FIXED_SIZE} симв., overlap {FIXED_OVERLAP}. «Посеред слова» — межа ріже слово; "
             "«без кінця речення» — чанк не закінчується `.!?:;` (у PDF-пам'ятці пункти списку без крапки, "
             "тому там і наша стратегія дає високий %).", "",
             "| документ | чанків (наша / fixed) | медіана симв. | ріже слово | без кінця речення |",
             "|---|---|---|---|---|"]
    for did in SAMPLE:
        o = cut_stats([c for c in ours_all if c["doc_id"] == did], by_id[did]["text"])
        f = cut_stats([c for c in fixed_all if c["doc_id"] == did], by_id[did]["text"])
        lines.append(f"| `{did}` | {o['n']} / {f['n']} | {o['median']} / {f['median']} | "
                     f"{o['mid_word']:.0%} / {f['mid_word']:.0%} | {o['mid_sent']:.0%} / {f['mid_sent']:.0%} |")

    law = by_id[LAW_DOC]
    ls = law_stats(law, [c for c in fixed_all if c["doc_id"] == LAW_DOC])
    lines += ["", "## Закон (глава 59 ЦК)", "",
              f"- статей: {ls['articles']}; у нашій стратегії кожна = рівно 1 чанк з полем `article` "
              "(для цитати «ст. 820 ЦК»);",
              f"- fixed: **{ls['split']}** статей розрізано між чанками, **{ls['mixed']}** чанків змішують ≥2 статті "
              "(номер статті для цитати невизначений)."]

    # Пошук правильної статті (BM25 по всьому корпусу, ті самі 12 питань, що й chunk-метрика в evals).
    # Для fixed «влучання» — будь-який чанк, що перетинається зі статтею (поблажливо до fixed).
    items = [json.loads(l) for l in TESTSET.open(encoding="utf-8") if l.strip()]
    spans = {a: (s, e) for a, s, e in ls["spans"]}
    fixed_law = {c["chunk_id"]: c for c in fixed_all if c["doc_id"] == LAW_DOC}
    res = {"ours": [], "fixed": []}
    for it in items:
        g_ours = gold_chunks(it, ours_all)
        if not g_ours:
            continue
        arts = {c["article"] for c in ours_all if c["chunk_id"] in g_ours}
        g_fixed = {cid for cid, c in fixed_law.items()
                   if any(c["start"] < spans[a][1] and c["end"] > spans[a][0] for a in arts)}
        for name, chunks, gold in (("ours", ours_all, g_ours), ("fixed", fixed_all, g_fixed)):
            rank = next((r for r, cid in enumerate(bm25_rank(chunks, it["question"]), 1) if cid in gold), None)
            res[name].append(rank)
    lines += ["", f"## Пошук потрібної статті ЦК (BM25, {len(res['ours'])} питань з еталонною статтею)", "",
              "| стратегія | hit@1 | hit@3 | MRR |", "|---|---|---|---|"]
    for name, ranks in res.items():
        n = len(ranks)
        lines.append(f"| {name} | {sum(r == 1 for r in ranks) / n:.2f} | "
                     f"{sum(r is not None and r <= 3 for r in ranks) / n:.2f} | "
                     f"{sum(1 / r for r in ranks if r) / n:.3f} |")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
