"""Шаг 1 RAG: извлечение и очистка текста из data/raw/ -> data/processed/docs.jsonl.

Запуск:  python -m src.rag.ingest
"""
import csv
import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import pymupdf
import trafilatura

RAW_DIR = Path("data/raw")
MANIFEST = RAW_DIR / "manifest.csv"
OUT_PATH = Path("data/processed/docs.jsonl")

MIN_LEN_WARN = 500  # короче — подозрительно, проверить глазами


# ---------- извлечение ----------

def extract_html(path: Path) -> str:
    raw = path.read_bytes()  # bytes: trafilatura сама определит кодировку
    text = trafilatura.extract(raw, include_tables=True, include_comments=False)
    if not text:
        # основной текст не найден — пробуем «жадный» режим (больше текста, больше мусора)
        text = trafilatura.extract(raw, include_tables=True, include_comments=False,
                                   favor_recall=True)
    return text or ""


def extract_pdf(path: Path) -> str:
    with pymupdf.open(path) as doc:
        pages = [page_text(page) for page in doc]
    pages = remove_running_lines(pages)
    return "\n\n".join(pages)


def page_text(page, n_cols: int = 3) -> str:
    """Портретная страница: sort=True (сверху вниз, слева направо).
    Альбомная (буклет, сложенный втрое): sort=True смешал бы строки соседних колонок,
    поэтому сортируем блоки по (номер колонки по x0, y0)."""
    if page.rect.width <= page.rect.height:
        return page.get_text(sort=True)
    col_w = page.rect.width / n_cols
    blocks = [b for b in page.get_text("blocks") if b[4].strip()]  # (x0, y0, x1, y1, text, ...)
    blocks.sort(key=lambda b: (int(b[0] // col_w), b[1]))
    # внутри блока строки — это переносы верстки, а не абзацы: склеиваем пробелом
    return "\n\n".join(" ".join(b[4].split()) for b in blocks)


def remove_running_lines(pages: list[str]) -> list[str]:
    """Убирает колонтитулы: короткие строки, повторяющиеся на большинстве страниц."""
    if len(pages) < 3:
        return pages
    counts = Counter()
    for p in pages:
        counts.update({ln.strip() for ln in p.splitlines() if ln.strip()})
    threshold = len(pages) * 0.6
    running = {ln for ln, c in counts.items() if c >= threshold and len(ln) < 80}
    return ["\n".join(ln for ln in p.splitlines() if ln.strip() not in running)
            for p in pages]


# ---------- очистка ----------

def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ").replace("­", "")  # nbsp, мягкий перенос
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)            # перенос слова по слогам в PDF
    text = re.sub(r"[ \t]+", " ", text)                      # схлопнуть пробелы
    text = "\n".join(ln.strip() for ln in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text)                   # не больше одной пустой строки
    return text.strip()


# ---------- main ----------

EXTRACTORS = {"html": extract_html, "pdf": extract_pdf}


def build_docs() -> list[dict]:
    docs = []
    with MANIFEST.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            path = RAW_DIR / row["file"]
            text = clean_text(EXTRACTORS[row["format"]](path))
            docs.append({
                "doc_id": path.stem,
                "title": row["title"],
                "doc_type": row["doc_type"],
                "doc_date": row["doc_date"],
                "url": row["url"],
                "format": row["format"],
                "text": text,
            })
    remove_site_boilerplate(docs)
    return docs


def remove_site_boilerplate(docs: list[dict], share: float = 0.6, min_docs: int = 3) -> None:
    """Убирает строки, общие для многих страниц одного сайта (лента «читайте також»,
    подписи, меню), которые trafilatura приняла за основной текст. Меняет docs на месте."""
    by_site: dict[str, list[dict]] = {}
    for d in docs:
        if d["format"] == "html":
            by_site.setdefault(urlparse(d["url"]).netloc, []).append(d)
    for site_docs in by_site.values():
        if len(site_docs) < min_docs:
            continue
        counts = Counter()
        for d in site_docs:
            counts.update({ln for ln in d["text"].splitlines() if ln})
        shared = {ln for ln, c in counts.items() if c >= len(site_docs) * share}
        for d in site_docs:
            kept = [ln for ln in d["text"].splitlines() if ln not in shared]
            d["text"] = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def main() -> None:
    docs = build_docs()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"{'doc_id':<55} {'type':<8} {'chars':>7}")
    for d in sorted(docs, key=lambda d: len(d["text"])):
        flag = "  <-- проверить" if len(d["text"]) < MIN_LEN_WARN else ""
        print(f"{d['doc_id']:<55} {d['doc_type']:<8} {len(d['text']):>7}{flag}")
    print(f"\nИтого: {len(docs)} документов -> {OUT_PATH}")


if __name__ == "__main__":
    main()
