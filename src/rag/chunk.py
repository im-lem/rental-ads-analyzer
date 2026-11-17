"""Шаг 1b RAG: нарезка документов на чанки -> data/processed/chunks.jsonl.

Закон (doc_type=law): 1 статья = 1 чанк (граница — строка «Стаття NNN»).
Остальное: жадно набираем строки/абзацы до MAX_CHARS, перекрытие — 1 единица.

Запуск:  python -m src.rag.chunk
"""
import json
import re
from pathlib import Path

DOCS_PATH = Path("data/processed/docs.jsonl")
OUT_PATH = Path("data/processed/chunks.jsonl")

MAX_CHARS = 1000   # целевой размер чанка
MIN_CHARS = 200    # хвост короче — приклеиваем к предыдущему чанку

ARTICLE_RE = re.compile(r"^Стаття\s+(\d+)", re.MULTILINE)
SENTENCE_RE = re.compile(r"(?<=[.!?;])\s+")


# ---------- закон: по статьям ----------

def split_law(text: str) -> list[dict]:
    matches = list(ARTICLE_RE.finditer(text))
    if not matches:
        return [{"text": t} for t in split_paragraphs(text)]
    chunks = []
    preamble = text[:matches[0].start()].strip()  # заголовок главы — приклеим к 1-й статье
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.start():end].strip()
        if i == 0 and preamble:
            body = preamble + "\n" + body
        chunks.append({"text": body, "article": m.group(1)})
    return chunks


# ---------- остальное: по абзацам ----------

def to_units(text: str) -> list[str]:
    """Единица = непустая строка (HTML: абзац = строка; PDF: пустые строки просто отбрасываем).
    Слишком длинную строку режем по предложениям."""
    units = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if len(line) <= MAX_CHARS:
            units.append(line)
        else:
            units.extend(pack(SENTENCE_RE.split(line), overlap=False, sep=" "))
    return units


def pack(units: list[str], overlap: bool = True, sep: str = "\n") -> list[str]:
    """Жадно набирает единицы в чанки до MAX_CHARS. overlap: новый чанк начинается
    с последней единицы предыдущего, чтобы мысль на границе не терялась."""
    chunks, cur, n_new = [], [], 0  # n_new — сколько единиц в cur ещё не попали в прошлый чанк
    for u in units:
        if cur and len(sep.join(cur + [u])) > MAX_CHARS:
            if n_new:
                chunks.append(sep.join(cur))
            # перекрытие берём, только если вместе с новой единицей влезаем в лимит
            keep = overlap and len(cur) > 1 and len(cur[-1]) + len(sep) + len(u) <= MAX_CHARS
            cur = [cur[-1]] if keep else []
            n_new = 0
        cur.append(u)
        n_new += 1
    if cur and n_new:
        tail = sep.join(cur)
        if chunks and len(tail) < MIN_CHARS:
            chunks[-1] += sep + sep.join(cur[-n_new:])  # не плодим крошечные чанки
        else:
            chunks.append(tail)
    return chunks


def split_paragraphs(text: str) -> list[str]:
    return pack(to_units(text))


# ---------- main ----------

META_KEYS = ("doc_id", "title", "doc_type", "doc_date", "url", "format")


def build_chunks(docs: list[dict]) -> list[dict]:
    out = []
    for d in docs:
        if d["doc_type"] == "law":
            parts = split_law(d["text"])
        else:
            parts = [{"text": t} for t in split_paragraphs(d["text"])]
        for i, p in enumerate(parts):
            out.append({
                "chunk_id": f"{d['doc_id']}#{i}",
                **{k: d[k] for k in META_KEYS},
                "chunk_index": i,
                "article": p.get("article"),
                "text": p["text"],
            })
    return out


def main() -> None:
    with DOCS_PATH.open(encoding="utf-8") as f:
        docs = [json.loads(line) for line in f]
    chunks = build_chunks(docs)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    lens = sorted(len(c["text"]) for c in chunks)
    print(f"Документов: {len(docs)}, чанков: {len(chunks)} -> {OUT_PATH}")
    print(f"Длина чанка: min {lens[0]}, медиана {lens[len(lens) // 2]}, max {lens[-1]}")


if __name__ == "__main__":
    main()
