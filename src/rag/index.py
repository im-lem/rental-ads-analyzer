"""Шаг 2a RAG: эмбеддинги чанков -> data/processed/embeddings.npy (+ chunk_ids.json).

Векторная «база» — матрица numpy (168 x 768): при таком объёме отдельная БД не нужна,
поиск = одно матричное умножение. BM25-индекс не сохраняем — строится за миллисекунды.

Запуск:  python -m src.rag.index
"""
import json
import os
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
EMB_PATH = Path("data/processed/embeddings.npy")
IDS_PATH = Path("data/processed/chunk_ids.json")

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768      # Matryoshka: можно урезать 3072 -> 768 почти без потери качества
BATCH = 90          # Free tier: 100 эмбеддингов/мин, каждый текст в батче = 1 запрос
PAUSE_S = 61        # пауза между батчами, чтобы не упереться в минутный лимит
MAX_RETRIES = 3


def load_chunks() -> list[dict]:
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def embed_input(chunk: dict) -> str:
    # название документа даёт контекст чанку («ставка 5%» — о чём?). В chunks.jsonl текст чистый.
    return f"{chunk['title']}\n{chunk['text']}"


def get_client() -> genai.Client:
    load_dotenv()
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def embed(client: genai.Client, texts: list[str], task_type: str) -> np.ndarray:
    """task_type: RETRIEVAL_DOCUMENT для чанков, RETRIEVAL_QUERY для вопроса —
    модель кодирует их по-разному (вопрос короткий, документ длинный)."""
    cfg = types.EmbedContentConfig(task_type=task_type, output_dimensionality=EMBED_DIM)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.models.embed_content(model=EMBED_MODEL, contents=texts, config=cfg)
            break
        except Exception as e:  # 429 / сеть — пробуем ещё раз с паузой
            if attempt == MAX_RETRIES:
                raise
            print(f"  ошибка ({e.__class__.__name__}), повтор через {PAUSE_S} с")
            time.sleep(PAUSE_S)
    vecs = np.array([e.values for e in resp.embeddings], dtype=np.float32)
    # при размерности < 3072 векторы не нормированы -> нормируем, тогда cos = dot
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def main() -> None:
    chunks = load_chunks()
    client = get_client()
    parts = []
    for i in range(0, len(chunks), BATCH):
        if i:
            time.sleep(PAUSE_S)
        batch = chunks[i:i + BATCH]
        print(f"эмбеддинги {i + 1}–{i + len(batch)} из {len(chunks)}")
        parts.append(embed(client, [embed_input(c) for c in batch], "RETRIEVAL_DOCUMENT"))
    emb = np.vstack(parts)
    np.save(EMB_PATH, emb)
    IDS_PATH.write_text(json.dumps([c["chunk_id"] for c in chunks]), encoding="utf-8")
    print(f"Готово: {emb.shape} -> {EMB_PATH}")


if __name__ == "__main__":
    main()
