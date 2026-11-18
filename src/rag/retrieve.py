"""Шаг 2b RAG: hybrid search (BM25 + вектор) -> RRF -> фильтры -> LLM-reranking.

Запуск (проверка руками):
  python -m src.rag.retrieve "Яка ставка ПДФО з доходу від оренди квартири?"
  python -m src.rag.retrieve "..." --k 5 --no-rerank --types law,dps --date-from 2025-01-01
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
from google.genai import types
from pydantic import BaseModel
from rank_bm25 import BM25Okapi

from src.rag.index import EMB_PATH, IDS_PATH, embed, get_client, load_chunks

RERANK_MODEL = "gemini-3.1-flash-lite"
RERANK_PROMPT = Path("prompts/rag/rerank_v1.txt")

N_CANDIDATES = 20   # сколько берём из каждого поиска до слияния
N_RERANK = 10       # сколько лучших после RRF отдаём LLM на переоценку
RRF_K = 60          # стандартная константа RRF: сглаживает разницу между 1-м и 5-м местом
STEM_LEN = 5        # «оренда/оренди/орендою», «знижка/знижку» -> «оренд», «знижк»: дешёвая замена стеммера


# ---------- BM25 ----------

TOKEN_RE = re.compile(r"[\w'’ʼ]+")


def tokenize(text: str) -> list[str]:
    tokens = []
    for t in TOKEN_RE.findall(text.lower()):
        t = re.sub(r"['’ʼ]", "", t)
        if len(t) < 2 and not t.isdigit():   # і, в, з, у — шум; но «5» (процент) оставляем
            continue
        tokens.append(t if t.isdigit() else t[:STEM_LEN])
    return tokens


# ---------- RRF ----------

def rrf(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    """Reciprocal Rank Fusion: score = Σ 1/(k + rank). Складываем места, а не сырые
    оценки, потому что BM25 (0..∞) и косинус (-1..1) в разных шкалах."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return scores


# ---------- reranking ----------

class RerankItem(BaseModel):
    id: int
    score: int


class Retriever:
    def __init__(self):
        self.chunks = load_chunks()
        ids = json.loads(IDS_PATH.read_text(encoding="utf-8"))
        if ids != [c["chunk_id"] for c in self.chunks]:
            raise RuntimeError("chunks.jsonl и embeddings.npy рассинхронизированы: "
                               "перезапусти python -m src.rag.index")
        self.emb = np.load(EMB_PATH)
        self.bm25 = BM25Okapi([tokenize(c["title"] + "\n" + c["text"]) for c in self.chunks])
        self.client = get_client()

    def _mask(self, doc_types: set[str] | None, date_from: str | None) -> np.ndarray:
        """Фильтр по метаданным. Документы без даты при date_from не отбрасываем (законы)."""
        keep = np.ones(len(self.chunks), dtype=bool)
        for i, c in enumerate(self.chunks):
            if doc_types and c["doc_type"] not in doc_types:
                keep[i] = False
            if date_from and c["doc_date"] and c["doc_date"] < date_from:
                keep[i] = False
        return keep

    def bm25_rank(self, query: str, mask: np.ndarray) -> list[int]:
        scores = np.where(mask, self.bm25.get_scores(tokenize(query)), -np.inf)
        order = np.argsort(-scores)[:N_CANDIDATES]
        return [int(i) for i in order if scores[i] > 0]   # 0 = ни одного общего слова

    def vector_rank(self, query: str, mask: np.ndarray) -> list[int]:
        q = embed(self.client, [query], "RETRIEVAL_QUERY")[0]
        scores = np.where(mask, self.emb @ q, -np.inf)    # векторы нормированы -> dot = cos
        return [int(i) for i in np.argsort(-scores)[:N_CANDIDATES] if mask[i]]

    def rerank(self, query: str, cand: list[int]) -> dict[int, int]:
        """LLM ставит каждому кандидату оценку 0–3. Один запрос на все кандидаты."""
        blocks = [f"[id={n}] {self.chunks[i]['title']}\n{self.chunks[i]['text']}"
                  for n, i in enumerate(cand)]
        prompt = (RERANK_PROMPT.read_text(encoding="utf-8")
                  + f"\n\nПИТАННЯ: {query}\n\nФРАГМЕНТИ:\n\n" + "\n\n".join(blocks))
        resp = self.client.models.generate_content(
            model=RERANK_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=list[RerankItem],
            ),
        )
        items = resp.parsed or []
        return {cand[it.id]: it.score for it in items if 0 <= it.id < len(cand)}

    def search(self, query: str, k: int = 5, rerank: bool = True,
               doc_types: set[str] | None = None, date_from: str | None = None,
               trace: dict | None = None) -> list[dict]:
        """trace (для evals): если передать dict, в него запишутся полные ранжирования
        каждого этапа (chunk_id) — так один вызов даёт метрики для BM25 / вектора / гибрида / rerank."""
        mask = self._mask(doc_types, date_from)
        bm, vec = self.bm25_rank(query, mask), self.vector_rank(query, mask)
        fused = rrf([bm, vec])
        order = sorted(fused, key=fused.get, reverse=True)
        if trace is not None:
            ids = lambda idx: [self.chunks[i]["chunk_id"] for i in idx]
            trace.update(bm25=ids(bm), vector=ids(vec), hybrid=ids(order))

        llm = {}
        if rerank and order:
            try:
                llm = self.rerank(query, order[:N_RERANK])
            except Exception as e:   # reranker упал — не роняем поиск, отдаём порядок RRF
                print(f"[rerank пропущен: {e.__class__.__name__}]")
            if llm:
                # сначала оценка LLM, при равенстве — место в RRF; непереоценённые — в конце
                order = sorted(order, key=lambda i: (-llm.get(i, -1), -fused[i]))
        if trace is not None:
            trace["rerank"] = [self.chunks[i]["chunk_id"] for i in order] if llm else None

        results = []
        for i in order[:k]:
            c = self.chunks[i]
            results.append({
                **{key: c[key] for key in ("chunk_id", "doc_id", "title", "doc_type",
                                           "doc_date", "url", "article", "text")},
                "rrf": round(fused[i], 4),
                "bm25_rank": bm.index(i) + 1 if i in bm else None,
                "vec_rank": vec.index(i) + 1 if i in vec else None,
                "llm_score": llm.get(i),
            })
        return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--types", help="через запятую: law,dps,article,guide")
    ap.add_argument("--date-from", help="YYYY-MM-DD")
    args = ap.parse_args()

    r = Retriever()
    hits = r.search(args.query, k=args.k, rerank=not args.no_rerank,
                    doc_types=set(args.types.split(",")) if args.types else None,
                    date_from=args.date_from)
    for n, h in enumerate(hits, 1):
        print(f"{n}. {h['chunk_id']}  [{h['doc_type']} {h['doc_date'] or '—'}]  "
              f"bm25#{h['bm25_rank']} vec#{h['vec_rank']} llm={h['llm_score']}")
        print("   " + h["text"][:200].replace("\n", " ") + "…\n")


if __name__ == "__main__":
    main()
