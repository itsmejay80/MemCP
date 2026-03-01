from __future__ import annotations

import re
from typing import Any

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from .config import settings
from .models import ListMemoryInput, SearchMemoryInput, new_memory_payload
from .storage import QdrantMemoryStore


class MemoryEngine:
    def __init__(self) -> None:
        self._embedder = SentenceTransformer(settings.embedding_model)
        vector_size = self._embedder.get_sentence_embedding_dimension()
        if vector_size is None:
            raise RuntimeError("Embedding model returned no vector dimension")
        self._store = QdrantMemoryStore(vector_size=vector_size)

    def add_memory(
        self,
        content: str,
        tags: list[str] | None = None,
        user_id: str | None = None,
        source: str = "mcp",
    ) -> dict[str, Any]:
        payload = new_memory_payload(
            content=content,
            atomic_facts=self._extract_atomic_facts(content),
            tags=tags or [],
            user_id=user_id,
            source=source,
        )
        searchable_text = "\n".join(payload["atomic_facts"] or [content])
        vector = self._embed_text(searchable_text)
        self._store.upsert(memory_id=payload["id"], vector=vector, payload=payload)
        return payload

    def search_memory(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float | None = None,
        tags: list[str] | None = None,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        search = SearchMemoryInput(
            query=query,
            top_k=top_k or settings.top_k,
            min_score=settings.min_score if min_score is None else min_score,
            tags=tags or [],
            user_id=user_id,
        )
        dense_candidates = self._store.search_dense(
            query_vector=self._embed_text(search.query),
            top_k=max(30, search.top_k * 4),
            user_id=search.user_id,
            tags=search.tags,
        )
        full_candidates = self._store.all_for_filter(user_id=search.user_id, tags=search.tags)

        if not dense_candidates and not full_candidates:
            return []

        combined = self._hybrid_rank(
            query=search.query,
            dense_candidates=dense_candidates,
            full_candidates=full_candidates,
        )
        if search.min_score is not None:
            combined = [item for item in combined if item["score"] >= search.min_score]
        return combined[: search.top_k]

    def list_memories(
        self,
        limit: int = 50,
        offset: int = 0,
        tags: list[str] | None = None,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        query = ListMemoryInput(limit=limit, offset=offset, tags=tags or [], user_id=user_id)
        return self._store.list_memories(
            limit=query.limit,
            offset=query.offset,
            user_id=query.user_id,
            tags=query.tags,
        )

    def delete_memory(self, memory_id: str) -> bool:
        return self._store.delete_memory(memory_id)

    def clear_memories(self, user_id: str | None = None) -> bool:
        return self._store.clear_memories(user_id=user_id)

    def _embed_text(self, text: str) -> list[float]:
        vec = self._embedder.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def _extract_atomic_facts(self, content: str) -> list[str]:
        if not settings.enable_atomic_extract:
            return [content]

        # Lightweight local-first extraction: split into sentence-level facts.
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", content.strip()) if s.strip()]
        if not sentences:
            return [content]
        return sentences[:5]

    def _hybrid_rank(
        self,
        query: str,
        dense_candidates: list[Any],
        full_candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        dense_by_id: dict[str, float] = {}
        payload_by_id: dict[str, dict[str, Any]] = {}
        for scored in dense_candidates:
            memory_id = str(scored.payload["id"])
            dense_by_id[memory_id] = scored.dense_score
            payload_by_id[memory_id] = scored.payload

        corpus: list[list[str]] = []
        corpus_ids: list[str] = []
        for payload in full_candidates:
            memory_id = str(payload["id"])
            facts = payload.get("atomic_facts") or []
            text = " ".join(facts) if facts else str(payload.get("content", ""))
            tokens = _tokenize(text)
            if not tokens:
                continue
            corpus.append(tokens)
            corpus_ids.append(memory_id)
            payload_by_id.setdefault(memory_id, payload)

        bm25_scores: dict[str, float] = {}
        if corpus:
            bm25 = BM25Okapi(corpus)
            q_tokens = _tokenize(query)
            raw_scores = bm25.get_scores(q_tokens)
            if len(raw_scores) > 0:
                max_score = max(raw_scores)
                min_bm25 = min(raw_scores)
                span = max(max_score - min_bm25, 1e-9)
                for idx, score in enumerate(raw_scores):
                    bm25_scores[corpus_ids[idx]] = (score - min_bm25) / span

        # Weighted combination favors semantic relevance while preserving keyword precision.
        hybrid: list[tuple[str, float]] = []
        for memory_id, payload in payload_by_id.items():
            dense = dense_by_id.get(memory_id, 0.0)
            sparse = bm25_scores.get(memory_id, 0.0)
            score = 0.7 * dense + 0.3 * sparse
            hybrid.append((memory_id, score))

        hybrid.sort(key=lambda x: x[1], reverse=True)
        output: list[dict[str, Any]] = []
        for memory_id, score in hybrid:
            item = dict(payload_by_id[memory_id])
            item["score"] = round(score, 4)
            output.append(item)
        return output


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_\-]+", text.lower())
