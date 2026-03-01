from __future__ import annotations

import re
from typing import Any

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from .config import settings
from .models import AddMemoryInput, ListMemoryInput, SearchMemoryInput, new_memory_payload
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
        dedupe: bool = True,
        dedupe_policy: str = "auto",
        dedupe_threshold: float | None = None,
        dedupe_skip_threshold: float | None = None,
        dedupe_merge_threshold: float | None = None,
    ) -> dict[str, Any]:
        add_input = AddMemoryInput(
            content=content,
            tags=tags or [],
            user_id=user_id,
            source=source,
            dedupe=dedupe,
            dedupe_policy=dedupe_policy,
            dedupe_threshold=dedupe_threshold,
            dedupe_skip_threshold=dedupe_skip_threshold,
            dedupe_merge_threshold=dedupe_merge_threshold,
        )
        if add_input.dedupe_policy not in {"auto", "skip", "merge"}:
            raise ValueError("dedupe_policy must be 'auto', 'skip' or 'merge'")

        payload = new_memory_payload(
            content=add_input.content,
            atomic_facts=self._extract_atomic_facts(add_input.content),
            tags=add_input.tags,
            user_id=add_input.user_id,
            source=add_input.source,
        )
        searchable_text = "\n".join(payload["atomic_facts"] or [add_input.content])
        vector = self._embed_text(searchable_text)

        if add_input.dedupe:
            skip_threshold, merge_threshold = self._resolve_dedupe_thresholds(add_input)
            dedupe_match = self._find_dedupe_match(
                vector=vector,
                user_id=add_input.user_id,
                threshold=skip_threshold,
            )
            if dedupe_match is not None:
                existing = dict(dedupe_match.payload)
                dedupe_score = round(dedupe_match.dense_score, 4)
                is_high_confidence = dedupe_match.dense_score >= merge_threshold
                if add_input.dedupe_policy == "skip":
                    existing["dedupe_action"] = "skipped"
                    existing["dedupe_score"] = dedupe_score
                    existing["dedupe_of"] = str(existing.get("id", ""))
                    existing["dedupe_band"] = "high_confidence" if is_high_confidence else "borderline"
                    return existing
                if add_input.dedupe_policy == "auto" and not is_high_confidence:
                    payload["possible_duplicate_of"] = str(existing.get("id", ""))
                    payload["possible_duplicate_score"] = dedupe_score
                    payload["dedupe_band"] = "borderline"
                    self._store.upsert(memory_id=payload["id"], vector=vector, payload=payload)
                    payload["dedupe_action"] = "inserted"
                    return payload

                merged = self._merge_payloads(existing=existing, incoming=payload)
                merged_searchable_text = "\n".join(merged["atomic_facts"] or [merged["content"]])
                merged_vector = self._embed_text(merged_searchable_text)
                self._store.upsert(memory_id=str(merged["id"]), vector=merged_vector, payload=merged)
                merged["dedupe_action"] = "merged"
                merged["dedupe_score"] = dedupe_score
                merged["dedupe_of"] = str(existing.get("id", ""))
                merged["dedupe_band"] = "high_confidence" if is_high_confidence else "borderline"
                return merged

        self._store.upsert(memory_id=payload["id"], vector=vector, payload=payload)
        payload["dedupe_action"] = "inserted"
        return payload

    def search_memory(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float | None = None,
        tags: list[str] | None = None,
        user_id: str | None = None,
        compact: bool = False,
        include_explain: bool = False,
    ) -> list[dict[str, Any]]:
        search = SearchMemoryInput(
            query=query,
            top_k=top_k or settings.top_k,
            min_score=settings.min_score if min_score is None else min_score,
            tags=tags or [],
            user_id=user_id,
            compact=compact,
            include_explain=include_explain,
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
            include_explain=search.include_explain,
        )
        if search.min_score is not None:
            combined = [item for item in combined if item["score"] >= search.min_score]
        combined = combined[: search.top_k]
        if search.compact:
            return [self._to_compact_result(item) for item in combined]
        return combined

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
        include_explain: bool,
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
        hybrid: list[tuple[str, float, float, float]] = []
        for memory_id in payload_by_id:
            dense = dense_by_id.get(memory_id, 0.0)
            bm25 = bm25_scores.get(memory_id, 0.0)
            final_score = 0.7 * dense + 0.3 * bm25
            hybrid.append((memory_id, dense, bm25, final_score))

        hybrid.sort(key=lambda x: x[3], reverse=True)
        output: list[dict[str, Any]] = []
        for memory_id, dense, bm25, final_score in hybrid:
            item = dict(payload_by_id[memory_id])
            item["score"] = round(final_score, 4)
            if include_explain:
                item["explain"] = {
                    "dense": round(dense, 4),
                    "bm25": round(bm25, 4),
                    "final": round(final_score, 4),
                }
            output.append(item)
        return output

    def _resolve_dedupe_thresholds(self, add_input: AddMemoryInput) -> tuple[float, float]:
        skip_threshold = (
            add_input.dedupe_threshold
            if add_input.dedupe_threshold is not None
            else (
                add_input.dedupe_skip_threshold
                if add_input.dedupe_skip_threshold is not None
                else settings.dedupe_skip_threshold
            )
        )
        merge_threshold = (
            add_input.dedupe_merge_threshold
            if add_input.dedupe_merge_threshold is not None
            else settings.dedupe_merge_threshold
        )
        if merge_threshold < skip_threshold:
            raise ValueError("dedupe_merge_threshold must be >= dedupe_skip_threshold")
        return skip_threshold, merge_threshold

    def _find_dedupe_match(
        self,
        vector: list[float],
        user_id: str | None,
        threshold: float,
    ) -> Any | None:
        candidates = self._store.search_dense(
            query_vector=vector,
            top_k=1,
            user_id=user_id,
            tags=[],
        )
        if not candidates:
            return None
        match = candidates[0]
        if match.dense_score < threshold:
            return None
        return match

    def _merge_payloads(self, existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
        merged = dict(existing)
        merged["tags"] = _unique_keep_order(
            [str(tag) for tag in existing.get("tags", [])] + [str(tag) for tag in incoming.get("tags", [])]
        )
        merged["atomic_facts"] = _unique_keep_order(
            [str(fact) for fact in existing.get("atomic_facts", [])]
            + [str(fact) for fact in incoming.get("atomic_facts", [])]
        )[:8]
        return merged

    def _to_compact_result(self, item: dict[str, Any]) -> dict[str, Any]:
        compact: dict[str, Any] = {
            "id": item.get("id"),
            "content": item.get("content"),
            "score": item.get("score"),
        }
        if "explain" in item:
            compact["explain"] = item["explain"]
        return compact


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_\-]+", text.lower())


def _unique_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered
