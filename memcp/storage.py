from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from .config import settings


@dataclass(slots=True)
class ScoredMemory:
    payload: dict[str, Any]
    dense_score: float


class QdrantMemoryStore:
    def __init__(self, vector_size: int) -> None:
        self._collection = settings.collection_name
        if settings.qdrant_mode == "remote":
            if not settings.qdrant_url:
                raise ValueError("QDRANT_URL is required when QDRANT_MODE=remote")
            self._client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
            )
        else:
            try:
                self._client = QdrantClient(path=settings.qdrant_path)
            except RuntimeError as exc:
                msg = str(exc)
                if "already accessed by another instance of Qdrant client" in msg:
                    raise RuntimeError(
                        "MemCP cannot open local Qdrant DB because it is locked by another running process.\n"
                        f"Path: {settings.qdrant_path}\n"
                        "Stop the other MemCP/Qdrant process, or run this instance with a different path:\n"
                        "QDRANT_PATH=~/.memcp/db-dev uv run memcp run"
                    ) from exc
                raise

        self._ensure_collection(vector_size)

    def _ensure_collection(self, vector_size: int) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection in existing:
            return

        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE),
        )

    def upsert(self, memory_id: str, vector: list[float], payload: dict[str, Any]) -> None:
        self._client.upsert(
            collection_name=self._collection,
            points=[qm.PointStruct(id=memory_id, vector=vector, payload=payload)],
            wait=True,
        )

    def search_dense(
        self,
        query_vector: list[float],
        top_k: int,
        user_id: str | None,
        tags: list[str],
    ) -> list[ScoredMemory]:
        query_filter = _build_filter(user_id=user_id, tags=tags)
        # qdrant-client API changed: older versions expose `search`, newer versions use `query_points`.
        if hasattr(self._client, "search"):
            results = self._client.search(
                collection_name=self._collection,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )
        else:
            response = self._client.query_points(
                collection_name=self._collection,
                query=query_vector,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )
            results = response.points
        scored: list[ScoredMemory] = []
        for point in results:
            payload = dict(point.payload or {})
            if "id" not in payload:
                payload["id"] = str(point.id)
            scored.append(ScoredMemory(payload=payload, dense_score=float(point.score)))
        return scored

    def list_memories(
        self,
        limit: int,
        offset: int,
        user_id: str | None,
        tags: list[str],
    ) -> list[dict[str, Any]]:
        query_filter = _build_filter(user_id=user_id, tags=tags)
        all_items: list[dict[str, Any]] = []
        cursor = None

        while True:
            points, cursor = self._client.scroll(
                collection_name=self._collection,
                scroll_filter=query_filter,
                with_payload=True,
                with_vectors=False,
                limit=max(100, limit + offset),
                offset=cursor,
            )
            for point in points:
                payload = dict(point.payload or {})
                if "id" not in payload:
                    payload["id"] = str(point.id)
                all_items.append(payload)
            if cursor is None:
                break

        all_items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return all_items[offset : offset + limit]

    def all_for_filter(self, user_id: str | None, tags: list[str]) -> list[dict[str, Any]]:
        query_filter = _build_filter(user_id=user_id, tags=tags)
        all_items: list[dict[str, Any]] = []
        cursor = None
        while True:
            points, cursor = self._client.scroll(
                collection_name=self._collection,
                scroll_filter=query_filter,
                with_payload=True,
                with_vectors=False,
                limit=200,
                offset=cursor,
            )
            for point in points:
                payload = dict(point.payload or {})
                if "id" not in payload:
                    payload["id"] = str(point.id)
                all_items.append(payload)
            if cursor is None:
                break
        return all_items

    def delete_memory(self, memory_id: str) -> bool:
        self._client.delete(
            collection_name=self._collection,
            points_selector=qm.PointIdsList(points=[memory_id]),
            wait=True,
        )
        return True

    def clear_memories(self, user_id: str | None) -> bool:
        if user_id is None:
            self._client.delete(
                collection_name=self._collection,
                points_selector=qm.FilterSelector(filter=qm.Filter()),
                wait=True,
            )
            return True

        self._client.delete(
            collection_name=self._collection,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[qm.FieldCondition(key="user_id", match=qm.MatchValue(value=user_id))]
                )
            ),
            wait=True,
        )
        return True



def _build_filter(user_id: str | None, tags: list[str]) -> qm.Filter | None:
    must: list[qm.FieldCondition] = []
    if user_id:
        must.append(qm.FieldCondition(key="user_id", match=qm.MatchValue(value=user_id)))
    for tag in tags:
        must.append(qm.FieldCondition(key="tags", match=qm.MatchValue(value=tag)))
    if not must:
        return None
    return qm.Filter(must=must)
