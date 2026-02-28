from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


@dataclass(slots=True)
class Memory:
    id: str
    content: str
    atomic_facts: list[str]
    tags: list[str]
    user_id: str | None
    source: str
    created_at: str


@dataclass(slots=True)
class AddMemoryInput:
    content: str
    tags: list[str] = field(default_factory=list)
    user_id: str | None = None
    source: str = "mcp"


@dataclass(slots=True)
class SearchMemoryInput:
    query: str
    top_k: int = 10
    tags: list[str] = field(default_factory=list)
    user_id: str | None = None


@dataclass(slots=True)
class ListMemoryInput:
    limit: int = 50
    offset: int = 0
    tags: list[str] = field(default_factory=list)
    user_id: str | None = None


def new_memory_payload(
    content: str,
    atomic_facts: list[str],
    tags: list[str],
    user_id: str | None,
    source: str,
) -> dict[str, Any]:
    return {
        "id": str(uuid4()),
        "content": content,
        "atomic_facts": atomic_facts,
        "tags": tags,
        "user_id": user_id,
        "source": source,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
