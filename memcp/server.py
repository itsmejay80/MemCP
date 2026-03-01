from __future__ import annotations

from fastmcp import FastMCP

from .config import settings
from .engine import MemoryEngine

mcp = FastMCP(name="memcp")
_engine: MemoryEngine | None = None


def _get_engine() -> MemoryEngine:
    global _engine
    if _engine is None:
        _engine = MemoryEngine()
    return _engine


def _require_auth(auth_token: str | None) -> None:
    if not settings.require_auth:
        return
    expected = settings.auth_token
    if not expected:
        raise RuntimeError(
            "Auth is enabled but MEMCP_AUTH_TOKEN is not configured. Set it in environment or .env."
        )
    if auth_token != expected:
        raise PermissionError("Invalid auth token for MemCP MCP tool call")


@mcp.tool()
def add_memory(
    content: str,
    tags: list[str] | None = None,
    user_id: str | None = None,
    source: str = "mcp",
    auth_token: str | None = None,
) -> dict:
    """Store a memory in local-first vector storage."""
    _require_auth(auth_token)
    return _get_engine().add_memory(content=content, tags=tags, user_id=user_id, source=source)


@mcp.tool()
def search_memory(
    query: str,
    top_k: int = 10,
    min_score: float | None = 0.2,
    tags: list[str] | None = None,
    user_id: str | None = None,
    auth_token: str | None = None,
) -> list[dict]:
    """Search memories with hybrid dense + BM25 ranking."""
    _require_auth(auth_token)
    return _get_engine().search_memory(
        query=query,
        top_k=top_k,
        min_score=min_score,
        tags=tags,
        user_id=user_id,
    )


@mcp.tool()
def list_memories(
    limit: int = 50,
    offset: int = 0,
    tags: list[str] | None = None,
    user_id: str | None = None,
    auth_token: str | None = None,
) -> list[dict]:
    """List stored memories, newest first."""
    _require_auth(auth_token)
    return _get_engine().list_memories(limit=limit, offset=offset, tags=tags, user_id=user_id)


@mcp.tool()
def delete_memory(memory_id: str, auth_token: str | None = None) -> bool:
    """Delete one memory by ID."""
    _require_auth(auth_token)
    return _get_engine().delete_memory(memory_id)


@mcp.tool()
def clear_memories(user_id: str | None = None, auth_token: str | None = None) -> bool:
    """Delete all memories (or all for a given user_id)."""
    _require_auth(auth_token)
    return _get_engine().clear_memories(user_id=user_id)


def run() -> None:
    mcp.run()
