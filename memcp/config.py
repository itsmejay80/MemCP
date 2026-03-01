from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(slots=True)
class Settings:
    qdrant_mode: str = os.getenv("QDRANT_MODE", "local")
    qdrant_path: str = os.getenv("QDRANT_PATH", os.path.expanduser("~/.memcp/db"))
    qdrant_url: str | None = os.getenv("QDRANT_URL")
    qdrant_api_key: str | None = os.getenv("QDRANT_API_KEY")
    collection_name: str = os.getenv("QDRANT_COLLECTION", "memories")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    top_k: int = int(os.getenv("TOP_K", "10"))
    min_score: float = float(os.getenv("MIN_SCORE", "0.2"))
    dedupe_skip_threshold: float = float(
        os.getenv("DEDUPE_SKIP_THRESHOLD", os.getenv("DEDUPE_THRESHOLD", "0.92"))
    )
    dedupe_merge_threshold: float = float(os.getenv("DEDUPE_MERGE_THRESHOLD", "0.97"))
    enable_atomic_extract: bool = os.getenv("ATOMIC_EXTRACT", "true").lower() == "true"
    auth_token: str | None = os.getenv("MEMCP_AUTH_TOKEN")
    require_auth: bool = os.getenv("MEMCP_REQUIRE_AUTH", "false").lower() == "true" or bool(
        os.getenv("MEMCP_AUTH_TOKEN")
    )


settings = Settings()
