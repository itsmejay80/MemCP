# MemCP MVP

Private, local-first memory MCP server.

This MVP is built entirely with open-source/free tools:
- FastMCP for MCP tooling
- Qdrant (local embedded mode) for vector storage
- sentence-transformers for local embeddings
- BM25 (`rank-bm25`) for keyword ranking

## Features in MVP

- `add_memory` (write-time dedupe with score bands and `dedupe_policy=auto|skip|merge`)
- `search_memory` (hybrid dense + BM25, with `top_k`, `min_score`, `compact`, `include_explain`)
- `list_memories`
- `delete_memory`
- `clear_memories`

## Quickstart (uv)

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
uv run memcp run
```

By default this runs fully local/offline (except first-time model download):
- `QDRANT_MODE=local`
- `QDRANT_PATH=~/.memcp/db`
- `EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2`

## Optional Auth Token

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Set:
- `MEMCP_AUTH_TOKEN=your-secret-token`

When this is set, auth is enforced for all tools automatically.
Every tool call must include `auth_token` matching `MEMCP_AUTH_TOKEN`.

## Claude Desktop MCP config

```json
{
  "mcpServers": {
    "memcp": {
      "command": "memcp",
      "args": ["run"],
      "env": {
        "QDRANT_MODE": "local",
        "MEMCP_AUTH_TOKEN": "your-secret-token"
      }
    }
  }
}
```

## Environment variables

- `QDRANT_MODE`: `local` or `remote` (default `local`)
- `QDRANT_PATH`: local database path (default `~/.memcp/db`)
- `QDRANT_URL`: required for remote mode
- `QDRANT_API_KEY`: optional for remote mode
- `QDRANT_COLLECTION`: collection name (default `memories`)
- `EMBEDDING_MODEL`: sentence-transformers model id
- `ATOMIC_EXTRACT`: `true`/`false` local sentence-based extraction (default `true`)
- `TOP_K`: default search result count (default `10`)
- `MIN_SCORE`: default minimum score filter for search results (default `0.2`)
- `DEDUPE_THRESHOLD`: legacy alias for skip threshold (default `0.92`)
- `DEDUPE_SKIP_THRESHOLD`: dedupe enter threshold (default `0.92`)
- `DEDUPE_MERGE_THRESHOLD`: auto-merge threshold (default `0.97`)
- `MEMCP_AUTH_TOKEN`: if set, all MCP tools require matching `auth_token`
- `MEMCP_REQUIRE_AUTH`: force auth mode even without auto-detection (`true`/`false`)

## Notes

- Atomic fact extraction in this MVP is fully local and rule-based (sentence splitting).
- Hosted/cloud LLM extraction can be added later as an optional mode.
- Dedupe policy is score-banded by default:
  - `score < DEDUPE_SKIP_THRESHOLD`: insert new
  - `DEDUPE_SKIP_THRESHOLD <= score < DEDUPE_MERGE_THRESHOLD`: borderline band
    - `dedupe_policy=auto` (default): insert new + mark `possible_duplicate_of`
    - `dedupe_policy=skip`: skip insert
    - `dedupe_policy=merge`: merge into existing
  - `score >= DEDUPE_MERGE_THRESHOLD`: high-confidence band
    - `dedupe_policy=auto|merge`: merge into existing
    - `dedupe_policy=skip`: skip insert
