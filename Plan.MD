## MemCP MVP+1 Improvement Plan (Quality-First, OSS-Only)

### Summary
Improve memory relevance while keeping local performance practical, using only open-source/free local components.  
Chosen direction:
- Track: **Memory Quality**
- Objective: **Balanced relevance + latency**
- Constraint: **Fully local only (no cloud extraction fallback)**

---

## 1. Highest-Impact Improvements (Prioritized)

### P0: Retrieval Quality Foundation
1. **Better chunking at ingest**
- Replace sentence split with configurable chunking:
  - semantic paragraph boundaries
  - max token/char budget
  - overlap window
- Store `chunk_index`, `chunk_count`, `parent_memory_id`.

2. **Two-stage ranking**
- Stage A: fast dense retrieval from Qdrant (`top_n_candidates`, e.g., 50).
- Stage B: local rerank with lexical + metadata boosts:
  - BM25 score
  - recency boost
  - exact tag/user match boost
- Keep final score transparent (`dense`, `bm25`, `boost`, `final`).

3. **Query normalization + expansion (lightweight)**
- Normalize punctuation/casing/numbers.
- Add optional synonym expansion dictionary (local static map).
- Preserve exact-match tokens for entities and numbers.

### P1: Better Ingest “Atomic Facts” (Local)
4. **Upgrade extraction from simple sentence split**
- Add local fact extractor pipeline:
  - clause splitting
  - dedupe near-duplicates
  - keep factual statements only (drop greetings/filler)
- Store `atomic_facts_version` for future migrations.

5. **Duplicate suppression**
- At write time, detect near-duplicate memories using embedding similarity threshold.
- If duplicate, either:
  - skip insert and return existing ID, or
  - merge metadata/tags (configurable).

### P1: Relevance Controls
6. **Configurable ranking weights**
- Add env-configurable weights:
  - `DENSE_WEIGHT`, `BM25_WEIGHT`, `RECENCY_WEIGHT`, `TAG_WEIGHT`
- Add caps and defaults to avoid unstable ranking.

7. **Query-time filters and explainability**
- Return `match_explain` structure in search results:
  - matched terms
  - active filters
  - component scores

---

## 2. Public API / Interface Changes

### MCP Tool Changes
1. `add_memory(...)`
- Add optional params:
  - `dedupe: bool = true`
  - `chunking_strategy: str | None = None`
- Return:
  - `chunks_created`
  - `dedupe_action` (`inserted|merged|skipped`)

2. `search_memory(...)`
- Add optional params:
  - `min_score: float | None`
  - `include_explain: bool = false`
- Return each item with:
  - `score_breakdown` (when `include_explain=true`)
  - `matched_terms`

3. `list_memories(...)`
- Add optional sort mode:
  - `sort_by: "created_at" | "updated_at" | "importance"`

### Storage Payload Additions
- `parent_memory_id`
- `chunk_index`
- `chunk_count`
- `normalized_content`
- `atomic_facts_version`
- `updated_at`
- `quality_flags` (e.g., deduped, short_text, low_signal)

---

## 3. Performance and Quality Targets

### Acceptance Targets (local machine, 10k memories)
- `search_memory` p95: **<= 100ms** (balanced mode)
- Top-5 relevance gain vs current baseline: **+20%+** on local eval set
- Duplicate rate reduction: **>= 40%** fewer redundant points
- No regressions in correctness for tag/user filtering

---

## 4. Testing Plan

### Unit Tests
1. Chunker behavior:
- paragraph split, overlap, max-size limits.
2. Local fact extraction:
- removes filler, preserves factual clauses, deterministic output.
3. Ranking combiner:
- weight application, normalization, stable ordering.
4. Duplicate suppression:
- threshold boundaries, merge/skip behavior.
5. Query normalization:
- token handling for numbers/symbols/entities.

### Integration Tests
1. End-to-end add/search/list/delete with local Qdrant.
2. Quality regression suite with fixed fixtures:
- known queries -> expected top-k IDs.
3. Filter correctness:
- `tags`, `user_id`, combined filters.
4. API compatibility:
- Qdrant client variants (`query_points`/`search`) still supported.

### Non-Functional Tests
1. Latency benchmark script for 1k/5k/10k memories.
2. Memory footprint check for larger corpora.
3. Startup behavior with locked DB path and friendly error handling.

---

## 5. Rollout Plan (Incremental)

1. **Phase A (safe):** chunking + ranking weights + score breakdown (no schema migration required where possible).
2. **Phase B:** improved local fact extraction + dedupe.
3. **Phase C:** benchmark harness + quality gate in CI.
4. **Phase D:** tune defaults from benchmark results and freeze “MVP+1 quality profile”.

---

## 6. Assumptions and Defaults
- All quality improvements remain **fully local OSS**.
- Keep current auth model; no auth redesign in this phase.
- Default ranking profile:
  - `dense=0.6`, `bm25=0.3`, `recency=0.1`, `tag boost on exact match`.
- Dedupe default: `enabled` with conservative threshold.
- No cloud dependencies introduced.
