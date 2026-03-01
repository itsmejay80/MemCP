# MemCP Implementation TODO

Last updated: 2026-03-01

## Done
- [x] Add `top_k` to search interface and engine flow
- [x] Add `min_score` to search (default `0.2`)
- [x] Add `MIN_SCORE` to config/env/docs

## Next Sprint (Highest Impact)
- [x] Add `compact=true` response mode for `search_memory` (token-efficient payload)
- [x] Add `include_explain=true` for score breakdown (`dense`, `bm25`, `final`)
- [x] Add write-time dedupe in `add_memory` (skip/merge policy + threshold)

## Ingestion Improvements
- [ ] Replace sentence split with configurable chunking (size + overlap)
- [ ] Improve local fact extraction (clause split, drop filler, dedupe facts)
- [ ] Add ingest metadata: `parent_memory_id`, `chunk_index`, `chunk_count`, `atomic_facts_version`
- [ ] Add `normalized_content` + `quality_flags` in payload

## Retrieval Improvements
- [ ] Add ranking weights via env:
  - `DENSE_WEIGHT`
  - `BM25_WEIGHT`
  - `RECENCY_WEIGHT`
  - `TAG_WEIGHT`
- [ ] Add fallback strategy when no result passes `min_score` (optional: return top 1)
- [ ] Add lightweight query normalization/synonym expansion

## API Enhancements
- [ ] Add `update_memory`
- [ ] Add `upsert_memory`
- [ ] Add bulk ingest endpoint
- [ ] Add search pagination metadata (`has_more`, `next_offset`)

## Performance + Reliability
- [ ] Build benchmark harness (1k/5k/10k memories; p50/p95 latency)
- [ ] Add quality regression tests (`query -> expected top IDs`)
- [ ] Add startup/lock diagnostics and friendlier runtime errors

## Security + Operations
- [ ] Add audit log for destructive actions (`delete_memory`, `clear_memories`)
- [ ] Add request rate limiting / abuse guardrails
- [ ] Add auth token rotation guidance + docs

## Nice to Have
- [ ] Add importance/pinning support
- [ ] Add project/namespace isolation
- [ ] Add recency decay tuning profile
