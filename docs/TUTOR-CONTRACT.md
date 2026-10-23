# Tutor contract v1 — the index MCP's frozen response shapes

Status: **frozen at M6 (2026-10-23)**. The future tutor package
codes against exactly these shapes; the fixtures that pin them
live in `tests/unit/index_mcp/test_contract.py`. A shape change is
a contract version bump: change the fixtures, this page, and the
version together — never silently.

Transport: MCP (Streamable HTTP) at the gateway route
`/mcp/index/`, bearer token (client `index-mcp`). All three tools
return an **envelope object** — never a bare list.

## The envelope

```json
{"match_quality": "strong" | "weak", "passages":  [ ... ]}
{"match_quality": "strong" | "weak", "resources": [ ... ]}
```

`match_quality` calibrates the result set AS A WHOLE:

- **"strong"** — the top result is real material on the topic;
  the tutor may point at it and cite it.
- **"weak"** — the corpus likely has NO real material on this.
  The tutor's contract: **say "I don't have material on that" and
  do not present the top hit as on-topic** — however plausible it
  looks. (On absent topics the engine's fallback is the Team
  Weekly Agenda at scores that overlap genuine weak hits; this
  field exists precisely so the tutor never points Sam there.)

An empty result list always carries `"weak"`.

## search_passages(query, course?, limit=5)

Envelope key `passages`. Each passage:

| Field | Type | Meaning |
|---|---|---|
| text | string | the chunk's text |
| score | float | cosine vs the query (bge-small scale) |
| item_id | string | corpus item the chunk belongs to |
| title | string | the item's title |
| course | string \| null | course as the adapter reports it |
| deep_link | string | **the original resource URL — cite this** |
| page | int \| null | starting page when the source has page structure |
| source_kind | string | adapter source kind (AGENDA, HAC, ...) |

Ranking: raw chunk cosine, best first. `course` filters to one
course (post-filter; results may be fewer than `limit`).

## resources_for_topic(topic, course?, limit=5)

Envelope key `resources`. Each resource:

| Field | Type | Meaning |
|---|---|---|
| item_id | string | corpus item id |
| title | string | the item's title |
| course | string \| null | as above |
| deep_link | string | **the original resource URL — cite this** |
| best_score | float | the item's single nearest chunk (cosine) |
| chunk_hits | int | of the item's chunks, how many appear in the candidate window |
| rank_score | float | best_score + breadth bonus (0.02/hit over the first, capped +0.20) |
| source_kind | string | as above |

Ranking (M6): items order FIRST by the band of `best_score`
(bands are 0.04 of cosine wide) — no breadth bonus lifts an item
over one whose best chunk is a full band nearer — then by
`rank_score`, then `best_score`, then item_id. Pointers should be
read in order; the first resource is the pointer of record when
`match_quality` is "strong".

## worked_examples(topic, limit=5)

Same envelope and passage shape as `search_passages`. Honest-v1
note: the tool reformulates the query as "worked example:
<topic>" and returns passages that read as worked material — they
are **not verified worked examples**, and the passages
`match_quality` calibration is inherited, not separately
benchmarked (open calibration item).

## match_quality — derivation and limits

Computed at runtime by `index_mcp.quality` (pure functions over
the returned window) from signals the response itself carries:

- **Passages:** strong iff top score ≥ 0.675 AND the top item's
  best chunk beats every other item's best in the window by
  ≥ 0.02 (no rival item in the window ⇒ margin satisfied).
- **Resources:** strong iff #1 `best_score` ≥ 0.67 AND (#1
  `chunk_hits` ≥ 5 OR #1 `best_score` ≥ 0.72).

Derived from the M5 benchmark's recorded distributions (labeled
set v1.0, 29 scored + 4 absent-topic queries, 38 items / 371
chunks): the absent probes' tops overlap genuine weak hits on raw
score alone (+0.10 separation), but absent winners never stand
clear of the field on margin, and the one spurious resources
winner above the score gate (0.689) held only 4 candidate hits.
Validated at M6 on the same set: all 4 absent probes "weak" on
both tools; scored tops "strong" 28/29 (resources) and 27/29
(passages) — the passage exceptions are ties between two
genuinely relevant items, an accepted under-claim.

**Limits:** the constants sit in the observed gaps of ONE
33-query run on ONE embedder (bge-small-en-v1.5). They are corpus-
and model-specific. Re-derive from a fresh benchmark run when the
corpus grows materially (post-Classroom expansion), the embedding
model changes, or score distributions move. `chunk_hits` is
counted within the candidate window the call fetched (depth
scales with `limit`); the calibration was validated at the
benchmark depth (limit 10) and spot-checked at the tutor default
(limit 5).
