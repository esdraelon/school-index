# school-index

Searchable index over Sam's school material: resources harvested
from the school-info-adapter, extracted to text, chunked and
embedded, and served as a search MCP on hal9k. It is the key
enabler for the school-assistant tutor/study personas — passage
search, topic → resource pointers (with the original deep links),
and worked-example retrieval.

Planning and design live in the **school-assistant** repo:

- `docs/PROJECT-INDEX.md` — the v1 plan + milestone checklist
- `docs/INDEX-DESIGN.md` — components, schemas, tool contracts
- `docs/CODING-STANDARD.md` — the conventions this repo follows
- `docs/DECISIONS.md` — settled decisions, incl. the boundary below

## Components (`src/school_index/`)

- **`corpus_sync`** — harvester over the school adapter's MCP
  contract; lands originals + metadata in the corpus store,
  incrementally, with tombstones and per-item failure records.
- **`doc_extract`** — format → extractor strategy registry
  producing item text, with a stored outcome record per item
  (extracted / empty / failed / not-extractable) and delta-only
  re-runs keyed on the item's content hash.
- **`embed_index`** — structure-aware chunking, an embedding
  provider strategy registry (ONNX default, Bedrock benchmark arm),
  and the sqlite-vec vector store.
- **`index_mcp`** — the read-only search MCP (`search_passages`,
  `resources_for_topic`, `worked_examples`); every result carries
  provenance and the original deep link.

## Boundary decision (M0, settled 2026-10-23)

`corpus_sync` consumes the adapter's resource types **via its MCP
contract** (`get_resources`) — never its database — and
`school-core` extraction is **not** triggered: there is no second
consumer of the domain types yet.

Evidence (live gateway probe, 2026-10-23): `get_resources("sam")`
returned all **200 resources in a single call, no pagination** —
TUTORIAL 102, LINK 57, DOCUMENT 21, NOTES 20 — each carrying
student, course, title, kind, url, source_kind, and scraped_at,
with a url on all 200 (docs.google.com 129, drive.google.com 44,
www.ixl.com 19, sites.google.com 7, www.google.com 1). At this
scale the contract is the boundary; revisit extraction only when
a second type consumer actually appears (see DECISIONS.md in
school-assistant).

## Development

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
./scripts/verify.sh   # unit tests, coverage >=90, mypy strict, 500-LOC cap
```

Status: **M6 — v1 complete** — `corpus_sync` (M1), `doc_extract`
(M2), `embed_index` (M3), `index_mcp` (M4), the benchmark (M5),
and the tutor contract (M6) landed. The harvester
pulls Sam's resources through the adapter's MCP contract into the
corpus store (SQLite metadata + original files on disk), with
per-item failure records, tombstones, and idempotent re-runs; the
extractor then turns harvested files into text — stored in
`item_text` with `[[page N]]` page-boundary markers — under a
per-item outcome record; the embedder chunks that text on page
structure and embeds it with BAAI/bge-small-en-v1.5 on ONNX,
delta-only, under a per-item `embed_state` record; the search MCP
then serves the index read-only (`search_passages`,
`resources_for_topic`, `worked_examples`) through the gateway at
`/mcp/index/`, and a systemd timer keeps the pipeline fresh
(sync → extract → embed every 6h at :45). See `deploy/README.md`.

## Tutor contract + match_quality (M6)

The three tools return envelopes — `{"match_quality", "passages"
| "resources"}` — frozen as **tutor contract v1** in
`docs/TUTOR-CONTRACT.md` (fixtures:
`tests/unit/index_mcp/test_contract.py`). `match_quality`
("strong" | "weak") calibrates a result set as a whole: on
"weak", the corpus likely has no material on the topic and the
tutor must say so instead of pointing at the top hit — on absent
topics the engine's fallback is the Team Weekly Agenda at scores
that overlap genuine weak hits, so raw scores alone cannot gate
pointers.

The calibration (`index_mcp/quality.py`) is derived from the M5
benchmark's recorded distributions, not tuned by feel: passages
need a top score ≥ 0.675 plus a standout (a ≥ 0.02 margin over
the runner-up item, or a decisive top ≥ 0.72); resources need a
#1 best_score ≥ 0.67 plus breadth (≥ 5 candidate hits) or a
decisive best (≥ 0.72). Validated on the labeled set (live
service and stored records agree exactly): all 4 absent probes
weak on both tools, scored tops strong 28/29 (resources) and
27/29 (passages). The constants
sit in one 33-query run's observed gaps on bge-small's cosine
scale — re-derive them when the corpus or the model changes (see
the contract doc's limits section).

M6 also retuned `resources_for_topic`: items now order by the
**band** of their best chunk (0.04 of cosine) before the breadth
bonus applies, so breadth can no longer compensate a full-band
relevance gap — the M5 colonial-trade miss (a 3-chunk deck with
the query's best chunk, outranked by the 63-chunk agenda) now
ranks #1, and the re-run benchmark records resources hit@3 1.00.

## Serving (M4): one database, WAL, readers beside writers

The corpus DB is shared by the host pipeline (writer, user
haliffer) and the index MCP container (reader, uid 10001). The
reconciliation is **WAL mode**: the stage stores' file factories
set `journal_mode=WAL` (+ `busy_timeout`), so the server's reads
never block the pipeline's writes or vice versa. The container's
data mount is read-write *for the WAL sidecars*; the server's
read-only guarantee lives in its code — `index_mcp`'s store issues
SELECTs only (no migrate, no schema creation) and a system test
proves serving moves no rows. Write access for uid 10001 is a
POSIX ACL on the data dir; ownership stays with haliffer.

## Vector store (M3): sqlite-vec, with a tested fallback

Vectors live behind a `VectorBackend` interface in
`embed_index/store.py` with two implementations:

- **`sqlite-vec` (the design, in use):** a vec0 virtual table with
  the cosine metric. Verified loadable and KNN-correct on this
  repo's dev machine and on hal9k (aarch64) — the live build over
  the real corpus runs on it.
- **`blob-numpy` (fallback):** vectors as BLOBs, brute-force
  cosine in numpy. `EmbedStore.open` probes the extension and
  picks this automatically if vec0 cannot load; the contract
  tests run identically over both backends, and every report line
  names the backend actually in use (`backend=` in the CLI
  output).

Model files live under `<corpus root>/models` (fetch once with
`python -m school_index.embed_index fetch-model`; runs are
offline after that).

## Store layering (M2)

One corpus database (`corpus.db` under the corpus root), one
ordered migration chain (in `corpus_sync/store.py`; append-only),
and one repository per pipeline stage over its own tables:

- `corpus_sync` owns `corpus_items` / `sync_state` and the blob
  files, through `CorpusStore`.
- `doc_extract` owns `item_text` / `extraction_state` (schema:
  migration v2), through `ExtractStore`. It *reads* items and blob
  paths through the `CorpusStore` — stages consume each other's
  records through repositories, never each other's SQL.
- `embed_index` owns `chunks` / `embed_state` (schema: migration
  v3) plus the vector backend's `embeddings` table, through
  `EmbedStore`. It *reads* items and extracted text through the
  `CorpusStore` / `ExtractStore` repositories.

`doc_extract`'s registry (`build_extractors`) holds exactly one
strategy in v1 — `pdf` via pypdf — because the live corpus's
harvested files are all PDFs (Google exports and Drive downloads
land as PDF). Formats with no registered strategy (and pointer-only
`external-link` items) resolve to a stored `not-extractable`
outcome; html/docx/slides strategies slot into the registry when
the corpus grows them.
