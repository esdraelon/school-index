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

Status: **M2** — `corpus_sync` (M1) and `doc_extract` (M2) landed.
The harvester pulls Sam's resources through the adapter's MCP
contract into the corpus store (SQLite metadata + original files on
disk), with per-item failure records, tombstones, and idempotent
re-runs; the extractor then turns harvested files into text —
stored in `item_text` with `[[page N]]` page-boundary markers —
under a per-item outcome record. Milestones M3–M6 per
`docs/PROJECT-INDEX.md` in school-assistant.

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

`doc_extract`'s registry (`build_extractors`) holds exactly one
strategy in v1 — `pdf` via pypdf — because the live corpus's
harvested files are all PDFs (Google exports and Drive downloads
land as PDF). Formats with no registered strategy (and pointer-only
`external-link` items) resolve to a stored `not-extractable`
outcome; html/docx/slides strategies slot into the registry when
the corpus grows them.
