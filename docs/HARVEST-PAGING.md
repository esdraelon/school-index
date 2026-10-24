# Harvest paging requirement — corpus_sync pages to exhaustion

**Status: requirement (Phase 0, 2026-10-24).** This doc states what
the M1 harvest got wrong, what the fix milestone must do, and what
evidence closes it. It changes no code; the fix is a later
milestone. Until that milestone lands, the 6-hour timer harvest
keeps running the truncated call — see "Live damage" below.

## The contract (adapter side; full text in the adapter repo's
`deploy/README.md`, "MCP collection contract")

- `get_resources`: default limit **200**, no cursor/offset,
  ordered by title ascending, no time-window params. Truncation
  is silent: a bare list, no total, no `has_more`. A page of
  exactly 200 is indistinguishable from a complete result.
- `get_resources` takes `student_id`, not `student`: an unknown
  argument is silently ignored (verified live 2026-10-24:
  `{"student": "sam"}` returned the unscoped default 200).
- Paging without a cursor means partitioning: `course_name` ×
  `kind`, deduplicated by `source_ref` (`student` + `url`). Note
  the course filter's union semantics — courseless rows match
  every course filter, so partition counts double-count them and
  only the deduplicated union is comparable to a total.

## Verdict: the M1 harvest was INCOMPLETE

Live gateway probe, 2026-10-24 (MCP client on hal9k, token
in-process; adapter store as scraped that morning):

| Measure | M1 harvest / corpus | Adapter truth | Gap |
|---|---|---|---|
| `get_resources` payload rows | 200 (single default call) | 398 rows (`limit: 100000`) | 198 rows never seen |
| Distinct resources (`student` + `url`) | 99 in `corpus_items` | 197 | **98 distinct URLs never harvested (49.7%)** |
| Corpus rows by status | ok 67, failed 27, tombstoned 5 | — | — |
| Chunks / extracted texts | 371 / 40 rows | — | benchmark census (2026-10-23) said 38 searchable items |
| Corpus refs no longer in adapter | 0 | — | nothing genuinely vanished |

### Live damage, not just omission

- **False tombstones.** All 5 tombstoned corpus items sit at
  adapter title-positions 200–204 — the first five rows past the
  cap. They were harvested while inside the first 200, drifted
  past the boundary as titles changed, and the timer harvest then
  tombstoned them as "vanished" although they are still live.
  Tombstoning on a capped page is the corruption mode this
  requirement exists to kill.
- **The timer perpetuates it.** The latest `sync_state` run
  (2026-10-24T11:45Z) still reports `total: 200` — every 6-hour
  run re-truncates and can tombstone the next drifted items.
- **Per-course skew.** The missed 98 are not uniform: by the
  retained adapter row, Algebra I 37, Math 8 47, courseless 14
  (multi-course URLs make per-course attribution indicative,
  not exact — the URL-level gap of 98 is the robust number).
  Adapter row counts by course run 6–85; kind counts on the full
  set are TUTORIAL 198, LINK 112, DOCUMENT 56, NOTES 32 — the
  M0 README figures (102/57/21/20) summed to the truncated 200
  and are superseded by this doc (README corrected alongside).
- **Drivers are truncated too, by the same rule** (not
  corpus_sync's path, but the same standing rule): default
  `get_drivers` returns 200 items ending 2026-09-22, against a
  true 562 (due 2026-04-26 → 2027-04-23). Monthly windows
  Aug 2026 – Jun 2027 sum to only 535 — 27 drivers are due
  Apr–May 2026, so even a school-year window does not exhaust
  drivers; windows must span the observed min/max.

## Requirement (fix milestone scope)

1. **Page to exhaustion.** `McpResourceSource.list_resources`
   partitions `get_resources` by `course_name` × `kind`
   (student-scoped with the correct `student_id` argument),
   deduplicates by `source_ref`, and asserts every partition
   returns fewer than the limit — a partition returning exactly
   200 subdivides (by kind, then by title-range discipline the
   milestone defines) until the assertion holds.
2. **Coverage proof in `sync_state`.** Each run records the
   partition-union distinct count AND an independent-leg total
   (explicit-large-limit probe count), and the run is `partial`
   (not `ok`) while the two disagree. The proof, not the call
   count, is what a later run trusts.
3. **Tombstone guard.** `tombstone_missing` fires only against
   the exhaustive union. A ref absent from a capped page is
   "unproven", never "vanished".
4. **Fix the argument name.** `{"student": ...}` →
   `{"student_id": ...}`. Harmless today (all 398 adapter rows
   are sam-visible, so the unscoped call changed nothing), but
   it is a silent contract violation waiting for a second
   student's rows to appear.
5. **Coverage test.** A seeded-adapter regression test with
   > 200 resources straddling the title boundary: the harvest
   must recover all of them, and a drifted-past-the-boundary item
   must NOT tombstone. A fixture under the cap proves nothing.
6. **Re-run order.** Re-run the harvest, then extract + embed
   deltas. Re-run the benchmark ONLY on the materiality rule —
   and the corpus roughly doubles (99 → 197 distinct), so the
   rule fires: re-run the benchmark, re-validate the labeled
   set's absent probes (q13/q27/q28/q33 were labeled absent
   against the truncated corpus, and the missed items' content
   is unextracted, so their absent labels are unproven until
   the enlarged corpus is extracted), and re-derive the
   `match_quality` calibration constants against the enlarged
   corpus. The M5/M6 numbers stand for the 99-item corpus only.

## Open item — agent-at-ingestion validation (NOT scoped work)

From school-assistant `docs/REFINEMENT-NOTES.md` (Noah,
2026-10-24; consideration only): an agent at ingestion could
validate soft formatting drift that deterministic parsers pass
silently. Two placements touch this package directly:

- a `doc_extract` post-pass flagging markup artifacts in
  extracted text (HTML fragments, SVG leftovers, encoding
  mojibake) and structure deviating from known deck templates;
- a `corpus_sync` change gate validating a new/changed item
  against the previous version's shape before overwrite,
  quarantining shape drift instead of overwriting.

(A third placement — validator review of low-confidence email
extractions before the events store — belongs to the calendar
collectors, not this package.) Open questions, unsettled:
sampling vs full pass; quarantine-vs-flag thresholds; whether
the validator only flags/quarantines or also writes corrections
(flagging, with the verdict recorded as provenance on the stored
item, is the safer default). If the fix milestone's plan would
absorb this, it records it as a follow-up instead. This section
authorizes no work.
