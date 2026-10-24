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

## Fix (branch `stack/harvest-paging-fix`, 2026-10-24 — UNMERGED)

Implements the requirement above. `McpResourceSource` pages to
exhaustion (probe leg at `limit: 100000` + course × kind cross
product at the default limit, deduped by `source_ref` into
`ListedResources`); a partition at exactly 200 raises
`PayloadError`; all calls use `student_id`. The harvester
tombstones only on an exhaustive listing and records
`partition_union` / `probe_distinct` / `partition_calls` in
`sync_state` (status `partial` on any disagreement or failure).
Regression tests seed a fake adapter past the 200 boundary
(250 items, drifted-title no-tombstone, partition-at-200 loud
failure, disagreement → partial). Gates green on both machines
(`scripts/verify.sh`: 98% unit coverage, mypy strict, LOC cap,
7 system tests; hal9k run from a separate worktree checkout so
the timer's checkout was untouched).

### Live re-run (hal9k, pipeline path, 2026-10-24T13:43Z)

Sync: `total=197 exhaustive=True probe_distinct=197
partition_calls=25` (1 probe + 6 courses × 4 kinds),
`downloaded=26 link_only=11 unchanged=65 failed=95
tombstoned=0`. Failures: sign-in page ×86, HTTP 401 ×7,
HTTP 404 ×2 — the sign-in-walled set is the profile-birth unlock
set, unchanged in kind by this fix.

| Census | Old (99-item corpus) | New | Delta |
|---|---|---|---|
| Distinct resources | 99 | 197 | +98 |
| ok / failed / tombstoned | 67 / 27 / 5 | 102 / 95 / 0 | +35 / +68 / −5 |
| Extracted texts (`item_text`) | 40 (38 extracted + 2 empty) | 64 (59 extracted + 5 empty) | +24 |
| Chunked items / chunks | 38 / 371 | 59 / 629 | +21 / +258 |

Chunk arithmetic, stated plainly: the embed stage wrote 273
new chunks; net is +258 because re-embedded items' old chunks
are replaced (15), per `replace_item_chunks`. The 5 false
tombstones are all restored, now `failed` (sign-in page) —
live in the adapter, unfetchable without profile birth, no
longer tombstoned.

**Timer caveat (read this before relying on the new census):**
the 6-hour timer still runs the old code until this branch is
merged + deployed. Its next run (12:45 CDT 2026-10-24) will
re-truncate to 200 and can re-tombstone items past the boundary.
The re-run above is evidence and a temporary correction, not a
durable fix — durability is the merge/deploy decision.

### Benchmark re-run (materiality fired; set v1.0, labels unchanged)

`onnx-production-v1.0-20261024T135217348051` on the new corpus
(59 items / 629 chunks) vs M6 `...20261023T161119188397`
(38 / 371). All 29 scored queries' expected ids are still ok +
chunked. No hit@3 misses in either run.

| Metric | Before (M6) | After (new corpus) |
|---|---|---|
| Resources hit@1 / hit@3 / MRR | 0.97 / 1.00 / 0.977 | 0.86 / 1.00 / 0.920 |
| Passages hit@1 / hit@3 / MRR | 0.97 / 1.00 / 0.977 | 0.79 / 1.00 / 0.891 |
| Top-chunk hit@1 | 0.97 | 0.79 |
| Separation res / pas | +0.099 / +0.104 | +0.056 / +0.057 |
| Absent-probe mean top (resources) | 0.677 | 0.681 |
| Mean latency | ~30 ms | 34 ms |

Reported plainly: **hit@1 and MRR degraded while hit@3 held.**
The enlarged corpus adds near-neighbors that steal #1 without
displacing the right pointer from the top 3 — e.g. q01
hypotenuse (new item tops at 0.785 over the expected 0.733),
q20 Jamestown (res 1→3, pas 1→2), q07/q08 slope-intercept
(passages 1→2 on new Algebra I keys). This is competition, not
embedding failure: the pointer still lands in the top 3 every
time, but #1 is now a noisier slot. The Bedrock arm was not
re-run (still credential-blocked, same as M5/M6).

### Absent-probe revalidation: NO re-labeling (evidence, not inertia)

All four v1.0 absent probes were re-examined against extracted
text, not just rankings — and all four stay absent, so no v1.1
set was cut:

- **q13 two-step equations:** zero "two-step"/"two step"
  mentions corpus-wide; "2-step" only in the agenda. The new
  passages top (0.719) is a Literal Equations answer key —
  solve-for-a-variable practice, not two-step practice. The
  "Multi-Step Equations" hit is a 469-char review outline
  (section header, no problems). A listing is not material.
- **q27 water cycle:** zero "water cycle" mentions; the new top
  is an ELA literary-essay deck (68 chunks, zero water
  vocabulary in 54,779 chars) — omnibus fallback, not science
  notes. The only "Water Cycling" mention corpus-wide is a
  standards phrase in the year-at-a-glance sheet.
- **q28 cells reaction:** only a standards listing ("Chemical
  Reactions & Matter") in the year-at-a-glance sheet; no
  explanatory material on reactions in cells.
- **q33 cylinder surface area:** the top is the Math 8
  curriculum overview, which names the objective ("solve for
  the lateral and total surface area of … cylinders") but
  teaches nothing; the only new geometry item is Volume of
  cones. Same item topped the probe in v1.0 and was judged
  absent then — consistency holds.

### match_quality re-derivation: constants NOT carried over blindly

Re-derived from the served window (limit 5) over the new
corpus with the current constants, 29 scored / 4 absent:

- **Passages: 4/4 absent weak (holds), scored 26/29 strong
  (was 27/29).** New scored weak: q31 (top 0.6755, margin
  exactly 0.0 — the identical-tutorial-schedules tie). Absent
  tops: 0.7194 / 0.6737 / 0.6601 / 0.6891; absent margins
  0.0099 / 0.0058 / 0.0416 / 0.0098. The decisive arm survives
  by **0.0006**: the q13 absent top is 0.719404 raw against
  the 0.72 threshold (was 0.701, margin 0.019). The 0.72
  constant now sits on a knife-edge — the next corpus growth
  can cross it, and the gate (0.675) no longer separates at
  all (absent max 0.7194 > scored min 0.6755).
- **Resources: 3/4 absent weak (REGRESSED from 4/4).**
  q27 water-cycle now validates **strong**: best 0.6737
  clears the 0.67 gate and the 68-chunk ELA omnibus holds 24
  candidate hits, clearing the breadth arm (≥ 5). The breadth
  arm's assumption — a document truly about a topic matches
  in several places — fails for omnibus decks, which match
  weakly in many places. Scored resources 27/29 strong (was
  28/29); newly weak is q23 (0.6812, 3 hits); q25 stays weak
  (genuinely wrong #1, as in M6).
- **Verdict + follow-up (no silent edit):** `quality.py` was
  deliberately NOT changed — the tutor contract is frozen and
  retuning thresholds is a contract decision, not a harvest
  fix. The follow-up for the coordinator: the resources
  breadth arm needs an omnibus-aware revision (e.g. a score
  floor on counted hits, or a higher gate) and the passages
  decisive threshold needs re-derivation headroom before the
  post-Classroom corpus lands; until then, `resources_for_topic`
  can over-claim on absent topics whose nearest neighbor is a
  large omnibus deck.

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
