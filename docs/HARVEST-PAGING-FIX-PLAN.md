# Harvest paging fix — plan + checklist

**Milestone:** school-index `stack/harvest-paging-fix`, stacked on
`stack/harvest-paging-req` (d87afc7) over main 67fe0a0. Implements
`docs/HARVEST-PAGING.md` (Phase 0 requirement, 2026-10-24).
**Do NOT merge to main.** Push to `ululuth`.

## The bug (one line)

`McpResourceSource.list_resources` made one default-limit
`get_resources` call with the wrong argument name (`student`,
silently ignored), harvesting a truncated 200-row page as if it
were the world — 98 of 197 distinct resources never harvested,
and 5 live items falsely tombstoned after drifting past the
title-200 boundary.

## Design

### `client.py` — page to exhaustion (Anti-Corruption Layer)

`ResourceSource.list_resources` now returns `ListedResources`
(new frozen dataclass in `corpus_sync/domain.py`):

- `resources`: the deduplicated union, as `tuple[ResourceRef, ...]`
- `probe_distinct`: distinct `source_ref` count of the
  independent leg (below)
- `partition_calls`: how many partition calls were made
- `exhaustive`: True only when the union proves complete

Call sequence, all with the correct `student_id` argument:

1. **Independent leg (probe):** `get_resources` with
   `{"student_id", "limit": 100000}`. If it returns exactly
   100000 rows, raise `PayloadError` — the probe itself may be
   truncated and proves nothing.
2. **Partition discovery from the probe** (never from assumed
   course/kind lists): courses = sorted non-None `course_name`s,
   kinds = sorted non-blank `kind`s.
3. **Partitions:** the course × kind cross product, each call at
   the DEFAULT limit (no `limit` argument), so a partition
   returning exactly 200 is possibly-truncated. Any partition at
   exactly 200 raises `PayloadError` — fail loudly, never silently
   truncate. There is no finer tool partition (no title-range
   param exists), so sub-partitioning past course × kind is not
   available; the loud failure is the specified behavior.
   - Courseless rows are covered: the adapter's course filter
     matches courseless rows under every course, so any
     (course, kind) partition carries the courseless rows of that
     kind. Cross product (not observed pairs) is what guarantees
     this — a courseless kind with no coursed pair is still
     carried by every course's partition of that kind.
   - If the probe has courses but no kinds, or kinds but no
     courses, the degenerate leg applies: kind-only partitions
     (course filter omitted) when there are no courses; an empty
     probe yields zero partitions and an empty union.
   - Blank-kind probe rows cannot be partitioned (the adapter
     rejects an unknown/blank kind) — they surface as a
     union-vs-probe disagreement (partial, no tombstone), never
     as silent coverage. Live data has no blank kinds (audit kind
     counts sum to the full 398), so this is a degraded-visible
     path, not a live gap.
4. **Union:** deduplicate by `source_ref`, first occurrence wins,
   partitions visited in sorted (course, kind) order (deterministic).
5. **Exhaustion proof:** `exhaustive` iff
   `len(union) == probe_distinct` AND no partition hit the limit
   AND the probe did not hit its large limit. Disagreement in
   either direction (adapter drift between legs, unpartitionable
   rows) means not-exhaustive.

### `harvester.py` — tombstone guard + visible degradation

- Tombstone ONLY when `listed.exhaustive` is True. A
  non-exhaustive run tombstones 0 and says so.
- `last_status`: `"ok"` iff `failed == 0` AND exhaustive, else
  `"partial"`. A source raising still aborts the run before any
  tombstoning (unchanged behavior).
- `sync_state.counts` gains `partition_union` (= total distinct),
  `probe_distinct`, `partition_calls` — the coverage proof lives
  in the store, not just in logs. `SyncReport` gains `exhaustive`,
  `probe_distinct`, `partition_calls` (defaults = not-proven, so
  hand-built reports never claim coverage they lack).
- `total` is now the distinct-union count (previously the raw
  payload row count, duplicates included). The in-run dedup loop
  stays as defense (a repeated `source_ref` still counts
  unchanged, never re-fetched).

### Tests — coverage proof in fixtures, not under the cap

New `tests/unit/corpus_sync/test_paging.py` with a
`FakeAdapterTransport` that mimics the adapter contract exactly
(title order, limit-after-filters with default 200, course-filter
union semantics for courseless rows, kind filter, `student_id`
scoping, large-limit probe):

- **>200 recovery:** 250 seeded resources across courses/kinds;
  the harvest recovers all 250 distinct refs.
- **Drifted-past-boundary no-tombstone:** an item seeded in the
  store, then a run whose title order puts it past position 200
  (it would have tombstoned under the old single-call code);
  assert status stays OK and `tombstoned == 0`.
- **Partition-at-200 fails loudly:** a single course × kind
  partition of ≥200 rows raises `PayloadError`, and the harvester
  aborts without tombstoning.
- **Disagreement degrades visibly:** a source whose union and
  probe disagree yields `last_status == "partial"` and zero
  tombstones.
- **Argument names:** every transport call carries `student_id`
  (never `student`).

Updated fakes: `test_harvester.py` / system `FixtureSource` /
`test_client.py` move to the `ListedResources` port shape;
`test_client.py`'s call assertion becomes
`{"student_id": "sam"}` (plus probe/partition legs for the paging
tests' transports). The in-process FastMCP round-trip test keeps
a fixture tool whose signature accepts the new arguments.

### Live re-run (hal9k, via the pipeline path)

1. From a hal9k checkout of this branch (separate from the
   timer's checkout — the timer, containers, and gateway routes
   are NOT touched), run `deploy/run_pipeline.py`
   (sync → extract → embed) against `~/school-index-data`.
   Token read in-process from the adapter deploy `.env`, never
   printed/logged/committed.
2. Confirm: the 5 false tombstones are no longer tombstoned,
   the 98 missed resources arrive; report the new census
   (totals, ok/failed, extracted, chunks) and the delta vs
   99 / 67 ok / 38 extracted / 371 chunks.
3. **Timer caveat (report prominently):** the 6h timer still runs
   the old code until this fix is merged + deployed, so its next
   run will re-truncate and can re-tombstone. The live re-run is
   evidence and a temporary correction, not a durable fix — the
   durable fix is the merge/deploy decision, which belongs to
   Noah via the coordinator.

### Benchmark re-run (materiality fired: corpus ~doubles)

1. Run `onnx-production` against the new corpus with the
   ORIGINAL v1.0 labeled set first — pure corpus-effect
   before/after (29 scored queries; expected ids are stable
   content hashes, unchanged by the harvest fix).
2. **Absent-probe revalidation** (q13, q27, q28, q33 — labeled
   absent against the truncated corpus): for each, inspect the
   enlarged corpus's extracted text for genuinely topical
   material (title/text read, not just "the system returned it").
   Only genuinely topical items become expected ids; a probe
   with no topical material stays absent, with the evidence
   noted. Labels change only on topical evidence, never to
   flatter the ranking.
3. Bump the labeled set to v1.1 (new census date 2026-10-24,
   re-labeled probes only) on this branch, run `onnx-production`
   again with v1.1, and report both runs.
4. **match_quality re-derivation:** from the new run's stored
   outcomes, compute the scored/absent score distributions and
   check the current constants (passages 0.675 / margin 0.02 /
   decisive 0.72; resources 0.67 / hits 5 / decisive 0.72) against
   the NEW observed gaps, plus the strong/weak validation counts
   under the current rule. Constants in `quality.py` change ONLY
   on a separate decision — this milestone reports the
   re-derivation and its verdict; it does not silently retune
   the frozen tutor contract. If the new gaps contradict the
   current constants, that is a reported finding with a
   follow-up, not a quiet edit.
5. Report before/after tables (resources + passages hit@1 /
   hit@3 / MRR, match_quality validation numbers). Degradation,
   if any, is reported plainly — no tuning to hide it.

### Docs on this branch (school-index repo only)

- `README.md`: census + benchmark-results sections to the new
  corpus (old numbers kept labeled as the 99-item corpus).
- A short fix note (this file's design summary) appended to
  `docs/HARVEST-PAGING.md` as the "Fix" section.
- NO commits to school-assistant (report numbers to the
  coordinator; it routes doc updates).

## Out of scope (recorded, not absorbed)

- Live MCP containers / gateway routes / the pipeline timer
  (no production changes of any kind).
- Profile-birth staging (read-only).
- `DECISIONS.md` (RAG decision is Noah's); D-1 (deferred).
- Ingestion-validator open item (REFINEMENT-NOTES.md) — follow-up.
- Title-fallback / why-wrong fixes (separate held milestone).

## Checklist

- [x] Plan + checklist written before code (this file)
- [x] `ListedResources` port + paging client (probe, partitions,
      loud failure at 200, `student_id` everywhere)
- [x] Harvester tombstone guard + `sync_state` coverage proof +
      `SyncReport` coverage fields + CLI print
- [x] Paging regression tests (>200 recovery, drifted-boundary
      no-tombstone, partition-at-200 loud failure, disagreement
      → partial + no tombstone, argument names)
- [x] Existing fakes migrated to the new port shape
- [x] Gates green on this VM: `scripts/verify.sh`
      (unit, ≥90% coverage, mypy strict, 500-LOC cap, system)
- [x] Branch pushed to `ululuth`
- [x] Gates green on hal9k (same `scripts/verify.sh`, branch checkout)
- [x] Live pipeline re-run on hal9k (sync → extract → embed);
      5 tombstones restored + 98 missed arrived; new census + delta
- [x] Benchmark v1.0 re-run on new corpus (before/after table)
- [x] Absent-probe revalidation with topical evidence; set v1.1
      — outcome: NO re-labeling (all 4 probes stay absent, see
      docs/HARVEST-PAGING.md fix section); no v1.1 was cut
- [x] Benchmark v1.1 re-run; match_quality re-derivation + verdict
      — no v1.1 run (labels unchanged); re-derivation done against
      the v1.0 re-run (see fix section): passages rule holds 4/4
      absent weak but the 0.72 decisive margin collapsed to 0.0006;
      resources rule REGRESSED (q27 absent now validates strong —
      omnibus breadth arm). Constants NOT changed: follow-up for
      the coordinator, not a silent contract edit
- [ ] Repo docs updated (README census/benchmark, HARVEST-PAGING fix note)
- [ ] Final report: branch + tip, gates evidence both machines,
      census old vs new, benchmark before/after, revalidation +
      re-derivation outcomes, surprises, timer caveat
