# Index fixes — plan + checklist (Phase 2, M6 closeout)

Branch: `stack/index-fixes`, stacked on school-index main `67fe0a0`
(directly — NOT on `stack/harvest-paging-fix`, so the dispatcher
can order the merges). Baseline corpus for validation is the
CORRECTED corpus (harvest-paging fix, 197 distinct / 102 ok /
59 chunked / 629 chunks): baseline benchmark run
`onnx-production-v1.0-20261024T135217348051` — resources hit@1
0.86 / hit@3 1.00 / MRR 0.920, passages 0.79 / 1.00 / 0.891.
hit@3 held; hit@1 degraded on near-neighbor competition, not
embedding failure. This milestone must not regress hit@3 or
match_quality, and must NOT retune `quality.py` constants
(frozen tutor contract; see "Not touched" below).

## Ground truth read before designing (2026-10-24, live corpus.db)

- `corpus_items.title` is `TEXT NOT NULL`: an empty title is `""`,
  never NULL. One chunked ok item has `""`: the Algebra I
  slope-intercept blank worksheet (`67c7420b…`, q07's second
  expected item, first chunk starts "Name: … Slope-Interc…").
- Duplicated titles among chunked ok items: `KEY` ×5 (all
  Algebra I, all first chunk `page 1`), `Stations` ×2 (both
  AGENDA, `page 1`), `Drive file` ×2 (both courseless TEAM_SITE,
  `page 1`). Conclusion: **course + first-chunk heading alone
  does NOT disambiguate** — five KEY items share both. The
  task's parenthetical allows an equivalent deterministic
  disambiguator already present in the record, and that is what
  this plan uses: course + first-chunk heading + a snippet of
  the first chunk's text (whitespace-collapsed, truncated).
  The five KEY first-chunk openings diverge within the snippet
  ("Writing Equations in standard form", "Word problems
  involving rate of change", "Key 1 c 2 h…", …); the Stations
  pair diverges at the colony name; the Drive file pair diverges
  immediately (U.S. History year-at-a-glance vs Math 8 overview).
  No new fields are invented: title, course, source_kind,
  deep_link, item_id, chunk heading, chunk text are all already
  on the records the read path carries.
- Chunks carry heading `"page N"` or NULL (markerless text).
  The first chunk is the lowest-ordinal chunk per item.

## (a) Title ambiguity fallback — design

Contract constraint first: `docs/TUTOR-CONTRACT.md` freezes the
wire shapes — passage/resource key sets are exact in
`tests/unit/index_mcp/test_contract.py`. So the fallback does
NOT add a wire field (no `display_title` key). It changes the
*presentation* of the existing `title` value at serialization
time; `item_id` and `deep_link` are untouched, so pointers are
still by deep link.

Where it lives (pure functions, no I/O, no store in the
serializer — coding standard: pure serializer functions):

1. `index_mcp/store.py` — `IndexReadStore.search` additionally
   joins each chunk's item's first chunk (lowest ordinal) and
   carries `first_heading` + `first_text` on `Passage`
   (internal domain fields, never serialized separately).
   `TopicResource` gains the same two fields, copied from the
   item's best hit in `queries.py` aggregation.
2. `index_mcp/presentation.py` (new, pure; pattern: pure
   presentation functions) — `fallback_title(...)` builds the
   fallback string, and `resolve_display_titles(...)` maps a
   presented window to display titles:
   - raw title kept when it is non-empty AND unique within the
     presented window;
   - empty title → fallback always (empty is ambiguous alone);
   - title duplicated within the window → fallback, keeping the
     raw title as a prefix so the result stays recognizable:
     `"<title> — <course> · <heading> · <snippet>"`;
     empty-title form: `"<course> · <heading> · <snippet>"`
     (course absent → `No course`, so the core is never empty;
     heading absent → no heading segment; snippet empty → no
     snippet segment).
   - snippet: first chunk text, whitespace collapsed, first 80
     chars, cut at a word boundary with an ellipsis when cut.
   Scope note (deliberate): duplication is judged within the
   presented window — the list the tutor actually shows. A
   singleton whose title exists elsewhere in the corpus keeps
   its raw title, because its pointer (deep_link/item_id) is
   unambiguous; the fixed ambiguity is collisions *in a
   presented list* plus empty titles. This is stated here so a
   reviewer can disagree with the scope on purpose, not by
   accident.
3. `index_mcp/serializers.py` — `passage_to_dict` /
   `topic_resource_to_dict` accept an optional display-title
   override (default: raw title, so existing single-record
   callers and tests are unaffected); `server.py` resolves the
   window's display titles via `presentation` and passes them
   through. No envelope change, no quality change.
4. Fixtures: unit tests for (i) empty title → fallback with
   course + heading + snippet, (ii) duplicated titles in one
   window → distinct fallback titles, (iii) unique non-empty
   title → raw title unchanged, (iv) contract tests still pin
   the exact wire key sets (title remains a non-empty str in
   the fallback cases — the fallback never emits `""`).

## (b) Why-wrong reformulation prototype — design

Why rules before any model-based approach (estate preference,
stated plainly): the estate spends nothing on model APIs
(Noah's rule); a deterministic rule set is unit-testable,
latency-free, dependency-free, and reversible (identity
fallback); a model reformulator would need an eval harness,
a spend decision, and a fallback story of its own before it
could beat "strip the misconception frame, keep the concept".
Rules are the right v1; a model is a later milestone's
decision, not this one's silent upgrade.

Where it lives: `index_mcp/reformulate.py` (new, pure function;
pattern: strategy registry — an ordered tuple of
(compiled-pattern, builder) rules, first match wins; no
if/else chain for selection). It does NOT change the MCP
tools' envelopes (contract frozen): it is a helper the future
tutor calls *before* querying. Unmatched input returns
unchanged (identity), which is what makes the benchmark
non-regression check meaningful rather than hoped-for.

Rule set v1 (misconception frames only — genuine "why"
questions must NOT match):
- `why is (the) <concept> not just <wrong-method>` /
  `why isn't <concept> just …` / `why can't I just … to get
  <concept>` → the concept phrase (the "not just" frame is
  the misconception; the concept is the query). E.g.
  "why is the hypotenuse not just a+b" → "hypotenuse".
  (The exact emitted form — bare concept vs concept +
  disambiguating words — is settled empirically against the
  live MCP in validation step 4; the rule's *frame* is fixed,
  its *emission* is validated, not assumed.)
- `why is my answer … wrong` / `why is <concept> wrong` →
  the concept phrase extracted from the frame.
- Explicit non-matches, unit-tested: "why did Europeans
  explore and colonize America" (benchmark q19, genuine why),
  plain concept queries, and worked-example phrasing pass
  through unchanged.

Validation (live index MCP on hal9k, token read in-process
from the deploy-run env, never printed/logged/committed):
- misconception cases (hypotenuse variants + one second
  concept present in the corpus, e.g. volume/area) called raw
  vs reformulated through the LIVE gateway route; success =
  reformulated scores `strong` where raw scored `weak`
  (or reformulated top is the expected item with a decisive
  margin where raw was not).
- benchmark non-regression: all 33 queries run through the
  helper path (same SearchService + depth as the benchmark
  runner, helper applied to the query text) and compared
  before/after; ANY regression is a stop, not a tune-it-quietly
  event. Expectation: zero changed queries except deliberate
  rule matches — q19 must be provably unchanged.

## Gates + evidence (both machines)

- `scripts/verify.sh` green on this VM and on hal9k (from a
  separate worktree checkout, so the timer's checkout and the
  live containers are untouched): unit, ≥90% coverage, mypy
  strict, 500-LOC cap, system tests.
- Benchmark re-run on hal9k against the corrected corpus
  (benchmark CLI, `onnx-production` arm): report the full
  before/after table vs the 2026-10-24 baseline above. Note:
  the title fallback changes serialized titles only, not
  ranking, so the raw benchmark (which reads rankings, not
  titles) is expected to reproduce the baseline exactly —
  a difference there would mean the corpus moved, and is
  reported as such, not smoothed over.
- Reformulation before/after table (33 queries, helper path)
  + live-MCP raw-vs-reformulated evidence for the
  misconception cases.

## Not touched (recorded, not fixed here)

- `quality.py` constants (frozen). Two contract-decision
  follow-ups for the coordinator, from the harvest-fix
  re-derivation: (1) resources breadth-arm omnibus regression
  — q27 water-cycle now validates STRONG because the 68-chunk
  ELA omnibus deck clears best ≥ 0.67 and 24 candidate hits
  (breadth arm's "truly about a topic matches in several
  places" assumption fails for omnibus decks); (2) passages
  decisive-threshold headroom — q13 absent top 0.719404 vs the
  0.72 line (0.0006 of headroom). Neither is retuned here.
- Live MCP containers / gateway routes (no production changes);
  pipeline timer (not paused/restarted/reconfigured);
  profile-birth staging (read-only); DECISIONS.md (RAG decision
  is Noah's); D-1 (deferred); ingestion-validator open item
  (REFINEMENT-NOTES.md, not this milestone); corpus_sync
  harvest code (fix milestone owns it — a harvest bug found
  here is reported, not fixed).

## Checklist

- [x] Plan + checklist committed (this file) before code
- [x] `presentation.py` + store first-chunk join + serializers
      override + server window resolution
- [x] `reformulate.py` rule registry + unit tests (matches,
      non-matches incl. q19, identity fallback)
- [x] Title-fallback fixtures: empty title, duplicated title,
      unique title unchanged, fallback never empty
- [x] `scripts/verify.sh` green on this VM
- [x] Branch pushed to ululuth; hal9k worktree checkout;
      `scripts/verify.sh` green on hal9k
- [x] Benchmark re-run on hal9k (onnx-production) + before/after
      table vs 2026-10-24 baseline
- [x] Helper-path benchmark (33 queries) before/after table;
      stop on any regression
- [x] Live-MCP raw vs reformulated evidence (misconception cases)
- [x] Report: branch + tip, gates evidence both machines,
      benchmark tables, reformulation evidence, surprises,
      contract follow-ups noted (not fixed)
