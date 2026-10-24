# Index fixes — validation record (2026-10-24)

Branch `stack/index-fixes` (on main `67fe0a0`). `quality.py`
constants UNCHANGED (frozen contract). Live MCP containers and
gateway routes untouched — live-MCP evidence below is
client-side only (raw vs reformulated queries through the
existing route); the title fallback is validated at the
worktree level (unit + contract tests), not against the live
container, because the live container still serves pre-fix code
by design (no production changes in this milestone).

## Benchmark re-run (hal9k, corrected corpus, 59 items / 629 chunks)

Run `onnx-production-v1.0-20261024T141134232660` (benchmark CLI,
`onnx-production` arm) vs the harvest-fix baseline
`...20261024T135217348051`:

| Metric | Baseline (corrected corpus) | Re-run (this branch) |
|---|---|---|
| Resources hit@1 / hit@3 / MRR | 0.86 / 1.00 / 0.920 | 0.86 / 1.00 / 0.920 |
| Passages hit@1 / hit@3 / MRR | 0.79 / 1.00 / 0.891 | 0.79 / 1.00 / 0.891 |
| Mean latency | 34 ms | 38 ms |

Exact reproduction, as designed: the title fallback changes
serialized titles only, never rankings, so a difference here
would have meant the corpus moved. It did not.

## Helper-path benchmark (33 queries, reformulate applied first)

RAW and HELPER identical on every metric (resources
0.86/1.00/0.920, passages 0.79/1.00/0.891), zero queries changed
by the helper, zero match_quality diffs at the served window
(limit 5), zero resource-rank regressions. This is identity by
construction + verified, not by luck: no benchmark query matches
a misconception frame — including q19 ("why did Europeans
explore and colonize America"), which the rules must and do
leave unchanged. The helper's value is therefore proven live
(below), not on the benchmark set, which contains no
misconception phrasing.

## Live-MCP validation (gateway route, limit 5, token in-process)

Scores are the served window's top (passages) / #1 (resources);
quality is the frozen `quality.py` assessment of that window.

| Case (raw → reformulated) | Raw pas / res | Reformulated pas / res |
|---|---|---|
| why is the hypotenuse not just a+b → hypotenuse | strong 0.6941 / **weak** 0.6941 (2 hits) | strong 0.7373 / **strong** 0.7373 (2 hits) |
| why can't you just add a and b to get the hypotenuse → hypotenuse | **weak** 0.6879 / **weak** 0.6879 | **strong** 0.7373 / **strong** 0.7373 |
| why is my answer for the hypotenuse wrong when I added a and b → hypotenuse | strong 0.7407 / strong 0.7407 | strong 0.7373 / strong 0.7373 (same item, no regression) |
| why is the area of a triangle not just base times height → area of a triangle | **weak** 0.6821 (3D Shapes) / **weak** 0.6821 | **strong** 0.7573 (Area/Perimeter) / **strong** 0.7501 (see pointer note) |
| why is the perimeter not just length plus width → perimeter | strong 0.6846 / **weak** 0.6846 (2 hits) | strong 0.7203 / **strong** 0.7203 (decisive arm) |
| why is the volume not just length times width → volume | weak 0.6433 / weak 0.6433 | weak 0.6666 / weak 0.6666 (right item tops both) |
| why is the area not just length plus width → area | strong 0.7038 / weak 0.7038 (2 hits) | strong 0.7059 / weak 0.7059 (2 hits) |

Reading, honestly:

- **The criterion is met where the concept is specific:**
  hypotenuse (both frames), triangle area, and perimeter all
  go weak → strong on at least one tool, and the reformulated
  top is the concept's item. Perimeter's resources flip is the
  decisive arm (0.7203 ≥ 0.72) rescuing a 2-chunk item that
  cannot clear the breadth arm — the same arm that stays shut
  for area (0.7059 < 0.72, 2 hits).
- **Bare generic nouns are the prototype's limit:** "volume"
  improves the score (+0.023) but stays below the 0.675 gate,
  and "area" barely moves. The failure is safe (weak ⇒ the
  tutor says "no material", never over-claims), but it bounds
  the claim: rule emission is the bare concept phrase, and for
  a one-word generic concept that is sometimes not enough to
  clear the gate. A future milestone could emit a more specific
  concept phrase; that is tuning, and it was not done quietly
  here.
- **Pointer note (triangle area):** reformulated passages top
  is Area/Perimeter (the right item) but resources #1 is Volume
  of cones (0.7501, 3 hits) — same band as the passages winner,
  lifted by breadth. Quality "strong" does not guarantee the
  resources pointer is the passages winner; band + breadth can
  lift a near-tie multi-hit item over the single best chunk.
  Recorded as an observation, not fixed here (ranking is M6
  scope, and quality constants are frozen).

## Contract follow-ups (NOT fixed here, for the coordinator)

1. Resources breadth-arm omnibus regression (harvest-fix
   re-derivation): q27 water-cycle validates STRONG because
   the 68-chunk ELA omnibus deck clears best ≥ 0.67 and 24
   candidate hits. The breadth arm's "truly about a topic
   matches in several places" assumption fails for omnibus
   decks. Needs a contract decision (e.g. a score floor on
   counted hits, or a higher gate).
2. Passages decisive-threshold headroom: q13 absent top
   0.719404 vs the 0.72 line — 0.0006 of headroom. The next
   corpus growth can cross it; re-derivation is due before
   the post-Classroom corpus lands.
3. Small-item resources ceiling (observed above): a 2-chunk
   item can only clear resources via the decisive arm
   (best ≥ 0.72); below that, no reformulation helps, because
   the weakness is chunk_hits, not score. Whether that is the
   intended behavior for thin-but-correct items is a contract
   question, not a bug found here.
