"""Match-quality calibration: is the top of a result set real?

Pattern: pure functions over the service's result records — no
store, no provider, no clock — so the wire layer, the tests, and
any offline validation all call the identical code the tutor
contract promises. ``match_quality`` is a property of a RESULT
SET, never of a single hit: it answers the tutor's actual
question — "may I point Sam at this, or must I say I don't have
material on that?"

The rule (a score gate plus a standout arm, per tool):

- Passages: top chunk score >= 0.675 AND the top stands out —
  either the top item's best chunk beats the best chunk of every
  OTHER item in the window by >= 0.02 (margin), or the top score
  itself is decisive (>= 0.72). A window holding a single item
  has no competitor, so the margin arm is satisfied by
  construction.
- Resources: the top item's best_score >= 0.67 AND (its
  chunk_hits >= 5 OR its best_score >= 0.72). An empty result set
  is always "weak".

Derivation (M5 benchmark, run onnx-production-v1.0-20261023,
labeled set v1.0: 29 scored queries + 4 absent-topic probes over
38 items / 371 chunks, bge-small-en-v1.5 cosine scale):

- Absent probes' resources #1 best scores were 0.654 / 0.660 /
  0.660 — and 0.689 for the surface-area probe, whose spurious
  winner held only 4 candidate hits. The 0.67 gate sits between
  the absent maximum genuine score and the scored queries' #1s;
  the hits/override arms are what catch the 0.689 case: a document
  that is truly about a topic either matches in several places,
  or its best chunk is decisively near (>= 0.72).
- Absent probes' passage tops were 0.660–0.701, but their
  distinct-item margins were +0.004 / +0.008 / +0.010 (the fourth,
  +0.041, is caught by the score gate at 0.660): for an absent
  topic the "winner" never stands clear of the field. That is why
  passages need the margin arm and a bare threshold cannot work —
  0.701 (absent) overlaps 0.676–0.690 (genuine).

Validation on the same set (M6, live service + records agree
exactly): all 4 absent probes "weak" on both tools; scored tops
"strong" 28/29 resources (only the science-units query is weak —
its #1 is genuinely the wrong item) and 27/29 passages. The
passages rule earned its decisive arm the hard way: margin alone
scored 24/29, because genuine ties are common in this corpus —
two slope-intercept items tie at 0.81/0.80, two ELA items at
0.727/0.724. The chunk windows showed those ties sit at HIGH top
scores while every absent winner stays <= 0.701, so a top score
>= 0.72 is decisive on its own. The two remaining passage
exceptions are flat-field ties (the DBQ/early-democracy query,
the two identical tutorial schedules at 0.676): no standout at a
modest score. The calibration accepts under-claiming there
deliberately — it costs a clarifying exchange, while
over-claiming sends Sam to the Team Weekly Agenda for a topic the
corpus doesn't have.

Limits (read before reusing these constants): they sit in the
OBSERVED GAPS of one 33-query run on one embedder's cosine scale.
They are corpus- and model-specific: re-derive them from a fresh
benchmark run when the corpus grows materially (the post-Classroom
expansion), when the embedding model changes, or when a benchmark
run's score distributions move. ``worked_examples`` inherits the
passages calibration; its reformulated query ("worked example:
...") was not separately benchmarked — an open calibration item,
recorded in the tutor contract.
"""

from collections.abc import Sequence
from typing import Literal

from school_index.index_mcp.domain import Passage, TopicResource

MatchQuality = Literal["strong", "weak"]

STRONG: MatchQuality = "strong"
WEAK: MatchQuality = "weak"

# Passages arms (derivation in the module docstring).
_PASSAGE_MIN_TOP = 0.675
_PASSAGE_MIN_MARGIN = 0.02
_PASSAGE_HIGH_TOP = 0.72

# Resources arms (derivation in the module docstring).
_RESOURCE_MIN_BEST = 0.67
_RESOURCE_MIN_HITS = 5
_RESOURCE_HIGH_BEST = 0.72


def assess_passages(hits: Sequence[Passage]) -> MatchQuality:
    """Calibrate a passage result set (search_passages' window)."""
    if not hits:
        return WEAK
    top = hits[0]
    if top.score < _PASSAGE_MIN_TOP:
        return WEAK
    best_by_item: dict[str, float] = {}
    for hit in hits:
        current = best_by_item.get(hit.item_id)
        if current is None or hit.score > current:
            best_by_item[hit.item_id] = hit.score
    if top.score >= _PASSAGE_HIGH_TOP:
        return STRONG
    rivals = [
        score for item_id, score in best_by_item.items() if item_id != top.item_id
    ]
    if not rivals:
        return STRONG
    margin = best_by_item[top.item_id] - max(rivals)
    return STRONG if margin >= _PASSAGE_MIN_MARGIN else WEAK


def assess_resources(resources: Sequence[TopicResource]) -> MatchQuality:
    """Calibrate a resources_for_topic result set."""
    if not resources:
        return WEAK
    top = resources[0]
    if top.best_score < _RESOURCE_MIN_BEST:
        return WEAK
    if top.best_score >= _RESOURCE_HIGH_BEST or top.chunk_hits >= _RESOURCE_MIN_HITS:
        return STRONG
    return WEAK
