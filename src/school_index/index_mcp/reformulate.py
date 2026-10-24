"""Why-wrong query reformulation: misconception phrasing -> concept query.

Pattern: strategy registry — an ordered tuple of (compiled
pattern, builder) rules, first match wins. No if/else chain for
selection: adding a frame means adding a registry entry, and the
order is the precedence. A pure function: same input, same
output, no provider, no store, no clock.

What it is for: a student who phrases a question around their
own wrong method ("why is the hypotenuse not just a+b") is
asking about the CONCEPT, but the negation frame embeds the
wrong method in the query text, and the embedding then scores
the misconception phrasing weakly against the concept material
(the M6 closeout finding this prototype answers). Stripping the
frame and keeping the concept recovers the strong match.

Why rules before any model-based approach: the estate spends
nothing on model APIs; a deterministic rule set is unit-testable
(the frame inventory is small and enumerable), adds no latency
and no new dependency, and fails safe — unmatched input returns
unchanged, so a query the rules do not understand is exactly
the query the tutor would have sent anyway. A model
reformulator would need a spend decision and its own eval
harness first; that is a later milestone's decision, not a
silent upgrade to this helper.

What it is NOT: it does not change any MCP tool envelope (the
tutor contract is frozen) — the future tutor calls this helper
BEFORE querying. It does not rewrite genuine "why" questions:
a rule only fires on a misconception frame (not-just, can't-
just, is-wrong), never on a bare "why did / why is" question.
"""

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from re import Match, Pattern

#: "why is the hypotenuse not just a+b" / "why isn't X just Y".
_NOT_JUST: Pattern[str] = re.compile(
    r"^\s*why\s+(?:is|isn'?t|are|aren'?t)\s+(?:the\s+)?"
    r"(?P<concept>.+?)\s+(?:not\s+just|just)\s+.+\s*$",
    re.IGNORECASE,
)

#: "why can't I just add a and b to get the hypotenuse".
_CANT_JUST: Pattern[str] = re.compile(
    r"^\s*why\s+can(?:not|'?t)\s+(?:you|we|i)\s+just\s+.+?"
    r"\s+to\s+get\s+(?:the\s+)?(?P<concept>.+?)\s*[?.!]*\s*$",
    re.IGNORECASE,
)

#: "why is my answer for the hypotenuse wrong when I added a and b".
_MY_ANSWER_WRONG: Pattern[str] = re.compile(
    r"^\s*why\s+is\s+my\s+answer\s+(?:for\s+)?(?:the\s+)?"
    r"(?P<concept>.+?)\s+wrong\b.*$",
    re.IGNORECASE,
)

#: "why is the hypotenuse wrong" (bare wrong-frame, no answer noun).
_IS_WRONG: Pattern[str] = re.compile(
    r"^\s*why\s+is\s+(?:the\s+)?(?P<concept>.+?)\s+wrong\b.*$",
    re.IGNORECASE,
)


#: Captures that name no topic (a bare pronoun, or the answer
#: noun itself with no topic attached — "why is my answer
#: wrong" names nothing to search for). These yield "" so the
#: caller falls back to the original query instead of searching
#: for "it" or "my answer".
_NO_TOPIC = frozenset({"it", "this", "that", "my answer", "my homework"})


def _concept(match: Match[str]) -> str:
    """The captured concept phrase, punctuation-trimmed."""
    concept = match.group("concept").strip().strip("?.!").strip()
    return "" if concept.lower() in _NO_TOPIC else concept


@dataclass(frozen=True)
class _Rule:
    pattern: Pattern[str]
    build: Callable[[Match[str]], str]


#: Registry order is precedence: the most specific frames first.
#: _MY_ANSWER_WRONG precedes _IS_WRONG because "my answer for X
#: wrong" would otherwise match _IS_WRONG with concept "my answer
#: for X". All builders emit the concept phrase alone — the
#: emission form was validated against the live MCP (see the
#: milestone report), not assumed.
_RULES: Sequence[_Rule] = (
    _Rule(_MY_ANSWER_WRONG, _concept),
    _Rule(_NOT_JUST, _concept),
    _Rule(_CANT_JUST, _concept),
    _Rule(_IS_WRONG, _concept),
)


def reformulate_query(query: str) -> str:
    """Map misconception phrasing to its concept query.

    Returns the original query unchanged when no rule matches —
    the identity fallback that makes benchmark non-regression
    checkable: a query the helper does not understand is sent
    exactly as the student typed it.
    """
    for rule in _RULES:
        match = rule.pattern.match(query)
        if match is not None:
            concept = rule.build(match)
            return concept if concept else query
    return query
