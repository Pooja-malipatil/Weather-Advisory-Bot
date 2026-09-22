"""
Deterministic matching for numeric SOPs, and the resolution policy for what
happens when multiple SOPs match at once.

Design decision (documented per the assignment's explicit ask):
  When more than one SOP matches, we RANK BY SEVERITY and answer with the
  single highest-severity match as the primary policy, but we still surface
  the ids of any other matches at "caution" level or above as secondary
  context in the final answer (not full advice text, just "also relevant:
  SOP-00X"). Rationale: a user asking about cycling in high UV *and* high
  wind should not be told only about UV while a genuine wind hazard is
  silently dropped -- but the reply still needs one clear headline
  recommendation, not several competing paragraphs of prompt-engineered
  advice fighting for attention.

  EXCEPTION: any SOP in the `severe_weather_system` category (SOP-012 in the
  shipped policy) is an explicit *override* per its own advice text: "this
  overrides category-specific advice." If it matches, it is the only SOP
  presented, full stop -- everything else is suppressed. This is checked by
  category name, not by hardcoding "SOP-012", so a newly added SOP in that
  same category on the review call would automatically get override
  behavior with zero code changes.

Numeric evaluation is plain Python -- no LLM involved. This is the
enforcement point for "numbers it reports must be the numbers that actually
came from the API," because the LLM is never asked "does X exceed Y," only
"here is the SOP that already matched, phrase it."
"""

from __future__ import annotations

import operator
from dataclasses import dataclass

from utils.sop_loader import SOP

_OPS = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}

OVERRIDE_CATEGORY = "severe_weather_system"


@dataclass
class MatchResult:
    primary: SOP | None
    also_relevant: list[SOP]  # other matches worth a one-line mention
    all_numeric_matches: list[SOP]
    fuzzy_candidates: list[SOP]  # fuzzy SOPs eligible to be evaluated by the LLM


def _activity_is_relevant(sop: SOP, activity_hints: list[str]) -> bool:
    if not sop.applies_when_activity_hint:
        return True  # no restriction declared -> applies broadly
    hints = {h.lower() for h in activity_hints}
    return any(h.lower() in hints for h in sop.applies_when_activity_hint)


def evaluate_numeric_sop(sop: SOP, facts: dict) -> bool:
    if sop.condition_type != "numeric" or not sop.numeric_clauses:
        return False
    for clause in sop.numeric_clauses:
        value = facts.get(clause.field)
        if value is None:
            return False  # missing data -> cannot claim this clause holds
        if not _OPS[clause.op](value, clause.value):
            return False
    return True


def match_numeric_sops(sops: list[SOP], facts: dict, activity_hints: list[str]) -> list[SOP]:
    matches = []
    for sop in sops:
        if sop.condition_type != "numeric":
            continue
        if not _activity_is_relevant(sop, activity_hints):
            continue
        if evaluate_numeric_sop(sop, facts):
            matches.append(sop)
    return matches


def get_fuzzy_candidates(sops: list[SOP], activity_hints: list[str]) -> list[SOP]:
    return [
        sop
        for sop in sops
        if sop.condition_type == "fuzzy" and _activity_is_relevant(sop, activity_hints)
    ]


def resolve(numeric_matches: list[SOP], fuzzy_matches: list[SOP]) -> MatchResult:
    """Combine numeric + (already LLM-confirmed) fuzzy matches into one decision."""
    all_matches = numeric_matches + fuzzy_matches

    override_matches = [s for s in all_matches if s.category == OVERRIDE_CATEGORY]
    if override_matches:
        primary = max(override_matches, key=lambda s: s.severity_rank())
        return MatchResult(
            primary=primary,
            also_relevant=[],
            all_numeric_matches=numeric_matches,
            fuzzy_candidates=[],
        )

    if not all_matches:
        return MatchResult(
            primary=None, also_relevant=[], all_numeric_matches=numeric_matches, fuzzy_candidates=[]
        )

    ranked = sorted(all_matches, key=lambda s: s.severity_rank(), reverse=True)
    primary = ranked[0]
    also_relevant = [
        s for s in ranked[1:] if s.severity_rank() >= 1 and s.id != primary.id  # caution+
    ]
    return MatchResult(
        primary=primary,
        also_relevant=also_relevant,
        all_numeric_matches=numeric_matches,
        fuzzy_candidates=[],
    )