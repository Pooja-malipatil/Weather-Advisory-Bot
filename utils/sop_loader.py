"""
Loads the SOP policy file. Deliberately dumb: no caching surprises, no
hardcoded rule content. Call load_sops(path) fresh whenever you want the
current policy -- this is what makes "add an SOP live, no code changes"
possible: editing sops.yaml and restarting the process is the entire
update mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import yaml

DEFAULT_SOP_PATH = Path(__file__).resolve().parent.parent / "sops" / "sops.yaml"

Severity = Literal["informational", "caution", "warning", "severe"]
SEVERITY_RANK = {"informational": 0, "caution": 1, "warning": 2, "severe": 3}


@dataclass
class NumericClause:
    field: str
    op: str  # one of >, >=, <, <=, ==, !=
    value: float


@dataclass
class SOP:
    id: str
    category: str
    title: str
    condition_type: Literal["numeric", "fuzzy"]
    severity: Severity
    advice: str
    applies_when_activity_hint: list[str]
    numeric_clauses: list[NumericClause] | None = None  # for condition_type == numeric
    situation_description: Optional[str] = None  # for condition_type == fuzzy

    def severity_rank(self) -> int:
        return SEVERITY_RANK.get(self.severity, 0)


def load_sops(path: Path | str = DEFAULT_SOP_PATH) -> list[SOP]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SOP policy file not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_list = yaml.safe_load(f) or []

    sops: list[SOP] = []
    for raw in raw_list:
        condition_type = raw["condition_type"]
        numeric_clauses = None
        situation_description = None

        if condition_type == "numeric":
            clauses = raw.get("condition", {}).get("all", [])
            numeric_clauses = [
                NumericClause(field=c["field"], op=c["op"], value=float(c["value"]))
                for c in clauses
            ]
        elif condition_type == "fuzzy":
            situation_description = raw.get("situation_description", "")
        else:
            raise ValueError(
                f"SOP {raw.get('id')} has unknown condition_type '{condition_type}' "
                "(must be 'numeric' or 'fuzzy')"
            )

        sops.append(
            SOP(
                id=raw["id"],
                category=raw["category"],
                title=raw["title"],
                condition_type=condition_type,
                severity=raw["severity"],
                advice=raw["advice"].strip(),
                applies_when_activity_hint=raw.get("applies_when_activity_hint", []),
                numeric_clauses=numeric_clauses,
                situation_description=situation_description,
            )
        )

    _validate_unique_ids(sops)
    return sops


def _validate_unique_ids(sops: list[SOP]) -> None:
    seen = set()
    for s in sops:
        if s.id in seen:
            raise ValueError(f"Duplicate SOP id found in policy file: {s.id}")
        seen.add(s.id)
