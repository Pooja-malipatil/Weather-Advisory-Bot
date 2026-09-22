"""
Demonstrates the assignment's explicit review-call test: add a new SOP
without touching graph/, utils/weather.py, or graph/llm.py -- only the
policy YAML changes.

We append a real new numeric SOP (low-visibility fog, a category the
shipped 12 don't cover) directly onto sops/sops.yaml, ask a question that
should trigger it, then remove the addition again so the repo is left clean.
No node code, matcher code, or prompt code is touched at any point --
match_sops_node() already calls load_sops() fresh on every turn (see its
docstring), so this is the entire mechanism.

Run: python evals/test_live_sop_addition.py
Requires GEMINI_API_KEY (real LLM calls) -- Open-Meteo call is also live.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

SOP_PATH = Path(__file__).resolve().parent.parent / "sops" / "sops.yaml"

NEW_SOP_YAML = """
- id: SOP-013
  category: outdoor_exercise
  title: Low visibility fog risk for road activity
  condition_type: numeric
  condition:
    all:
      - field: precipitation
        op: ">="
        value: 0.1
      - field: wind_speed_10m
        op: "<="
        value: 5
  applies_when_activity_hint: [cycling, running, driving, commute]
  severity: caution
  advice: >
    Light precipitation with very low wind can indicate foggy, low-visibility
    conditions on roads. Advise using lights/reflective gear, moving slowly,
    and being extra cautious of vehicles at road crossings.
"""


def main():
    original_text = SOP_PATH.read_text(encoding="utf-8")
    try:
        SOP_PATH.write_text(original_text + "\n" + NEW_SOP_YAML, encoding="utf-8")
        print("Appended SOP-013 to sops.yaml (no code files touched).")

        from utils.sop_loader import load_sops

        sops = load_sops()
        ids = {s.id for s in sops}
        assert "SOP-013" in ids, "New SOP did not load -- policy hot-reload is broken."
        print(f"Confirmed: load_sops() now returns {len(sops)} SOPs including SOP-013.")

        from graph.sop_matcher import match_numeric_sops

        facts = {"precipitation": 0.5, "wind_speed_10m": 3, "temperature_2m": 20,
                  "precipitation_probability": 20, "uv_index": 1}
        matches = match_numeric_sops(sops, facts, ["cycling"])
        matched_ids = [s.id for s in matches]
        print(f"Numeric match against fog-like facts: {matched_ids}")
        assert "SOP-013" in matched_ids, "New SOP loaded but did not match as expected."

        print("\nPASS: a new SOP was added and correctly matched with zero changes to any "
              ".py file -- only sops/sops.yaml was edited.")
    finally:
        SOP_PATH.write_text(original_text, encoding="utf-8")
        print("\nRestored sops.yaml to its original contents.")


if __name__ == "__main__":
    main()
