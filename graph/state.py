"""
Graph state.

`messages` uses LangGraph's add_messages reducer so raw conversation history
accumulates automatically across turns within a thread (session memory is
just: same thread_id -> same checkpointed state, via MemorySaver in
graph/build.py; a new thread_id is a fresh session, matching the assignment's
"memory resets between sessions" requirement).

The `session_*` fields are OUR chosen structured summary of what's worth
carrying forward beyond raw text -- last resolved location and last activity
hints -- so a follow-up like "what about this evening instead" can be
resolved without re-asking the user, per the assignment's memory requirement.
We carry structured facts rather than re-summarizing raw history with another
LLM call, to keep that continuity deterministic and cheap.

The `turn_*` fields are scratch space for the current turn only; they get
overwritten every turn but are still part of persisted state so they're easy
to inspect/debug and so eval scripts can assert on them directly.
"""

from __future__ import annotations

from typing import Annotated, Optional, TypedDict

from langgraph.graph.message import add_messages


class GraphState(TypedDict, total=False):
    messages: Annotated[list, add_messages]

    # --- carried across turns within a session ---
    session_location_name: Optional[str]
    session_latitude: Optional[float]
    session_longitude: Optional[float]
    session_activity_hints: list[str]
    session_last_sop_id: Optional[str]

    # --- scratch space for the current turn ---
    turn_user_message: str
    turn_location_query: Optional[str]
    turn_activity_hints: list[str]
    turn_reused_prior_location: bool

    turn_error_stage: Optional[str]   # "geocoding" | "forecast" | None
    turn_error_message: Optional[str]

    turn_location_name: Optional[str]
    turn_facts: dict

    turn_numeric_match_ids: list[str]
    turn_fuzzy_match_id: Optional[str]
    turn_primary_sop_id: Optional[str]
    turn_also_relevant_ids: list[str]

    turn_final_answer: str
