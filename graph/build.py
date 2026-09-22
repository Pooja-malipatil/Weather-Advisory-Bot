"""
The graph, spelled out:

    parse_query --> geocode --[error]--> honest_fallback --> END
                       |
                     [ok]
                       v
                 fetch_weather --[error]--> honest_fallback --> END
                       |
                     [ok]
                       v
                  match_sops
                       |
                       v
                compose_answer --> END

Two real conditional branches (geocode failure, forecast failure) both
funnel into one honest-fallback node, plus the SOP-matched vs no-match
distinction is handled *inside* compose_answer by branching prompts rather
than as a third graph edge -- because both paths still need the same
"real facts must ground the reply" treatment and produce one AIMessage, so
splitting it into two more graph nodes would be a distinction without a
difference. The failure paths, by contrast, are genuinely different from the
happy path (no weather facts exist to compose anything from), which is why
those get real edges.

Session memory: MemorySaver checkpointer, keyed by `thread_id` in the
runtime config. Same thread_id across .invoke() calls = same session with
full state carried forward (raw messages + our structured session_* fields).
A new thread_id is an entirely fresh session, per the assignment's explicit
"memory resets between sessions" scope.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from graph.nodes import (
    compose_answer_node,
    fetch_weather_node,
    geocode_node,
    honest_fallback_node,
    match_sops_node,
    parse_query_node,
)
from graph.state import GraphState


def _geocode_router(state: GraphState) -> str:
    return "honest_fallback" if state.get("turn_error_stage") == "geocoding" else "fetch_weather"


def _weather_router(state: GraphState) -> str:
    return "honest_fallback" if state.get("turn_error_stage") == "forecast" else "match_sops"


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("parse_query", parse_query_node)
    graph.add_node("geocode", geocode_node)
    graph.add_node("fetch_weather", fetch_weather_node)
    graph.add_node("match_sops", match_sops_node)
    graph.add_node("compose_answer", compose_answer_node)
    graph.add_node("honest_fallback", honest_fallback_node)

    graph.set_entry_point("parse_query")
    graph.add_edge("parse_query", "geocode")
    graph.add_conditional_edges(
        "geocode", _geocode_router, {"honest_fallback": "honest_fallback", "fetch_weather": "fetch_weather"}
    )
    graph.add_conditional_edges(
        "fetch_weather", _weather_router, {"honest_fallback": "honest_fallback", "match_sops": "match_sops"}
    )
    graph.add_edge("match_sops", "compose_answer")
    graph.add_edge("compose_answer", END)
    graph.add_edge("honest_fallback", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


_compiled = None


def get_compiled_graph():
    global _compiled
    if _compiled is None:
        _compiled = build_graph()
    return _compiled
