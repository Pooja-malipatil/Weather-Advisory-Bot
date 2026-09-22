"""
Single shared entry point into the graph. CLI, Streamlit frontend, and the
eval suite all call `ask()` so there's exactly one way requests flow through
the system.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.messages import HumanMessage

from graph.build import get_compiled_graph


@dataclass
class BotResponse:
    answer: str
    primary_sop_id: str | None
    also_relevant_sop_ids: list[str]
    facts: dict
    error_stage: str | None
    location_name: str | None


def ask(thread_id: str, user_message: str) -> BotResponse:
    graph = get_compiled_graph()
    config = {"configurable": {"thread_id": thread_id}}

    result = graph.invoke(
        {
            "messages": [HumanMessage(content=user_message)],
            "turn_user_message": user_message,
        },
        config=config,
    )

    return BotResponse(
        answer=result.get("turn_final_answer", ""),
        primary_sop_id=result.get("turn_primary_sop_id"),
        also_relevant_sop_ids=result.get("turn_also_relevant_ids", []),
        facts=result.get("turn_facts", {}),
        error_stage=result.get("turn_error_stage"),
        location_name=result.get("turn_location_name") or result.get("session_location_name"),
    )
