from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from graph import llm
from graph.sop_matcher import get_fuzzy_candidates, match_numeric_sops, resolve
from graph.state import GraphState
from utils.sop_loader import load_sops
from utils.weather import ForecastError, GeocodingError, geocode_city, fetch_weather


def parse_query_node(state: GraphState) -> dict:
    """Extract location + activity hints from the latest user message, using
    prior session context to resolve pronoun-like follow-ups."""
    user_message = state["turn_user_message"]

    prior_turns = state.get("messages", [])[:-1]  # exclude the message just added
    summary_lines = []
    for m in prior_turns[-6:]:
        role = "User" if isinstance(m, HumanMessage) else "Bot"
        text = m.content if isinstance(m.content, str) else str(m.content)
        summary_lines.append(f"{role}: {text}")
    prior_summary = "\n".join(summary_lines)

    has_prior_location = bool(state.get("session_location_name"))

    intent = llm.extract_intent(
        latest_message=user_message,
        prior_turns_summary=prior_summary,
        has_prior_location=has_prior_location,
    )

    location_query = intent.get("location_query")
    activity_hints = [
        h for h in (intent.get("activity_hints") or []) if h in llm.ALLOWED_ACTIVITY_HINTS
    ]
    reused_prior = bool(intent.get("reused_prior_location")) and has_prior_location

    if not activity_hints:
        activity_hints = ["outdoor_general", "general"]

    return {
        "turn_location_query": location_query,
        "turn_activity_hints": activity_hints,
        "turn_reused_prior_location": reused_prior,
        "turn_error_stage": None,
        "turn_error_message": None,
    }


def geocode_node(state: GraphState) -> dict:
    """Resolve a location for this turn: a new one from the message, the
    carried-over session location for a follow-up, or an error."""
    location_query = state.get("turn_location_query")
    reused_prior = state.get("turn_reused_prior_location")

    if not location_query and reused_prior and state.get("session_location_name"):
        # Reuse session location without re-hitting the geocoder.
        return {
            "turn_location_name": state["session_location_name"],
            "session_latitude": state["session_latitude"],
            "session_longitude": state["session_longitude"],
        }

    if not location_query:
        return {
            "turn_error_stage": "geocoding",
            "turn_error_message": (
                "I don't have a location for this question yet -- please tell me which "
                "city or place you mean."
            ),
        }

    try:
        resolved = geocode_city(location_query)
    except GeocodingError as exc:
        return {"turn_error_stage": "geocoding", "turn_error_message": str(exc)}

    return {
        "turn_location_name": f"{resolved.name}, {resolved.country}" if resolved.country else resolved.name,
        "session_location_name": f"{resolved.name}, {resolved.country}" if resolved.country else resolved.name,
        "session_latitude": resolved.latitude,
        "session_longitude": resolved.longitude,
    }


def fetch_weather_node(state: GraphState) -> dict:
    from utils.weather import ResolvedLocation

    lat, lon = state.get("session_latitude"), state.get("session_longitude")
    name = state.get("turn_location_name") or state.get("session_location_name") or "the requested location"

    if lat is None or lon is None:
        return {
            "turn_error_stage": "forecast",
            "turn_error_message": "No coordinates available to fetch weather for.",
        }

    try:
        snapshot = fetch_weather(ResolvedLocation(name=name, country=None, latitude=lat, longitude=lon, candidate_count=1))
    except ForecastError as exc:
        return {"turn_error_stage": "forecast", "turn_error_message": str(exc)}

    return {"turn_facts": snapshot.as_fact_dict()}


def match_sops_node(state: GraphState) -> dict:
    sops = load_sops()  # reloaded every turn on purpose: policy is hot-reloadable
    facts = state.get("turn_facts", {})
    activity_hints = state.get("turn_activity_hints", [])
    user_message = state["turn_user_message"]

    numeric_matches = match_numeric_sops(sops, facts, activity_hints)

    fuzzy_candidates = get_fuzzy_candidates(sops, activity_hints)
    fuzzy_candidate_dicts = [
        {"id": s.id, "title": s.title, "situation_description": s.situation_description}
        for s in fuzzy_candidates
    ]
    fuzzy_match_id = llm.pick_fuzzy_sop(user_message, fuzzy_candidate_dicts, facts)
    fuzzy_matches = [s for s in fuzzy_candidates if s.id == fuzzy_match_id] if fuzzy_match_id else []

    result = resolve(numeric_matches, fuzzy_matches)

    return {
        "turn_numeric_match_ids": [s.id for s in numeric_matches],
        "turn_fuzzy_match_id": fuzzy_match_id,
        "turn_primary_sop_id": result.primary.id if result.primary else None,
        "turn_also_relevant_ids": [s.id for s in result.also_relevant],
    }


def compose_answer_node(state: GraphState) -> dict:
    sops = {s.id: s for s in load_sops()}
    primary_id = state.get("turn_primary_sop_id")
    location_label = state.get("turn_location_name") or state.get("session_location_name") or "your location"
    facts = state.get("turn_facts", {})
    user_message = state["turn_user_message"]

    if not primary_id:
        answer = llm.compose_answer(user_message, None, [], facts, location_label)
        return _finalize(answer, primary_id)

    primary = sops[primary_id]
    also_relevant = [sops[i] for i in state.get("turn_also_relevant_ids", []) if i in sops]

    answer = llm.compose_answer(
        user_message,
        {"id": primary.id, "title": primary.title, "severity": primary.severity, "advice": primary.advice},
        [{"id": s.id, "title": s.title, "severity": s.severity} for s in also_relevant],
        facts,
        location_label,
    )
    return _finalize(answer, primary_id)


def _finalize(answer: str, sop_id: str | None) -> dict:
    return {
        "turn_final_answer": answer,
        "session_last_sop_id": sop_id,
        "messages": [AIMessage(content=answer)],
    }


def honest_fallback_node(state: GraphState) -> dict:
    stage = state.get("turn_error_stage")
    detail = state.get("turn_error_message") or "an unknown issue"

    if stage == "geocoding":
        answer = (
            f"I couldn't pin down that location ({detail}). Could you clarify the city/place "
            "name (e.g. add a state or country if it's a common name)? I don't want to guess "
            "at weather for the wrong place."
        )
    else:
        answer = (
            f"I couldn't get live weather data right now ({detail}). I don't want to guess at "
            "conditions I haven't actually checked, so please try again in a moment."
        )

    return {
        "turn_final_answer": answer,
        "messages": [AIMessage(content=answer)],
    }
