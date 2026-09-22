"""
Every LLM call in this system is one of exactly three narrow jobs:

  1. extract_intent   -- turn free text into {location_query, activity_hints}
                          drawn from a fixed vocabulary. No advice, no facts.
  2. pick_fuzzy_sop    -- given ONLY the fuzzy-SOP candidates' ids + situation
                          descriptions (never the numeric SOPs, never advice
                          text of other SOPs) plus the real fetched numbers,
                          return one sop_id from the closed list, or NO_MATCH.
                          The model cannot invent an id: we validate whatever
                          it returns against the candidate id set and treat
                          anything else as NO_MATCH.
  3. compose_answer    -- given ONE resolved SOP's advice text + the real
                          fetched numbers, write the reply. The prompt
                          explicitly forbids introducing any number that
                          isn't in the provided fact dict, and forbids citing
                          any policy other than the one provided.

This separation is what makes "the bot only composes language, it doesn't
decide facts" checkable: jobs 1 and 2 never see or invent weather numbers to
act on beyond what's handed to them, job 3 never decides which policy
applies, and NONE of them do numeric threshold comparison -- that happens in
graph/sop_matcher.py in plain Python.

Model: Google Gemini via the free tier (Google AI Studio API key, no
billing required). Uses the unified `google-genai` SDK. Swapping providers
again later only means editing this file -- graph/nodes.py calls these three
functions by name and never touches the underlying SDK.
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

from google import genai
from google.genai import types
from google.genai import errors as genai_errors

MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
# Used only if MODEL_NAME is overloaded (503) for every retry attempt.
# A more established, heavily-provisioned model is less likely to be
# simultaneously overloaded, so this is a "keep the app answering" safety
# net, not a quality upgrade or downgrade choice.
FALLBACK_MODEL_NAME = os.environ.get("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash")

ALLOWED_ACTIVITY_HINTS = [
    "exercise", "cycling", "running", "hiking", "sports", "outdoor_general",
    "motorbike", "scooter", "two_wheeler",
    "travel", "commute", "driving", "walking", "boating", "fishing", "coastal",
    "elderly", "children", "infant", "senior_citizen", "vulnerable",
    "pet", "dog", "dog_walk", "walk_dog",
    "picnic", "park", "leisure", "family_outing", "outdoor_leisure",
    "general",
]

_client: Optional["genai.Client"] = None

# Retry policy: only for genuinely transient server-side overload (5xx).
# Quota/auth/bad-request errors (4xx) are NOT retried -- sleeping and
# retrying a 429 "resource exhausted" just burns the user's wait time for
# an error that won't resolve itself in seconds. Those fail immediately
# with a clear message instead.
_MAX_RETRIES = 3
_BASE_BACKOFF_SECONDS = 1  # 1s, 2s, 4s -> ~7s worst case, not ~31s


class LLMQuotaError(RuntimeError):
    """Raised immediately (no retry) when the API key is out of quota or invalid."""


def _get_client() -> "genai.Client":
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to your .env file (see .env.example). "
                "Get a free key at https://aistudio.google.com/apikey"
            )
        # TEMP DEBUG -- remove once the auth issue is confirmed fixed.
        print(f"DEBUG: GEMINI_API_KEY length={len(api_key)}, starts_with={api_key[:10]!r}, ends_with={api_key[-6:]!r}")
        _client = genai.Client(
            api_key=api_key,
            # Per-request timeout, not a retry budget -- keep this reasonable
            # so a single hung request can't stall the whole turn.
            http_options=types.HttpOptions(timeout=20_000),
        )
    return _client


def _call_model(model: str, system: str, user: str, max_tokens: int) -> str:
    """Retry loop for ONE model. Raises LLMQuotaError immediately on 4xx
    (never worth retrying), or the last ServerError once retries on this
    model are exhausted (caller decides whether to fall back)."""
    client = _get_client()
    last_exc = None

    for attempt in range(_MAX_RETRIES):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    max_output_tokens=max_tokens,
                    temperature=0.2,
                ),
            )
            result = (resp.text or "").strip()
            if not result:
                print(
                    "DEBUG: empty response. finish_reason:",
                    resp.candidates[0].finish_reason if resp.candidates else "no candidates",
                )
            return result

        except genai_errors.ClientError as e:
            # 4xx: quota exhausted (429), bad/expired key (401/403), bad
            # request (400). None of these get better by waiting a few
            # seconds and retrying the same key -- fail fast with a clear
            # message so the caller (and the user) knows to check the key
            # or quota rather than sitting through a silent multi-second
            # retry loop that was never going to succeed. A different
            # model won't fix an auth/quota problem either, so this is
            # not caught by the fallback-model logic in _call().
            status = getattr(e, "code", None) or getattr(e, "status_code", None)
            if status == 429:
                raise LLMQuotaError(
                    "Gemini API quota exceeded for this key. Switch GEMINI_API_KEY to a key "
                    "with remaining quota, or wait for the quota window to reset."
                ) from e
            raise LLMQuotaError(f"Gemini API rejected the request ({status}): {e}") from e

        except genai_errors.ServerError as e:
            # 5xx: genuinely transient overload -- worth a short, capped retry.
            last_exc = e
            if attempt < _MAX_RETRIES - 1:
                wait = _BASE_BACKOFF_SECONDS * (2 ** attempt)
                print(f"{model} overloaded (attempt {attempt + 1}/{_MAX_RETRIES}), retrying in {wait}s...")
                time.sleep(wait)

    raise last_exc


def _call(system: str, user: str, max_tokens: int = 500) -> str:
    """Tries MODEL_NAME first (with its own retries). If every attempt on
    that model fails with a transient ServerError (5xx overload), falls
    back to FALLBACK_MODEL_NAME once, with its own retries too. A 4xx from
    either model raises LLMQuotaError immediately, unchanged from before --
    switching models never fixes an auth or quota problem."""
    try:
        return _call_model(MODEL_NAME, system, user, max_tokens)
    except genai_errors.ServerError as e:
        if FALLBACK_MODEL_NAME and FALLBACK_MODEL_NAME != MODEL_NAME:
            print(f"{MODEL_NAME} still overloaded after retries, falling back to {FALLBACK_MODEL_NAME}...")
            return _call_model(FALLBACK_MODEL_NAME, system, user, max_tokens)
        raise


def extract_intent(
    latest_message: str,
    prior_turns_summary: str,
    has_prior_location: bool,
) -> dict:
    """Returns {"location_query": str|None, "activity_hints": [str,...], "reused_prior_location": bool}"""

    system = f"""You extract structured intent from a user's message to a weather-advisory bot.
Return ONLY a JSON object, no prose, no markdown fences, matching exactly this shape:
{{"location_query": <string or null>, "activity_hints": [<zero or more strings from the allowed list>], "reused_prior_location": <true or false>, "is_future_request": <true or false>}}

Rules:
- location_query: a city/place name mentioned in THIS message. If none is mentioned in this
  message, set it to null (do not guess or reuse a previous one yourself).
- reused_prior_location: set true if this message clearly continues a prior conversation about
  a location without repeating its name (e.g. "what about this evening instead", "is it still bad
  now"), AND {"a prior location IS available" if has_prior_location else "there is NO prior location available, so this must be false"}.
- is_future_request: set true if the user is asking about a time meaningfully LATER than right now
  (e.g. "tomorrow", "this weekend", "next Tuesday", "tonight" if it's currently daytime). Set false
  if the question is about right now / today in general, with no specific future time named.
- activity_hints: choose only from this exact list, pick every tag that reasonably applies, and
  return an empty list if nothing fits: {ALLOWED_ACTIVITY_HINTS}
  Map naturally: "bike"/"cycle"/"cycling" -> "cycling"; "motorbike"/"bike ride" (motorized) ->
  ["motorbike","two_wheeler"]; "scooter" -> ["scooter","two_wheeler"]; "walk the dog" ->
  ["dog_walk","walking"]; "picnic"/"park"/"hang out outside" -> ["picnic","park","leisure"] as
  fitting; a general/unclear outdoor question with no better fit -> ["outdoor_general","general"].

Conversation context so far (for reference only, do not extract location/activity from this,
only from the LATEST message below):
{prior_turns_summary or "(no prior turns)"}
"""
    user = f"Latest message: {latest_message}"
    raw = _call(system, user, max_tokens=300)
    return _safe_json(
        raw,
        fallback={
            "location_query": None,
            "activity_hints": [],
            "reused_prior_location": False,
            "is_future_request": False,
        },
    )


def pick_fuzzy_sop(
    user_message: str,
    fuzzy_candidates: list[dict],
    facts: dict,
) -> Optional[str]:
    """fuzzy_candidates: list of {"id","title","situation_description"}. Returns a sop_id or None."""

    if not fuzzy_candidates:
        return None

    candidate_ids = {c["id"] for c in fuzzy_candidates}
    candidates_text = "\n\n".join(
        f"id: {c['id']}\ntitle: {c['title']}\nsituation_description: {c['situation_description']}"
        for c in fuzzy_candidates
    )

    system = f"""You are matching a user's question to AT MOST ONE policy (SOP) from a fixed list.
You may ONLY select an id that literally appears below. If none genuinely fits the actual
situation described by the real weather facts and the user's question, respond with exactly:
NO_MATCH

Do not guess generously -- a policy only applies if the situation it describes is actually
present in the facts given, not merely "vaguely related." Respond with ONLY the sop id (e.g.
"SOP-010") or the literal string NO_MATCH. No other text.

Candidate policies:
{candidates_text}

Real weather facts for this location right now (these are the ONLY facts that exist; do not
assume anything beyond them):
{json.dumps(facts, indent=2)}
"""
    user = f"User's question: {user_message}"
    raw = _call(system, user, max_tokens=20).strip()

    if raw in candidate_ids:
        return raw
    return None


def compose_answer(
    user_message: str,
    sop: Optional[dict],
    also_relevant: list[dict],
    facts: dict,
    location_label: str,
    is_future_request: bool = False,
) -> str:
    """sop: {"id","title","severity","advice"} or None for the no-guidance path."""

    future_caveat = (
        "\nIMPORTANT: The user asked about a FUTURE time (e.g. tomorrow/later), but the only "
        "data you have is the CURRENT weather snapshot below -- this system does not fetch "
        "multi-day forecasts. You MUST explicitly tell the user these are current conditions, "
        "not a forecast for the time they asked about, and that they should check again closer "
        "to that time. Do not imply these numbers describe the future."
        if is_future_request
        else ""
    )

    if sop is None:
        system = f"""You are a weather-advisory bot. No written policy (SOP) covers this question.
Say so plainly and kindly in 1-3 sentences. Do NOT invent safety advice of your own. You may
briefly restate the real weather facts you were given (nothing else), and you may suggest the
user rephrase or ask about a specific activity if that would plausibly be covered.
Never claim a policy exists if it doesn't.{future_caveat}"""
        user = (
            f"Location: {location_label}\n"
            f"Real weather facts: {json.dumps(facts, indent=2)}\n"
            f"User's question: {user_message}"
        )
        return _call(system, user, max_tokens=250)

    also_text = (
        "\n".join(f"- {s['id']}: {s['title']} (severity: {s['severity']})" for s in also_relevant)
        or "(none)"
    )

    system = f"""You are a weather-advisory bot. Compose a reply to the user based STRICTLY on the
one policy (SOP) provided below and the real weather facts provided below.

Hard rules:
- Only use numbers that literally appear in the "Real weather facts" JSON. Never state a number,
  even an approximate one, that isn't there. If a fact needed to fully answer is null/missing,
  say that specific fact is unavailable rather than guessing it.
- Your advice must reflect the SOP's "advice" text below -- do not add extra hazards or
  reassurances the SOP doesn't mention, and do not contradict it.
- End your reply on its own short line citing the policy like this: "(Policy: SOP-00X)" using
  the real id given below -- never invent an id.
- If "Also relevant" policies are listed, you may add ONE brief sentence noting they're also
  relevant by id, but do not restate their full advice text.
- Be concise, warm, and direct. This is a chat reply, not a report.{future_caveat}

SOP being applied:
id: {sop['id']}
title: {sop['title']}
severity: {sop['severity']}
advice: {sop['advice']}

Also relevant (mention briefly by id only, if any):
{also_text}

Real weather facts for {location_label}:
{json.dumps(facts, indent=2)}
"""
    user = f"User's question: {user_message}"
    return _call(system, user, max_tokens=400)


def _safe_json(raw: str, fallback: dict) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    return fallback