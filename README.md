# Weather-Advisory Support Bot

A LangGraph agent that answers outdoor-activity-safety questions by pulling
live weather from Open-Meteo, matching the situation against a written,
hot-reloadable policy file (SOPs), and composing a reply that only ever says
what the matched SOP + real fetched numbers support — or says plainly that
no policy applies.

## Setup

```bash
git clone <this repo>
cd weather-advisory-bot
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then edit .env and paste your GEMINI_API_KEY
```

Get a free Gemini API key (no billing/credit card required) at
https://aistudio.google.com/apikey. No weather API key is needed —
Open-Meteo is free and keyless too.

**Note on key format:** keys issued by AI Studio now start with `AQ.`
(Google's current standard, replacing the older `AIzaSy...` format) — that's
expected, not a sign something's wrong. If `graph/llm.py` raises a 401 with
`ACCESS_TOKEN_TYPE_UNSUPPORTED`, double-check the exact key value made it
into `.env`/your host's secrets unmodified (a stale or truncated copy is the
usual cause), rather than assuming the key type itself is invalid.

`graph/llm.py` also falls back once to a second model
(`GEMINI_FALLBACK_MODEL`, default `gemini-2.5-flash`) if the primary model
(`GEMINI_MODEL`, default `gemini-3.1-flash-lite`) is overloaded (503) on
every retry attempt — see that file's `_call`/`_call_model` docstrings.

## Running it

**Terminal chat:**
```bash
python main.py
```

**Web chat (Streamlit):**
```bash
streamlit run frontend/app.py
```
Run this from the repo root (not from inside `frontend/`) so the `bot`,
`graph`, and `utils` imports resolve.

Each app run / browser session gets a fresh `thread_id` (shown in the
Streamlit sidebar), which is the unit of "session" — conversation memory
persists across turns within that id and resets when you start a new one,
per the assignment's scope (no cross-session persistence).

## Running the evals

```bash
python evals/eval_suite.py                  # the 8 required eval cases
python evals/test_live_sop_addition.py      # the "add an SOP live" demo
```

Both need a real `GEMINI_API_KEY` (free); most cases in `eval_suite.py` also make
real Open-Meteo calls on purpose (see **Eval suite** below for why, and for
an honest note on what I could and couldn't actually run myself).

## Repo layout

```
sops/sops.yaml          <- the policy. Edit this to change bot behavior. Never touch code for this.
utils/weather.py        <- the only file that talks to Open-Meteo
utils/sop_loader.py     <- loads/validates sops.yaml into plain objects
graph/state.py          <- LangGraph state schema
graph/nodes.py          <- node functions (parse, geocode, fetch, match, compose, fallback)
graph/sop_matcher.py    <- deterministic numeric matching + multi-match resolution policy
graph/llm.py            <- the only file that talks to the LLM (3 narrow jobs, see docstring)
graph/build.py          <- wires the actual StateGraph, conditional edges, checkpointer
bot.py                  <- ask(thread_id, message) — the one shared entry point
main.py                 <- terminal chat loop
frontend/app.py         <- Streamlit chat UI
evals/                  <- eval suite + the live-SOP-addition demo
```

## How SOPs are represented, and why

SOPs live in `sops/sops.yaml` as a flat list of records with `id`, `category`,
`condition_type` (`numeric` or `fuzzy`), a machine-checkable `condition` (for
numeric ones) or a `situation_description` (for fuzzy ones), a `severity`,
and free-text `advice`.

**Why YAML over, say, a database or a DSL:** it's human-readable and
diffable in git (so policy changes get code review like anything else), it
needs zero new tooling to edit, and it's trivial to validate on load
(`utils/sop_loader.py` raises immediately on a bad `condition_type` or a
duplicate id). A domain-specific rule language would be more powerful but
was overkill for ~10-15 rules where the two condition shapes I need (simple
AND-of-comparisons, and free-text situational description) are exhaustive
of what showed up while writing them.

**34 SOPs shipped**, across 6 categories (`outdoor_exercise`, `travel`,
`vulnerable_groups`, `pets`, `outdoor_leisure`, `severe_weather_system`), at
all 4 severities (`informational`, `caution`, `warning`, `severe`). IDs
follow a `SOP-<CATEGORY>-<AXIS>-<NN>` scheme (e.g. `SOP-EX-UV-02`,
`SOP-VG-TEMP-04`) so a policy's category and hazard axis are readable from
its id alone; `SOP-SEVERE-SYSTEM` and the three `SOP-LEISURE-*` ids are the
fuzzy (non-numeric) ones. Most numeric axes (temperature, wind, rain,
UV) are split into exhaustive, non-overlapping bands per category, so every
possible reading matches exactly one SOP on that axis — see the comment
block at the top of `sops/sops.yaml` for why that avoids the "falls between
two AND-combined conditions and matches nothing" failure mode.

## Architecture (the actual graph)

```
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
```

Two real conditional branches (`geocode` failure, `fetch_weather` failure),
both routing into one `honest_fallback` node. See `graph/build.py`'s
docstring for why the "SOP matched" vs. "no SOP matched" split is handled as
a branch *inside* `compose_answer`'s prompt rather than as a third graph
edge: both cases still do the same thing structurally (produce one grounded
AIMessage from real facts), whereas the failure paths genuinely have no
facts to compose from at all.

### What's deterministic code vs. what's the LLM's job

This was the central design decision, and it maps directly onto the
assignment's non-negotiables:

| Job | Where | Why |
|---|---|---|
| Evaluate numeric thresholds (`uv_index >= 8`, etc.) | Plain Python, `graph/sop_matcher.py` | A number comparison is not a language task. Doing it in code is the only way to *guarantee* — not just prompt for — that the reported facts are the real fetched facts. |
| Recognize a fuzzy situation (picnic weather, an active severe-weather system) | LLM, given ONLY that SOP's `situation_description` + real facts, must answer with a real id from a closed list or `NO_MATCH` | There's no clean threshold for "is this a nice picnic day" or "is this a named weather system overriding category-specific advice" — that's explicitly the assignment's fuzzy-case ask. The LLM can't invent an id here: whatever it returns is validated against the candidate set in `graph/llm.py::pick_fuzzy_sop`, and anything else is treated as `NO_MATCH`. |
| Decide which SOP "wins" when several match | Plain Python, `graph/sop_matcher.py::resolve()` | Documented explicitly in that file's docstring: rank by severity, highest wins as primary, others at caution+ severity get a one-line "also relevant" mention — **except** any SOP in the `severe_weather_system` category, which always overrides everything else (checked by category name, so a new SOP added to that category on the review call inherits override behavior for free). |
| Extract location + activity from free text | LLM, `graph/llm.py::extract_intent`, constrained to a fixed vocabulary of activity tags | This is genuinely a language-understanding task ("laying out a blanket and eating dinner outside" → `picnic`/`leisure`), but it never touches weather numbers or advice, so there's no path for it to hallucinate a fact into the pipeline. |
| Compose the final sentence | LLM, `graph/llm.py::compose_answer`, given exactly one resolved SOP's advice text + the real facts dict, explicitly forbidden from citing any other policy or number | Phrasing is a language task; policy selection and fact retrieval already happened deterministically before this node runs. |

### Multi-match resolution (the thing the assignment says to "decide on
purpose about")

Documented in `graph/sop_matcher.py`: **rank by severity, present the
highest as primary, name any other caution+ matches briefly** — with a hard
override for the `severe_weather_system` category, which suppresses
everything else per its own advice text ("this overrides category-specific
advice"). I chose "primary + brief mention" over "surface all matches
equally" because a user asking one question expects one clear headline
recommendation, but silently dropping a second real hazard (e.g. high wind
on a high-UV cycling day) felt like it defeated the point of writing SOPs
carefully in the first place.

### Session memory

`graph/build.py` uses LangGraph's `MemorySaver` checkpointer keyed by
`thread_id`. Beyond raw message history (accumulated automatically via the
`add_messages` reducer), I carry a small set of **structured** session
fields forward explicitly: last resolved location (name + lat/lon) and last
activity hints. A follow-up like "what about this evening instead" is
recognized by `extract_intent` (told explicitly whether a prior location
exists) and, if so, reuses the session's lat/lon directly — skipping a
second geocoding call entirely rather than re-summarizing raw history with
another LLM call. I chose structured facts over relying purely on raw
message replay because it's cheaper, deterministic, and easy to unit-test
(see `evals` and the manual smoke tests I describe below) — a design
trade-off I'd revisit if the bot needed to track many more kinds of
carried-over context than "where" and "what."

### Grounding guarantee (numbers can't be invented)

`utils/weather.py::WeatherSnapshot.as_fact_dict()` is a flat dict of exactly
what Open-Meteo returned. That dict is the *only* thing passed into
`compose_answer`'s prompt as "real weather facts," and the prompt explicitly
forbids stating any number not present in it. The LLM never sees the SOP
threshold values it's being asked to phrase around — because it never had
to evaluate them; `sop_matcher.py` already did, in plain Python, before the
LLM is invoked at all. If you want to verify this mechanically rather than
trust the prompt wording: `graph/nodes.py::compose_answer_node` is the only
call site for `llm.compose_answer`, and it always passes `state["turn_facts"]`
straight from `fetch_weather_node`'s output — there's no code path where a
model-recalled number could substitute for a fetched one.

## Eval suite — what's covered and an honest note on what I actually ran

`evals/eval_suite.py` implements all 8 required case types:

1. **Clear match #1** — direct high-UV running question, checked against
   `SOP-EX-UV-02` (fires iff `uv_index >= 8`).
2. **Clear match #2** — direct high-wind cycling question, checked against
   `SOP-EX-WIND-03` (fires iff `wind_speed_10m > 40`).
3. **Paraphrase #1** — "my 78-year-old grandmother on the porch" (never says
   "elderly" or "heat"), must still resolve to `SOP-VG-TEMP-03` (34-40°C) or
   `SOP-VG-TEMP-04` (>=40°C) at the right threshold.
4. **Paraphrase #2** — "eating dinner on a blanket outside" (never says
   "picnic"/"park"), must resolve to a real fuzzy leisure SOP id
   (`SOP-LEISURE-GOOD`/`-MARGINAL`/`-POOR`) or an honest no-match, never a
   fabricated one.
5. **Severe live weather** — asks about biking in Bhopal against live
   Open-Meteo data. See the case's docstring for why the assertion is
   intentionally weaker than "must say `SOP-SEVERE-SYSTEM`": a real weather
   system is a moving target (see "Known limitations" below).
6. **No SOP applies** — an unrelated indoor question ("what podcast should I
   listen to"), asserts `primary_sop_id is None` and no invented advice.
7. **Simulated API outage** — monkeypatches `requests.get` to raise
   `ConnectionError`, asserts an honest failure with no fabricated facts.
8. **Adversarial: prompt injection** — a user message instructing the bot to
   ignore its SOPs and cite a fabricated `SOP-999`. See the case's docstring
   for why I picked this over other adversarial angles (malformed/huge
   input is a robustness concern; policy-spoofing is a *trust* concern that
   attacks the exact guarantee this whole system exists to provide).

`evals/test_live_sop_addition.py` is the separate "add an SOP with zero
code changes" demonstration the assignment explicitly says will be asked for
live on the review call. It appends a real new SOP (`SOP-EX-FOG-01`, low-
visibility fog — a scenario the shipped 34 don't cover) directly to
`sops/sops.yaml`, proves it loads and matches, then restores the file. I ran
this one myself (no LLM call needed, just the loader + matcher) — output:

```
Appended SOP-EX-FOG-01 to sops.yaml (no code files touched).
Confirmed: load_sops() now returns 35 SOPs including SOP-EX-FOG-01.
Numeric match against fog-like facts: ['SOP-EX-TEMP-03', 'SOP-EX-WIND-01', 'SOP-EX-UV-01', 'SOP-EX-RAIN-01', 'SOP-EX-FOG-01']

PASS: a new SOP was added and correctly matched with zero changes to any .py file -- only sops/sops.yaml was edited.

Restored sops.yaml to its original contents.
```

**What I could NOT run myself, and why (read this — it matters):** I built
this in a sandboxed environment with no `GEMINI_API_KEY` and no network
access to `api.open-meteo.com` or `geocoding-api.open-meteo.com`. So:

- Every LLM-dependent function (`extract_intent`, `pick_fuzzy_sop`,
  `compose_answer`) was verified with unit tests that mock those functions'
  *return values* and check the graph wires them correctly end-to-end (see
  the mocked full-pipeline runs I did during development — same technique
  `eval_suite.py`'s outage case uses for `requests.get`), not with real
  model calls.
- `utils/weather.py`'s HTTP-calling functions were verified with mocked
  `requests.get` responses shaped exactly like Open-Meteo's documented
  schema (including the "current has no matching hourly index" and
  "geocoding returns zero results" edge cases), plus a real ConnectionError
  and Timeout simulation — not a live call.
- **`eval_suite.py` and the Bhopal live-weather case have not been run
  end-to-end against the real API and a real model by me.** You (with a real
  key and normal network access) need to run `python evals/eval_suite.py`
  yourself to get the authoritative results, and I'd genuinely expect 1-2
  cases to need a prompt tweak on first real run — that's normal, and I'd
  rather say that plainly than claim untested numbers passed.

### The "live weather doesn't sit still" problem

The Madhya Pradesh system referenced in the assignment is forecast to weaken
by Sept 5; by the time this is reviewed it will likely be an ordinary day in
Bhopal. `case_severe_live_weather` is written so it doesn't assume elevated
numbers — it asserts the mechanically-checkable thing (real data was
fetched, the cited SOP id is real, nothing was fabricated) and prints the
actual facts + chosen SOP for a human to judge against whatever's really
happening on the day it's run. If I were hardening this further for a suite
that has to keep working across an entire monsoon season, I'd add a second,
narrower live case that runs against a small rotating list of coordinates
known for currently-active alerts (fetched at test-run time from a public
advisory feed rather than hardcoded), and treat "did it match something
severity >= warning" as the pass condition only on days that feed confirms
an active advisory — falling back to a monsoon-season low-latitude coastal
city (which reliably has *some* elevated precipitation probability most of
the year) as a softer proxy on days with no confirmed active system anywhere.
I didn't build that rotating-feed check here — it's a meaningfully bigger
piece of infrastructure than the assignment's one-day budget allows, and I'd
rather flag it as the honest next step than half-build it.

## Known limitations / things I'd flag rather than hide

- **Geocoding ambiguity** is resolved by taking the first result silently
  (the assignment calls this a reasonable default). `ResolvedLocation`
  carries `candidate_count` so a future version could mention "there are
  multiple places named X" when it's > 1; I didn't wire that into the
  reply text to keep the composed answer simple, and I'm flagging that
  as a deliberate scope cut, not an oversight.
- **The fuzzy matcher makes one LLM call per turn** even when no fuzzy SOP
  could plausibly apply given the activity hints (it's still fast/cheap
  since candidates are pre-filtered by activity hint, but it's not free).
  A larger policy set (hundreds of fuzzy SOPs) would need a cheaper
  pre-filter than "all fuzzy SOPs matching this activity tag."
- **`extract_intent`'s activity-hint vocabulary is fixed** (see
  `graph/llm.py::ALLOWED_ACTIVITY_HINTS`). Adding a genuinely new *kind* of
  activity (say, a new `gardening` category) requires adding both a new SOP
  *and* a new tag to that vocabulary list — which technically means the
  "no code changes" guarantee is scoped to *adding rules within existing
  categories/activities*, not inventing a wholly new activity type. I think
  this is a reasonable scope boundary (the review-call ask was for a new
  *rule*, not a new *domain*), but I'm naming it rather than letting it be
  a surprise.
- **Tie-breaking among equal-severity matches** falls back to the order
  SOPs appear in `sops.yaml` (Python's stable sort preserves input order for
  equal keys) — this is deterministic but implicit; I'd make it an explicit
  `priority` field if the policy set grew much larger.