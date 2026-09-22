"""
Minimal Streamlit chat frontend.

Run from the repo root: streamlit run frontend/app.py
(Run from the repo root so `bot`/`graph`/`utils` imports resolve.)
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Make repo root importable when Streamlit launches this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

load_dotenv()

from bot import ask  # noqa: E402

st.set_page_config(page_title="Weather-Advisory Support Bot", page_icon="🌦️")
st.title("🌦️ Weather-Advisory Support Bot")
st.caption(
    "Ask about outdoor activity safety (cycling, picnics, travel, kids, pets...). "
    "Every answer is grounded in live weather data and a written policy (SOP) — "
    "or the bot will say plainly that it doesn't have guidance."
)

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "history" not in st.session_state:
    st.session_state.history = []  # list of (role, content, meta)

with st.sidebar:
    st.subheader("Session")
    st.code(st.session_state.thread_id, language=None)
    if st.button("Start new session"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.history = []
        st.rerun()
    st.caption("A new session id clears memory — the bot won't recall earlier turns.")

for role, content, meta in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)
        if meta:
            st.caption(meta)

user_message = st.chat_input("e.g. Is it safe to bike to work today in Bhopal?")
if user_message:
    st.session_state.history.append(("user", user_message, None))
    with st.chat_message("user"):
        st.markdown(user_message)

    with st.chat_message("assistant"):
        with st.spinner("Checking live weather + policy..."):
            response = ask(st.session_state.thread_id, user_message)

        st.markdown(response.answer)

        meta_bits = []
        if response.primary_sop_id:
            meta_bits.append(f"Policy: {response.primary_sop_id}")
        if response.also_relevant_sop_ids:
            meta_bits.append(f"Also relevant: {', '.join(response.also_relevant_sop_ids)}")
        if response.location_name:
            meta_bits.append(f"Location: {response.location_name}")
        if response.error_stage:
            meta_bits.append(f"Fallback triggered ({response.error_stage})")
        meta = " · ".join(meta_bits) if meta_bits else None
        if meta:
            st.caption(meta)

        if response.facts:
            with st.expander("Raw weather facts used"):
                st.json(response.facts)

    st.session_state.history.append(("assistant", response.answer, meta))
