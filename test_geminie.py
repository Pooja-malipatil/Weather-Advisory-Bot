"""
Quick standalone test for your Gemini API key.

Run this LOCALLY (not on Streamlit Cloud) with:

    pip install google-genai
    export GEMINI_API_KEY="your-actual-key-here"   # or set it inline below
    python test_gemini_key.py

It will print the REAL error (status code + message) instead of the
redacted version Streamlit Cloud shows you.
"""

import os
from google import genai
from google.genai import types
from google.genai import errors as genai_errors

# Paste your key directly here between the quotes as a quick alternative
# to setting an environment variable. Remove it again when you're done
# testing so you don't accidentally commit it.
HARDCODED_KEY = ""

api_key = HARDCODED_KEY or os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("GEMINI_API_KEY is not set in this shell, and HARDCODED_KEY is empty.")
    print("Either set the env var, or paste your key into HARDCODED_KEY above.")
    raise SystemExit(1)

print(f"Using key starting with: {api_key[:8]}... (length {len(api_key)})")

client = genai.Client(api_key=api_key)

try:
    resp = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents="Say hello in one word.",
        config=types.GenerateContentConfig(max_output_tokens=20),
    )
    print("SUCCESS. Response:", resp.text)

except genai_errors.ClientError as e:
    status = getattr(e, "code", None) or getattr(e, "status_code", None)
    print(f"\nCLIENT ERROR - status: {status}")
    print(f"Full exception: {e}")
    # Try to get the raw response body too, if present
    body = getattr(e, "response", None)
    if body is not None:
        try:
            print("Raw response JSON:", body.json())
        except Exception:
            print("Raw response text:", getattr(body, "text", "<unavailable>"))

except genai_errors.ServerError as e:
    print(f"\nSERVER ERROR (transient, Google's side): {e}")

except Exception as e:
    print(f"\nUNEXPECTED ERROR ({type(e).__name__}): {e}")