"""
gemini_setup.py
----------------
Loads a Gemini API key from the environment (a local .env file, a Colab
secret, or manual getpass input) and returns a configured GenerativeModel.
This IS the placeholder standing in for the trained CNN in vision_model.py
(see that file's docstring) — swapping it for a real fine-tuned model later
only touches vision_model.py, not this file or the pipeline.

SECURITY NOTE: never hardcode your real API key into a tracked .py or
.ipynb file. Put it in a local .env file (already gitignored in this repo,
see .env.example) or Colab's "Secrets" panel, and load it from there. If a
key was ever pasted into a chat, a doc, or a commit, treat it as burned and
regenerate it in Google AI Studio (https://aistudio.google.com/apikey).
"""
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from google import genai  # the current, supported SDK (pip install google-genai)
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False

DEFAULT_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


class GeminiModel:
    """Thin wrapper around google-genai's Client so vision_model.py can call
    `.generate_content([...])` the same way regardless of SDK generation —
    the new SDK calls generate_content on the client, not on a per-model
    object, so this just remembers which model name to pass each time."""

    def __init__(self, client, model_name: str):
        self.client = client
        self.model_name = model_name

    def generate_content(self, parts):
        return self.client.models.generate_content(model=self.model_name, contents=parts)


def configure_gemini(api_key: str = None, model_name: str = None):
    """Returns a ready-to-use GeminiModel, or None if unavailable/no key is
    found — the pipeline still runs end-to-end either way, because
    vision_model.run_vision_branch() falls back to a deterministic
    simulated response when no model is passed in.

    Note: even with a valid key, Google's free tier restricts which models
    are usable per-project. If every call fails with 403/404/429, check
    https://aistudio.google.com/apikey for your project's billing/model
    access — the vision branch will keep working via its simulated
    fallback in the meantime, clearly labeled as such in its output."""
    if not _GEMINI_AVAILABLE:
        print("[gemini_setup] google-genai not installed — vision branch will use the simulated fallback.")
        return None

    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        print("[gemini_setup] No GEMINI_API_KEY found (checked argument + .env) — "
              "vision branch will use the simulated fallback.")
        return None

    resolved_model_name = model_name or DEFAULT_MODEL_NAME
    client = genai.Client(api_key=key)
    print(f"[gemini_setup] Gemini configured — model '{resolved_model_name}' ready "
          f"(falls back to simulated response per-call if the API rejects the request).")
    return GeminiModel(client, resolved_model_name)
