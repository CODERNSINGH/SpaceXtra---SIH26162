"""
vision_model.py  (Branch A — "Vision CNN")
-------------------------------------------
DEMO-ONLY MODULE, clearly labeled.

In the real architecture, this file is a Convolutional Neural Network
(ResNet/EfficientNet) trained on our own labeled satellite-image dataset
(see the FlareSat / Kaggle datasets in the architecture PDF).

For today's PROTOTYPE, this file calls Google's Gemini vision model instead,
so the pipeline runs end-to-end right now while the real CNN is being
trained. The function names below intentionally mirror what the real CNN
module's interface will look like, so swapping it in later is a one-file
change, not a rewrite of the whole pipeline.

Say this plainly in your pitch: "this block is a placeholder standing in for
our CNN while it trains."
"""
import io
import json
import random

import numpy as np
from PIL import Image

try:
    from google.genai import types as genai_types
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False

from utils import coord_seed

VISION_PROMPT = """You are simulating one branch of a fire-classification system:
a computer-vision model looking ONLY at a satellite image (no other context).

Look at this satellite image and judge whether the surroundings look more
INDUSTRIAL (buildings, tanks, roads, bare/paved ground, factories) or more
NATURAL (forest, farmland, grass, undeveloped land).

Respond with ONLY valid JSON, no other text:
{
  "industrial_probability_percent": <integer 0-100>,
  "natural_probability_percent": <integer 0-100 -- should sum to ~100 with the above>,
  "visual_notes": ["<short observation 1>", "<short observation 2>"]
}
"""


def _parse_json(raw_text: str) -> dict:
    cleaned = raw_text.strip().strip("`")
    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:]
    return json.loads(cleaned.strip())


def run_vision_branch(image: Image.Image, lat: float, lon: float, gemini_model=None) -> dict:
    """Returns the vision branch's opinion. Uses Gemini if a model object is
    passed in; otherwise falls back to a deterministic simulated response
    (so the notebook still runs end-to-end even with no API key / no quota,
    e.g. for a quick offline rehearsal)."""
    if gemini_model is not None and _GEMINI_AVAILABLE:
        try:
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            image_part = genai_types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")
            response = gemini_model.generate_content([VISION_PROMPT, image_part])
            result = _parse_json(response.text)
            result["source"] = f"gemini_vision:{getattr(gemini_model, 'model_name', 'unknown')} (placeholder for trained CNN)"
            return result
        except Exception as e:
            print(f"[vision_model] Gemini call failed, falling back to simulated response: {e}")

    # --- Deterministic simulated fallback (no API key / offline rehearsal) ---
    rng = random.Random(coord_seed(lat, lon))
    industrial_p = rng.randint(20, 95)
    return {
        "industrial_probability_percent": industrial_p,
        "natural_probability_percent": 100 - industrial_p,
        "visual_notes": ["[simulated — no Gemini call made]"],
        "source": "simulated_fallback",
    }


def fake_activation_chart_values(lat: float, lon: float, n_filters: int = 16):
    """PURELY DECORATIVE — generates plausible-looking 'CNN filter activation'
    bar values for the dashboard visual. These are NOT real network
    activations (there is no trained network in this prototype). Label any
    chart built from this clearly as illustrative."""
    rng = np.random.RandomState(coord_seed(lat, lon) % (2**31 - 1))
    return rng.beta(2, 2, size=n_filters)
