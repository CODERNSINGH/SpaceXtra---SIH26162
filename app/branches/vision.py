"""
Branch A — Vision.

*** THIS ENTIRE MODULE IS A PLACEHOLDER FOR A NOT-YET-TRAINED CNN. ***
Given the event's fetched satellite/aerial image, a general-purpose
multimodal LLM (Google Gemini, primary) is asked to classify whether the
surroundings look industrial or natural and to return a structured JSON
verdict. This stands in for a custom-trained CNN.

Swap-in point for a real CNN: implement a new `VisionProvider` subclass
below (e.g. `TrainedCNNProvider`) whose `classify()` returns the same
`{classification, confidence, evidence}` shape, and set it as the
`PRIMARY_PROVIDER` at the bottom of this file. Nothing in fusion.py, the API
routes, or the frontend needs to change.

Multi-provider / ensemble: Groq and OpenRouter are wired in as alternate
providers using the same prompt and interface, so a single event's image can
be judged by more than one independent model and the answers shown side by
side (agreement is itself evidence).
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Callable

import httpx
from PIL import Image

from app.branches import CATEGORIES
from app.config import settings
from app.processing.image_ops import to_base64_png

LogFn = Callable[[str], None]


def _noop_log(_msg: str) -> None:
    pass


PROMPT_TEMPLATE = """You are assisting an analyst reviewing a satellite/aerial image crop \
centered on a NASA FIRMS thermal hotspot detection at latitude {lat}, longitude {lon}, \
inside India. Look at the image and judge what kind of thermal source this most likely is.

Classify into EXACTLY one of these categories:
{categories}

Respond with ONLY a single JSON object, no other text, in this exact shape:
{{"classification": "<one category from the list>", "confidence": <float 0.0-1.0>, \
"evidence": ["<short observation 1>", "<short observation 2>", "<short observation 3>"]}}

Base your evidence on what is visible in the image (built structures, roads, bare/cleared \
land, vegetation cover, water, cropland patterns, mining pits, etc.), not on any prior \
knowledge of the specific coordinates."""


def _build_prompt(lat: float, lon: float) -> str:
    return PROMPT_TEMPLATE.format(lat=lat, lon=lon, categories=", ".join(CATEGORIES))


def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object found in model response: {text[:200]!r}")
    return json.loads(match.group(0))


class VisionProvider(ABC):
    name: str

    @abstractmethod
    def is_configured(self) -> bool:
        ...

    @abstractmethod
    def classify(self, image_b64_png: str, lat: float, lon: float, client: httpx.Client) -> dict:
        """Returns {"classification": str, "confidence": float, "evidence": [str, ...]}"""
        ...


class GeminiProvider(VisionProvider):
    name = "gemini"

    def is_configured(self) -> bool:
        return bool(settings.GEMINI_API_KEY)

    def classify(self, image_b64_png, lat, lon, client):
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{settings.GEMINI_MODEL}:generateContent?key={settings.GEMINI_API_KEY}")
        body = {
            "contents": [{
                "parts": [
                    {"text": _build_prompt(lat, lon)},
                    {"inline_data": {"mime_type": "image/png", "data": image_b64_png}},
                ]
            }],
            "generationConfig": {"temperature": 0.1},
        }
        resp = client.post(url, json=body)
        resp.raise_for_status()
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return _extract_json(text)


class GroqProvider(VisionProvider):
    name = "groq"

    def is_configured(self) -> bool:
        return bool(settings.GROQ_API_KEY)

    def classify(self, image_b64_png, lat, lon, client):
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {settings.GROQ_API_KEY}"}
        body = {
            "model": settings.GROQ_MODEL,
            "temperature": 0.1,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": _build_prompt(lat, lon)},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64_png}"}},
                ],
            }],
        }
        resp = client.post(url, json=body, headers=headers)
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
        return _extract_json(text)


class OpenRouterProvider(VisionProvider):
    name = "openrouter"

    def is_configured(self) -> bool:
        return bool(settings.OPENROUTER_API_KEY)

    def classify(self, image_b64_png, lat, lon, client):
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {"Authorization": f"Bearer {settings.OPENROUTER_API_KEY}"}
        body = {
            "model": settings.OPENROUTER_MODEL,
            "temperature": 0.1,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": _build_prompt(lat, lon)},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64_png}"}},
                ],
            }],
        }
        resp = client.post(url, json=body, headers=headers)
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
        return _extract_json(text)


# --- Future swap-in point -------------------------------------------------
# class TrainedCNNProvider(VisionProvider):
#     name = "trained_cnn"
#     def is_configured(self) -> bool: return MODEL_WEIGHTS_PATH.exists()
#     def classify(self, image_b64_png, lat, lon, client) -> dict:
#         ... run the real model, return the same dict shape ...
# ---------------------------------------------------------------------------

ALL_PROVIDERS: list[VisionProvider] = [GeminiProvider(), GroqProvider(), OpenRouterProvider()]
PRIMARY_PROVIDER = ALL_PROVIDERS[0]  # Gemini — swap this line to change the primary engine


def _to_probabilities(classification: str, confidence: float) -> dict[str, float]:
    classification = classification if classification in CATEGORIES else "unknown"
    confidence = max(0.0, min(1.0, float(confidence)))
    remainder = (1.0 - confidence) / max(1, len(CATEGORIES) - 1)
    return {c: (confidence if c == classification else remainder) for c in CATEGORIES}


def run_vision_branch(image: Image.Image, lat: float, lon: float, log: LogFn = _noop_log) -> dict:
    ensemble = []
    with httpx.Client(timeout=settings.HTTP_TIMEOUT_SECONDS) as client:
        image_b64 = to_base64_png(image)
        for provider in ALL_PROVIDERS:
            if not provider.is_configured():
                log(f"[VISION:{provider.name}] No API key configured — skipping.")
                ensemble.append({"provider": provider.name, "status": "unconfigured"})
                continue
            try:
                log(f"[VISION:{provider.name}] Sending image + prompt for classification...")
                result = provider.classify(image_b64, lat, lon, client)
                classification = result.get("classification", "unknown")
                confidence = float(result.get("confidence", 0.5))
                evidence = result.get("evidence", [])
                log(f"[VISION:{provider.name}] -> {classification} ({confidence*100:.0f}%)")
                ensemble.append({
                    "provider": provider.name, "status": "ok",
                    "classification": classification, "confidence": confidence, "evidence": evidence,
                })
            except httpx.TimeoutException:
                log(f"[VISION:{provider.name}] ERROR: timed out after {settings.HTTP_TIMEOUT_SECONDS}s")
                ensemble.append({"provider": provider.name, "status": "error", "message": "timeout"})
            except Exception as e:  # noqa: BLE001
                log(f"[VISION:{provider.name}] ERROR: {e}")
                ensemble.append({"provider": provider.name, "status": "error", "message": str(e)})

    primary_result = next((e for e in ensemble if e["provider"] == PRIMARY_PROVIDER.name and e["status"] == "ok"), None)
    if primary_result is None:
        # fall back to any provider that succeeded
        primary_result = next((e for e in ensemble if e["status"] == "ok"), None)

    if primary_result is None:
        log("[VISION] No provider available (no keys configured or all failed) — "
            "returning neutral/unknown-leaning distribution.")
        class_probabilities = {c: (0.4 if c == "unknown" else 0.6 / (len(CATEGORIES) - 1)) for c in CATEGORIES}
        evidence = ["no vision provider configured or reachable — vision branch abstained"]
        top_class = "unknown"
        used_provider = None
    else:
        class_probabilities = _to_probabilities(primary_result["classification"], primary_result["confidence"])
        evidence = list(primary_result.get("evidence", []))
        top_class = primary_result["classification"]
        used_provider = primary_result["provider"]

    return {
        "class_probabilities": class_probabilities,
        "top_class": top_class,
        "evidence": evidence,
        "used_provider": used_provider,
        "ensemble": ensemble,
        "model_label": "Gemini vision (placeholder for trained CNN)",
    }
