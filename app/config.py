"""
Central configuration for AGNIDRISHTI.

Everything that comes from the environment is read here, once, so the rest of
the app never calls os.environ directly. Nothing here raises if a key is
missing — callers must check `bool(settings.SOME_KEY)` and degrade gracefully
(see SOURCES.md, "Fallback / failure behavior").
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class Settings:
    # NASA FIRMS
    FIRMS_MAP_KEY: str = os.getenv("FIRMS_MAP_KEY", "").strip()

    # AI providers
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "").strip()
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-flash-latest").strip()

    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "").strip()
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").strip()

    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "").strip()
    OPENROUTER_MODEL: str = os.getenv(
        "OPENROUTER_MODEL", "minimax/minimax-m3:free"
    ).strip()

    GEE_SERVICE_ACCOUNT_JSON: str = os.getenv("GEE_SERVICE_ACCOUNT_JSON", "").strip()

    # Storage
    DATABASE_PATH: Path = BASE_DIR / os.getenv("DATABASE_PATH", "data/thermowatch.db")
    IMAGE_CACHE_DIR: Path = BASE_DIR / os.getenv("IMAGE_CACHE_DIR", "data/cache/images")

    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").strip()

    # India bounding box used for the default FIRMS pull. North bound extended
    # to 37.6 (not 35.5) to actually include all of Jammu & Kashmir and Ladakh
    # up through the Siachen Glacier area — the tighter box was silently
    # excluding real Indian territory from hotspot detection entirely.
    INDIA_BBOX = (68.0, 6.5, 97.5, 37.6)  # west, south, east, north

    # Fusion weights (Layer 3) — documented, config-editable.
    # Must sum to 1.0. Change these to re-weight branch trust without
    # touching any branch's internal logic.
    FUSION_WEIGHTS = {
        "vision": 0.40,
        "context": 0.35,
        "temporal": 0.25,
    }

    HTTP_TIMEOUT_SECONDS: float = 10.0


settings = Settings()
settings.IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
settings.DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
