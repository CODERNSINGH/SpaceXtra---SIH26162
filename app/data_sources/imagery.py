"""
Satellite/aerial imagery providers — Branch A's input image.

Provider interface is intentionally small (`ImageryProvider.fetch`) so a
future `GoogleEarthEngineProvider` (pending access approval — see
SOURCES.md) can be dropped into `PROVIDER_CHAIN` below without touching
anything else in the app: Branch A, the image-processing gallery, and the
caching layer all just call `get_satellite_image(lat, lon)`.

Order tried: Esri World Imagery -> EOX Sentinel-2 Cloudless -> deterministic
synthetic placeholder (clearly labeled, never presented as real imagery).
"""
from __future__ import annotations

import hashlib
import io
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import httpx
import numpy as np
from PIL import Image

from app.config import settings

LogFn = Callable[[str], None]


def _noop_log(_msg: str) -> None:
    pass


@dataclass
class ImageryResult:
    image: Image.Image
    source: str          # e.g. "esri_world_imagery", "eox_sentinel2_cloudless", "synthetic_placeholder"
    is_real: bool
    bbox: tuple[float, float, float, float]  # west, south, east, north


class ImageryProvider(ABC):
    name: str

    @abstractmethod
    def fetch(self, lat: float, lon: float, size_px: int, half_width_m: float,
              client: httpx.Client) -> Image.Image | None:
        ...


def _bbox_for_point(lat: float, lon: float, half_width_m: float) -> tuple[float, float, float, float]:
    dlat = half_width_m / 111_320.0
    dlon = half_width_m / (111_320.0 * max(0.1, np.cos(np.radians(lat))))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


class EsriWorldImageryProvider(ImageryProvider):
    name = "esri_world_imagery"
    URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"

    def fetch(self, lat, lon, size_px, half_width_m, client):
        west, south, east, north = _bbox_for_point(lat, lon, half_width_m)
        params = {
            "bbox": f"{west},{south},{east},{north}",
            "bboxSR": "4326",
            "imageSR": "4326",
            "size": f"{size_px},{size_px}",
            "format": "png",
            "f": "image",
        }
        resp = client.get(self.URL, params=params)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        # Esri returns a generic "no georeferenced data" gray tile at some
        # zooms/locations — treat a near-uniform image as a soft failure.
        if np.array(img).std() < 2.0:
            raise ValueError("Esri returned a blank/uniform tile")
        return img


class EOXSentinel2Provider(ImageryProvider):
    name = "eox_sentinel2_cloudless"
    URL = "https://tiles.maps.eox.at/wms"

    def fetch(self, lat, lon, size_px, half_width_m, client):
        west, south, east, north = _bbox_for_point(lat, lon, half_width_m)
        params = {
            "service": "WMS",
            "request": "GetMap",
            "version": "1.1.1",
            "layers": "s2cloudless-2020",
            "styles": "",
            "bbox": f"{west},{south},{east},{north}",
            "width": str(size_px),
            "height": str(size_px),
            "srs": "EPSG:4326",
            "format": "image/png",
        }
        resp = client.get(self.URL, params=params)
        resp.raise_for_status()
        if "image" not in resp.headers.get("content-type", ""):
            raise ValueError("EOX did not return an image")
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        if np.array(img).std() < 2.0:
            raise ValueError("EOX returned a blank/uniform tile")
        return img


PROVIDER_CHAIN: list[ImageryProvider] = [EsriWorldImageryProvider(), EOXSentinel2Provider()]


def _synthetic_placeholder_image(lat: float, lon: float, size_px: int) -> Image.Image:
    """Deterministic procedural image so the pipeline never breaks if both
    real imagery providers are unreachable. Clearly not real imagery."""
    seed = int(hashlib.sha1(f"{lat:.4f},{lon:.4f}".encode()).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)
    base = rng.integers(60, 140, size=(size_px, size_px, 3), dtype=np.uint8)
    # a few blocky "structures" so edge/segmentation steps have something to find
    for _ in range(6):
        x, y = rng.integers(0, size_px - 40, size=2)
        w, h = rng.integers(15, 40, size=2)
        color = rng.integers(80, 220, size=3)
        base[y:y + h, x:x + w] = color
    img = Image.fromarray(base, mode="RGB")
    return img


def _cache_path(lat: float, lon: float) -> Path:
    key = hashlib.sha1(f"{lat:.4f},{lon:.4f}".encode()).hexdigest()
    return settings.IMAGE_CACHE_DIR / f"{key}.png"


def get_satellite_image(lat: float, lon: float, size_px: int = 512,
                         half_width_m: float = 400.0, log: LogFn = _noop_log) -> ImageryResult:
    cache_file = _cache_path(lat, lon)
    bbox = _bbox_for_point(lat, lon, half_width_m)

    with httpx.Client(timeout=settings.HTTP_TIMEOUT_SECONDS, headers={"User-Agent": "AGNIDRISHTI/1.0"}) as client:
        for provider in PROVIDER_CHAIN:
            try:
                log(f"[IMAGERY] Trying {provider.name} for ({lat:.4f}, {lon:.4f})...")
                img = provider.fetch(lat, lon, size_px, half_width_m, client)
                if img is not None:
                    log(f"[IMAGERY] OK — image fetched from {provider.name}.")
                    img.save(cache_file)
                    return ImageryResult(image=img, source=provider.name, is_real=True, bbox=bbox)
            except httpx.TimeoutException:
                log(f"[IMAGERY] ERROR: {provider.name} timed out after {settings.HTTP_TIMEOUT_SECONDS}s")
            except Exception as e:  # noqa: BLE001
                log(f"[IMAGERY] ERROR: {provider.name} failed: {e}")

    log("[IMAGERY] All real providers failed — using synthetic placeholder image (labeled).")
    img = _synthetic_placeholder_image(lat, lon, size_px)
    return ImageryResult(image=img, source="synthetic_placeholder", is_real=False, bbox=bbox)
