"""
image_fetcher.py
----------------
REAL module: fetches an actual satellite/aerial image for a coordinate.
Tries Esri World Imagery first (free, no key), falls back to EOX Sentinel-2
Cloudless (free, no key) if that fails.
"""
import io
import requests
from PIL import Image
from utils import PipelineConfig


def _esri_url(lat, lon, cfg: PipelineConfig):
    d = cfg.image_buffer_deg
    return (
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"
        f"?bbox={lon-d},{lat-d},{lon+d},{lat+d}&bboxSR=4326&imageSR=4326"
        f"&size={cfg.image_size},{cfg.image_size}&format=png32&f=image"
    )


def _eox_url(lat, lon, cfg: PipelineConfig):
    d = cfg.image_buffer_deg
    return (
        "https://tiles.maps.eox.at/wms?service=WMS&request=GetMap&version=1.1.1"
        "&layers=s2cloudless-2023&styles=&format=image/png"
        f"&bbox={lon-d},{lat-d},{lon+d},{lat+d}&srs=EPSG:4326"
        f"&width={cfg.image_size}&height={cfg.image_size}"
    )


def fetch_satellite_image(lat: float, lon: float, cfg: PipelineConfig = None) -> Image.Image:
    """Returns a PIL Image. Raises RuntimeError only if every source fails."""
    cfg = cfg or PipelineConfig()
    errors = []
    for name, url_fn in [("Esri World Imagery", _esri_url), ("EOX Sentinel-2 Cloudless", _eox_url)]:
        try:
            url = url_fn(lat, lon, cfg)
            resp = requests.get(url, timeout=cfg.request_timeout)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            print(f"[image_fetcher] got image from {name}")
            return img
        except Exception as e:
            errors.append(f"{name}: {e}")

    raise RuntimeError("All imagery sources failed:\n" + "\n".join(errors))
