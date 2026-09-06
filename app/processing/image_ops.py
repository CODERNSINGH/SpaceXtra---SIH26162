"""
Image-processing gallery for a clicked hotspot's satellite image.

Every function here runs on the real pixels of whatever image
`get_satellite_image` returned (real imagery or the labeled synthetic
placeholder if providers were unreachable). The k-means segmentation is a
genuine unsupervised algorithm run on real pixel data; the "false color /
heat-style" views are explicitly simulated colormaps applied to luminance,
not real thermal-band data — labeled as such everywhere they're shown.
"""
from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image, ImageFilter
from sklearn.cluster import KMeans

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def to_base64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def make_grayscale(img: Image.Image) -> Image.Image:
    return img.convert("L").convert("RGB")


def make_edges(img: Image.Image) -> Image.Image:
    gray = img.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    return edges.convert("RGB")


def _apply_colormap(img: Image.Image, cmap_name: str) -> Image.Image:
    gray = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    colormap = matplotlib.colormaps[cmap_name]
    colored = (colormap(gray)[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(colored, mode="RGB")


def make_false_color(img: Image.Image) -> Image.Image:
    """Simulated heat-style view #1 (inferno colormap over luminance)."""
    return _apply_colormap(img, "inferno")


def make_false_color_alt(img: Image.Image) -> Image.Image:
    """Simulated heat-style view #2 (jet-like colormap over luminance)."""
    return _apply_colormap(img, "turbo")


def make_kmeans_segmentation(img: Image.Image, k: int = 5) -> Image.Image:
    """Real unsupervised segmentation: k-means over RGB pixel values."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w, _ = arr.shape
    pixels = arr.reshape(-1, 3)
    # Downsample for speed if the image is large, then upsample labels back.
    sample_n = min(20000, pixels.shape[0])
    rng = np.random.default_rng(0)
    idx = rng.choice(pixels.shape[0], size=sample_n, replace=False)
    km = KMeans(n_clusters=k, n_init=4, random_state=0)
    km.fit(pixels[idx])
    labels = km.predict(pixels)
    palette = (matplotlib.colormaps["tab10"](np.linspace(0, 1, k))[:, :3] * 255).astype(np.uint8)
    seg = palette[labels].reshape(h, w, 3).astype(np.uint8)
    return Image.fromarray(seg, mode="RGB")


def build_image_gallery(img: Image.Image) -> dict[str, str]:
    """Returns all gallery views as base64 PNG strings, keyed for the frontend."""
    return {
        "original": to_base64_png(img),
        "grayscale": to_base64_png(make_grayscale(img)),
        "edges": to_base64_png(make_edges(img)),
        "false_color": to_base64_png(make_false_color(img)),
        "false_color_alt": to_base64_png(make_false_color_alt(img)),
        "kmeans": to_base64_png(make_kmeans_segmentation(img)),
    }
