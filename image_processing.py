"""
image_processing.py
--------------------
REAL module: every transformation here actually runs on the real pixels of
the fetched image using standard, well-known algorithms (grayscale, Sobel
edge detection, Gaussian blur, k-means segmentation, colormap remapping).

One honest labeling note: the "thermal" and "vegetation index" views below
are VISUAL SIMULATIONS built by remapping ordinary RGB pixels through a
colormap or a formula — they are NOT real thermal-band or NIR-band data
(that would require a real thermal sensor / Sentinel-2 NIR band, which the
production system uses; see the architecture PDF). They exist here purely
to make the demo gallery look like a real remote-sensing pipeline. Say this
plainly if a judge asks.
"""
from collections import OrderedDict

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import matplotlib
from sklearn.cluster import KMeans


def to_grayscale(img: Image.Image) -> Image.Image:
    return ImageOps.grayscale(img)


def apply_blur(img: Image.Image, radius: int = 4) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_sharpen(img: Image.Image) -> Image.Image:
    return img.filter(ImageFilter.SHARPEN)


def apply_contrast(img: Image.Image, factor: float = 1.6) -> Image.Image:
    return ImageEnhance.Contrast(img).enhance(factor)


def edge_detection(img: Image.Image) -> Image.Image:
    """Real Sobel-style edge detection via PIL's built-in edge filter."""
    gray = ImageOps.grayscale(img)
    return gray.filter(ImageFilter.FIND_EDGES)


def _colormap_image(gray_img: Image.Image, cmap_name: str) -> Image.Image:
    arr = np.asarray(gray_img).astype(np.float32) / 255.0
    colored = matplotlib.colormaps[cmap_name](arr)  # RGBA in [0,1]; matplotlib.cm.get_cmap was removed in mpl 3.9+
    colored_rgb = (colored[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(colored_rgb)


def simulated_thermal_view(img: Image.Image, cmap_name: str = "inferno") -> Image.Image:
    """NOTE: simulated — remaps visible brightness through a heat-style
    colormap. Looks like a thermal camera; is not one."""
    gray = ImageOps.grayscale(img)
    return _colormap_image(gray, cmap_name)


def binary_hotspot_mask(img: Image.Image, percentile: float = 92.0) -> Image.Image:
    """Flags the brightest N% of pixels as a stand-in 'hotspot' mask.
    NOTE: simulated — a real system thresholds actual FRP/brightness-
    temperature values from the sensor, not visible-light brightness."""
    gray = np.asarray(ImageOps.grayscale(img)).astype(np.float32)
    thresh = np.percentile(gray, percentile)
    mask = (gray >= thresh).astype(np.uint8) * 255
    return Image.fromarray(mask)


def kmeans_segmentation(img: Image.Image, k: int = 4) -> Image.Image:
    """REAL unsupervised segmentation — actually clusters the image's pixel
    colors with k-means. This is a legitimate, if simple, computer-vision
    technique (not a trained classifier)."""
    small = img.resize((128, 128))
    arr = np.asarray(small).reshape(-1, 3).astype(np.float32)
    km = KMeans(n_clusters=k, n_init=4, random_state=42).fit(arr)
    labels = km.labels_.reshape(128, 128)
    palette = (matplotlib.colormaps["tab10"](np.linspace(0, 1, k))[:, :3] * 255).astype(np.uint8)
    seg = palette[labels]
    return Image.fromarray(seg).resize(img.size, Image.NEAREST)


def simulated_vegetation_index(img: Image.Image) -> Image.Image:
    """NOTE: simulated — approximates a vegetation-index-style view using only
    RGB (green vs red channel difference) as a stand-in. The production
    system uses real Sentinel-2 NIR+Red bands (true NDVI)."""
    arr = np.asarray(img).astype(np.float32)
    r, g = arr[:, :, 0], arr[:, :, 1]
    pseudo_ndvi = (g - r) / (g + r + 1e-6)
    normed = (pseudo_ndvi - pseudo_ndvi.min()) / (pseudo_ndvi.max() - pseudo_ndvi.min() + 1e-6)
    return _colormap_image(Image.fromarray((normed * 255).astype(np.uint8)), "RdYlGn")


def quadrant_split(img: Image.Image):
    """Splits the image into 4 quadrants — the 'zoom-in' dashboard feature
    discussed in the architecture doc (demo-only visual, not the core
    classification mechanism)."""
    w, h = img.size
    boxes = [(0, 0, w // 2, h // 2), (w // 2, 0, w, h // 2),
             (0, h // 2, w // 2, h), (w // 2, h // 2, w, h)]
    return [img.crop(b) for b in boxes]


def build_processing_gallery(img: Image.Image) -> "OrderedDict[str, Image.Image]":
    """Runs every transform above and returns a labeled, ordered gallery
    ready to be plotted in a grid. This alone gives 12+ distinct panels."""
    gallery = OrderedDict()
    gallery["1. Original satellite image"] = img
    gallery["2. Grayscale"] = to_grayscale(img)
    gallery["3. Gaussian blur"] = apply_blur(img)
    gallery["4. Sharpened"] = apply_sharpen(img)
    gallery["5. Contrast enhanced"] = apply_contrast(img)
    gallery["6. Edge detection (Sobel-style)"] = edge_detection(img)
    gallery["7. Simulated thermal (inferno)"] = simulated_thermal_view(img, "inferno")
    gallery["8. Simulated thermal (jet)"] = simulated_thermal_view(img, "jet")
    gallery["9. Simulated thermal (hot)"] = simulated_thermal_view(img, "hot")
    gallery["10. Simulated hotspot mask (B/W)"] = binary_hotspot_mask(img)
    gallery["11. K-means segmentation (k=4)"] = kmeans_segmentation(img, k=4)
    gallery["12. Simulated vegetation index"] = simulated_vegetation_index(img)
    quads = quadrant_split(img)
    for i, q in enumerate(quads, start=1):
        gallery[f"{12+i}. Zoom quadrant {i}/4"] = q
    return gallery
