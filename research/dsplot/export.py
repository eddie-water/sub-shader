"""PNG post-processing shared by every README figure exporter.

Matplotlib figures (``bbox_inches="tight"`` or fixed-canvas) and draw.io CLI
exports end up with different outer margins; this normalises them to the one
rule in ``style.README_EDGE_PAD_PX``: content bounding box plus a fixed pad.
"""
import numpy as np
from PIL import Image

from . import style


def crop_to_content(path: str, pad_px: int = style.README_EDGE_PAD_PX,
                    bg: str = style.BG_COLOR, threshold: int = 20) -> str:
    """Rewrite ``path`` so the figure's content sits ``pad_px`` from every edge.

    Works in both directions: trims surplus margin and grows a border for a
    figure that bled to its edge. Content = pixels whose luminance differs from
    ``bg`` by more than ``threshold``.
    """
    im = Image.open(path).convert("RGB")
    bg_rgb = tuple(int(bg.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    a = np.asarray(im).astype(int)
    diff = np.abs(a - np.array(bg_rgb)).max(axis=2)
    ys, xs = np.where(diff > threshold)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    out = Image.new("RGB", (x1 - x0 + 2 * pad_px, y1 - y0 + 2 * pad_px), bg_rgb)
    out.paste(im.crop((x0, y0, x1, y1)), (pad_px, pad_px))
    out.save(path)
    return path


def fit_readme_width(path: str, shown_px: int = style.README_SHOWN_PX) -> str:
    """Crop to content, then scale so the PNG is README_SUPERSAMPLE x its shown
    width: one exported pixel means the same on-screen size in every figure."""
    crop_to_content(path)
    im = Image.open(path).convert("RGB")
    pad = style.README_EDGE_PAD_PX
    content = im.crop((pad, pad, im.width - pad, im.height - pad))
    w = shown_px * style.README_SUPERSAMPLE - 2 * pad
    content = content.resize((w, round(content.height * w / content.width)), Image.LANCZOS)
    bg_rgb = tuple(int(style.BG_COLOR.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    out = Image.new("RGB", (w + 2 * pad, content.height + 2 * pad), bg_rgb)
    out.paste(content, (pad, pad))
    out.save(path)
    return path
