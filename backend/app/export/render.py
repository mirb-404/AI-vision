"""Draw annotations onto a copy of the image (for "annotated image" downloads)."""

import io
from pathlib import Path

from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont

from ..tables import Annotation, Image, Label

FALLBACK_COLOR = "#94a3b8"


def _rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def render_annotated(path: Path, image: Image, annotations: list[Annotation], labels: dict[int, Label]) -> bytes:
    """JPEG bytes of the image with boxes / polygons, a light fill, and a label tag per annotation."""
    with PILImage.open(path) as src:
        base = src.convert("RGB")
    overlay = PILImage.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Scale strokes and text with the image so they stay visible on large photos.
    side = max(base.size)
    width = max(2, round(side / 300))
    font = ImageFont.load_default(size=max(12, round(side / 45)))

    for a in annotations:
        label = labels.get(a.label_id)
        r, g, b = _rgb(label.color if label else FALLBACK_COLOR)
        g_ = a.geometry
        if a.kind == "box":
            x, y, w, h = g_["x"], g_["y"], g_["w"], g_["h"]
            draw.rectangle([x, y, x + w, y + h], fill=(r, g, b, 60), outline=(r, g, b, 255), width=width)
            tx, ty = x, y
        else:
            pts = g_["points"]
            xy = list(zip(pts[0::2], pts[1::2]))
            draw.polygon(xy, fill=(r, g, b, 70))
            draw.line(xy + [xy[0]], fill=(r, g, b, 255), width=width, joint="curve")
            tx, ty = min(pts[0::2]), min(pts[1::2])

        text = label.name if label else "?"
        if a.score is not None:
            text += f" {a.score * 100:.0f}%"
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        pad = max(2, width)
        tw, th = right - left + 2 * pad, bottom - top + 2 * pad
        ty = ty - th if ty >= th else ty  # above the shape, or inside when at the top edge
        draw.rectangle([tx, ty, tx + tw, ty + th], fill=(r, g, b, 230))
        draw.text((tx + pad - left, ty + pad - top), text, fill=(255, 255, 255, 255), font=font)

    out = PILImage.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=90)
    return buf.getvalue()
