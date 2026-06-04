from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from scarecrow.frame import frame_bounds, frame_mask, pattern_values

PLATE_WIDTH_IN = 12.0
PLATE_HEIGHT_IN = 6.0
SVG_NS = "http://www.w3.org/2000/svg"


@dataclass
class ExportResult:
    width_in: float
    height_in: float


def _fmt(v: float) -> str:
    return f"{v:.10g}"


def _iter_runs(mask: np.ndarray, vals: np.ndarray):
    for y in range(mask.shape[0]):
        xs = np.flatnonzero(mask[y])
        if len(xs) == 0:
            continue

        start = last = int(xs[0])
        gray = int(vals[y, start])
        for x in xs[1:]:
            x = int(x)
            v = int(vals[y, x])
            if x != last + 1 or v != gray:
                yield y, start, last - start + 1, gray
                start = x
                gray = v
            last = x
        yield y, start, last - start + 1, gray


def _check_not_clipped(image_shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> None:
    h, w = image_shape
    left, top, right, bottom = frame_bounds(bbox)[0]
    if left < 0 or top < 0 or right >= w or bottom >= h:
        raise ValueError("Reference image needs margin around the detected plate before exporting")


def export_svg(
    pattern: np.ndarray,
    image_shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
    output_path: str | Path,
) -> ExportResult:
    """Export an SVG frame template for a detected plate bbox."""
    _check_not_clipped(image_shape, bbox)

    x, y, bw, bh = bbox
    sx = PLATE_WIDTH_IN / bw
    sy = PLATE_HEIGHT_IN / bh

    mask = frame_mask(image_shape, bbox)
    vals = pattern_values(pattern, image_shape)
    ys, xs = np.nonzero(mask)

    min_x = (int(xs.min()) - x) * sx
    min_y = (int(ys.min()) - y) * sy
    max_x = (int(xs.max()) + 1 - x) * sx
    max_y = (int(ys.max()) + 1 - y) * sy
    width = max_x - min_x
    height = max_y - min_y

    ET.register_namespace("", SVG_NS)
    root = ET.Element(
        f"{{{SVG_NS}}}svg",
        {
            "width": f"{_fmt(width)}in",
            "height": f"{_fmt(height)}in",
            "viewBox": f"{_fmt(min_x)} {_fmt(min_y)} {_fmt(width)} {_fmt(height)}",
        },
    )
    group = ET.SubElement(root, f"{{{SVG_NS}}}g", {"shape-rendering": "crispEdges"})

    for py, px, run_length, gray in _iter_runs(mask, vals):
        fill = f"#{gray:02x}{gray:02x}{gray:02x}"
        ET.SubElement(
            group,
            f"{{{SVG_NS}}}rect",
            {
                "x": _fmt((px - x) * sx),
                "y": _fmt((py - y) * sy),
                "width": _fmt(run_length * sx),
                "height": _fmt(sy),
                "fill": fill,
            },
        )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(out, encoding="utf-8", xml_declaration=True)
    return ExportResult(width_in=width, height_in=height)
