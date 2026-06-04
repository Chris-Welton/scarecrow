import cv2
import numpy as np

THICKNESS = 0.035  # outward border, ~0.42" / 12" plate width
OVERLAP = 0.12  # inward overlap, ~0.72" / 6" plate height (past ~1.0" covers characters)


def frame_margins(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    """Return outward thickness and inward overlap for a plate bbox."""
    _, _, w, h = bbox
    t = max(1, int(w * THICKNESS))
    o = max(1, int(h * OVERLAP))
    return t, o


def frame_bounds(
    bbox: tuple[int, int, int, int],
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    """Return inclusive outer and inner frame bounds."""
    x, y, w, h = bbox
    t, o = frame_margins(bbox)
    return (x - t, y - t, x + w + t, y + h + t), (x + o, y + o, x + w - o, y + h - o)


def frame_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Generate a hollow rectangular frame mask around a bbox."""
    outer, inner = frame_bounds(bbox)
    m = np.zeros(shape, dtype=np.uint8)
    cv2.rectangle(m, outer[:2], outer[2:], 1, -1)
    cv2.rectangle(m, inner[:2], inner[2:], 0, -1)
    return m


def pattern_values(pattern: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize pattern to image shape and convert to uint8 grayscale."""
    pr = cv2.resize(pattern, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (pr * 255).astype(np.uint8)


def apply_pattern(img, pattern, bboxes):
    """Overlay grayscale pattern onto frame regions. Modifies img in-place."""
    vals = pattern_values(pattern, img.shape[:2])
    for bbox in bboxes:
        mask = frame_mask(img.shape[:2], bbox)
        for c in range(3):
            img[:, :, c] = np.where(mask, vals, img[:, :, c])
