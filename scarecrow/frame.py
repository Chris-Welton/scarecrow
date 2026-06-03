import cv2
import numpy as np

THICKNESS = 0.035  # outward border, ~0.42" / 12" plate width
OVERLAP = 0.12  # inward overlap, ~0.72" / 6" plate height (past ~1.0" covers characters)


def frame_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Generate a hollow rectangular frame mask around a bbox."""
    x, y, w, h = bbox
    t = max(1, int(w * THICKNESS))
    o = max(1, int(h * OVERLAP))
    m = np.zeros(shape, dtype=np.uint8)
    cv2.rectangle(m, (x - t, y - t), (x + w + t, y + h + t), 1, -1)
    cv2.rectangle(m, (x + o, y + o), (x + w - o, y + h - o), 0, -1)
    return m


def apply_pattern(img, pattern, bboxes):
    """Overlay grayscale pattern onto frame regions. Modifies img in-place."""
    pr = cv2.resize(pattern, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    vals = (pr * 255).astype(np.uint8)
    for bbox in bboxes:
        mask = frame_mask(img.shape[:2], bbox)
        for c in range(3):
            img[:, :, c] = np.where(mask, vals, img[:, :, c])
