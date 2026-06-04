import numpy as np

from scarecrow.frame import OVERLAP, THICKNESS, apply_pattern, frame_bounds, frame_margins, frame_mask, pattern_values


class TestFrameHelpers:
    def test_frame_margins(self):
        assert frame_margins((100, 50, 200, 100)) == (int(200 * THICKNESS), int(100 * OVERLAP))

    def test_frame_margins_minimum(self):
        assert frame_margins((20, 15, 10, 10)) == (1, 1)

    def test_frame_bounds_are_inclusive(self):
        outer, inner = frame_bounds((100, 50, 200, 100))
        assert outer == (93, 43, 307, 157)
        assert inner == (112, 62, 288, 138)

    def test_pattern_values_uses_nearest_neighbor(self):
        pattern = np.array([[0.0, 0.25], [0.5, 1.0]], dtype=np.float32)
        vals = pattern_values(pattern, (4, 4))
        expected = np.array(
            [
                [0, 0, 63, 63],
                [0, 0, 63, 63],
                [127, 127, 255, 255],
                [127, 127, 255, 255],
            ],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(vals, expected)


class TestFrameMask:
    def test_interior_clear(self):
        """Plate interior inside the overlap margin is fully clear."""
        bbox = (100, 50, 200, 100)
        x, y, w, h = bbox
        o = max(1, int(h * OVERLAP))
        m = frame_mask((300, 500), bbox)
        assert m[y + o : y + h - o, x + o : x + w - o].sum() == 0

    def test_frame_filled(self):
        """Frame region between outer edge and inner cutout is set."""
        bbox = (100, 50, 200, 100)
        x, y, w, h = bbox
        t = max(1, int(w * THICKNESS))
        m = frame_mask((300, 500), bbox)
        assert m[y - 1, x + w // 2] == 1
        assert m[y - t, x + w // 2] == 1

    def test_outside_clear(self):
        """Pixels beyond the outer frame edge are clear."""
        bbox = (100, 50, 200, 100)
        x, y, w, h = bbox
        t = max(1, int(w * THICKNESS))
        m = frame_mask((300, 500), bbox)
        assert m[y - t - 1, x + w // 2] == 0

    def test_small_bbox(self):
        """Small bboxes clamp thickness and overlap to 1px minimum."""
        m = frame_mask((50, 50), bbox=(20, 15, 10, 10))
        assert m.any()
        assert m[20, 25] == 0

    def test_bbox_at_edge(self):
        """Bbox touching image edge clips frame without crashing."""
        m = frame_mask((100, 100), bbox=(0, 0, 50, 50))
        assert m.any()
        assert m.shape == (100, 100)


class TestApplyPattern:
    def test_masked_pixels_take_pattern(self):
        img = np.full((60, 80, 3), 10, dtype=np.uint8)
        pattern = np.ones((1, 1), dtype=np.float32)
        bbox = (20, 20, 20, 10)

        apply_pattern(img, pattern, [bbox])

        mask = frame_mask(img.shape[:2], bbox)
        assert (img[mask.astype(bool)] == 255).all()
        assert (img[~mask.astype(bool)] == 10).all()
