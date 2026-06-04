from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from scarecrow.export import SVG_NS, export_svg


def _parse_svg(path: Path):
    root = ET.parse(path).getroot()
    ns = {"svg": SVG_NS}
    rects = root.findall(".//svg:rect", ns)
    group = root.find("svg:g", ns)
    return root, group, rects


class TestExportSvg:
    def test_writes_physical_svg_with_run_rects(self, tmp_path):
        path = tmp_path / "frame.svg"
        pattern = np.full((1, 1), 0.5, dtype=np.float32)

        result = export_svg(pattern, (20, 30), (10, 8, 10, 5), path)

        root, group, rects = _parse_svg(path)
        assert root.tag == f"{{{SVG_NS}}}svg"
        assert group.attrib["shape-rendering"] == "crispEdges"
        assert root.attrib["width"] == "15.6in"
        assert root.attrib["height"] == "9.6in"
        assert root.attrib["viewBox"] == "-1.2 -1.2 15.6 9.6"
        assert result.width_in == pytest.approx(15.6)
        assert result.height_in == pytest.approx(9.6)
        assert len(rects) == 12

        first = rects[0].attrib
        assert first["x"] == "-1.2"
        assert first["y"] == "-1.2"
        assert first["width"] == "15.6"
        assert first["height"] == "1.2"
        assert first["fill"] == "#7f7f7f"

    def test_preserves_inclusive_outer_bounds(self, tmp_path):
        path = tmp_path / "frame.svg"
        pattern = np.ones((1, 1), dtype=np.float32)

        export_svg(pattern, (20, 40), (10, 8, 20, 5), path)

        root, _, _ = _parse_svg(path)
        _, _, width, height = [float(v) for v in root.attrib["viewBox"].split()]
        sx = 12.0 / 20
        sy = 6.0 / 5
        assert width == pytest.approx(23 * sx)
        assert height == pytest.approx(8 * sy)

    def test_rejects_clipped_reference_without_writing(self, tmp_path):
        path = tmp_path / "nested" / "frame.svg"
        pattern = np.ones((1, 1), dtype=np.float32)

        with pytest.raises(ValueError, match="needs margin"):
            export_svg(pattern, (20, 30), (0, 8, 10, 5), path)

        assert not path.exists()
