import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tobii_pytracker.utils.bbox_generator import (  # noqa: E402
    ImageBoundingBoxGenerator,
    bbox_from_top_left,
    bbox_to_top_left,
    contains_point,
    evaluate_gaze_bbox_mapping,
)


def _write_synthetic_image(path: Path) -> None:
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    image[:, :] = [25, 25, 25]
    image[10:35, 12:45] = [240, 240, 240]
    image[45:70, 70:110] = [200, 40, 40]
    Image.fromarray(image).save(path)


def test_bbox_coordinate_round_trip():
    bbox = bbox_from_top_left(20, 10, 80, 50, area_x=120, area_y=80)

    assert bbox == {"cx": -10.0, "cy": 10.0, "w": 60.0, "h": 40.0}
    assert bbox_to_top_left(bbox, area_x=120, area_y=80) == (20.0, 10.0, 80.0, 50.0)
    assert contains_point(bbox, -10, 10)
    assert not contains_point(bbox, 55, 35)


def test_grid_generator_covers_aoi():
    boxes = ImageBoundingBoxGenerator((120, 80), method="grid").grid(grid_x=3, grid_y=2)

    assert len(boxes) == 6
    assert boxes[0]["bbox"] == {"cx": -40.0, "cy": 20.0, "w": 40.0, "h": 40.0}
    assert boxes[-1]["bbox"] == {"cx": 40.0, "cy": -20.0, "w": 40.0, "h": 40.0}

    result = evaluate_gaze_bbox_mapping(
        gaze_samples=[(-59.0, 39.0), (0.0, 0.0), (59.0, -39.0)],
        bbox_records=boxes,
    )
    assert result["coverage"] == 1.0
    assert result["mapped_samples"] == 3


def test_contrast_and_saliency_alias_generate_boxes(tmp_path):
    image_path = tmp_path / "stimulus.png"
    _write_synthetic_image(image_path)

    generator = ImageBoundingBoxGenerator((120, 80), method="contrast")
    contrast_boxes = generator.generate(str(image_path), percentile=80, min_area_ratio=0.0005)
    saliency_boxes = generator.generate(str(image_path), method="saliency", percentile=80, min_area_ratio=0.0005)

    assert contrast_boxes
    assert saliency_boxes
    assert all("bbox" in record for record in contrast_boxes)
    assert all(record["class"] == "saliency" for record in saliency_boxes)


def test_superpixel_generator_outputs_aoi_aligned_boxes(tmp_path):
    image_path = tmp_path / "stimulus.png"
    _write_synthetic_image(image_path)

    boxes = ImageBoundingBoxGenerator((120, 80), method="superpixel").generate(
        str(image_path),
        target_regions=16,
        max_regions=32,
    )

    assert boxes
    assert len(boxes) <= 32
    for record in boxes:
        x_min, y_min, x_max, y_max = bbox_to_top_left(record["bbox"], 120, 80)
        assert 0 <= x_min <= 120
        assert 0 <= x_max <= 120
        assert 0 <= y_min <= 80
        assert 0 <= y_max <= 80
        assert record["bbox"]["w"] > 0
        assert record["bbox"]["h"] > 0


def test_evaluate_gaze_bbox_mapping_counts_hits():
    boxes = [
        {"class": "left", "conf": 1.0, "bbox": {"cx": -30.0, "cy": 0.0, "w": 40.0, "h": 40.0}},
        {"class": "right", "conf": 1.0, "bbox": {"cx": 30.0, "cy": 0.0, "w": 40.0, "h": 40.0}},
    ]
    gaze = [
        {"gaze_x": -30, "gaze_y": 0},
        {"position": [30, 0]},
        (0, 50),
        {"not_a_gaze": True},
    ]

    result = evaluate_gaze_bbox_mapping(gaze, boxes)

    assert result["valid_samples"] == 3
    assert result["mapped_samples"] == 2
    assert result["coverage"] == 2 / 3
    assert result["boxes"][0]["gaze_hits"] == 1
    assert result["boxes"][1]["gaze_hits"] == 1
