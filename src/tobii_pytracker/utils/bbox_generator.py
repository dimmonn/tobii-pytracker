"""Image bounding-box generation and gaze-to-box evaluation utilities.

The experiment stores screenshots cropped to the AOI (area of interest).  The
helpers in this module therefore use the same AOI coordinate system as the rest
of the toolkit: bounding boxes are represented as ``{"cx", "cy", "w", "h"}``,
where ``cx`` and ``cy`` are measured in pixels from the AOI centre and positive
``y`` points upwards.

No GUI, PsychoPy window, eye tracker, OpenCV, or scikit-image dependency is
required here.  This makes the implementation usable both during experiments
and in post-hoc tests/analysis.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
from PIL import Image

BBoxMethod = Literal["grid", "contrast", "superpixel"]


@dataclass(frozen=True)
class BBox:
    """A bbox in AOI-centred PsychoPy pixel coordinates."""

    cx: float
    cy: float
    w: float
    h: float

    def as_dict(self) -> Dict[str, float]:
        return {"cx": float(self.cx), "cy": float(self.cy), "w": float(self.w), "h": float(self.h)}


def bbox_from_top_left(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    area_x: int,
    area_y: int,
) -> Dict[str, float]:
    """Convert top-left-origin pixel coordinates into the toolkit bbox format."""

    x_min = float(np.clip(x_min, 0.0, area_x))
    x_max = float(np.clip(x_max, 0.0, area_x))
    y_min = float(np.clip(y_min, 0.0, area_y))
    y_max = float(np.clip(y_max, 0.0, area_y))

    if x_max < x_min:
        x_min, x_max = x_max, x_min
    if y_max < y_min:
        y_min, y_max = y_max, y_min

    width = max(1.0, x_max - x_min)
    height = max(1.0, y_max - y_min)
    return BBox(
        cx=x_min + width / 2.0 - area_x / 2.0,
        cy=area_y / 2.0 - (y_min + height / 2.0),
        w=width,
        h=height,
    ).as_dict()


def bbox_to_top_left(bbox: Mapping[str, float], area_x: int, area_y: int) -> Tuple[float, float, float, float]:
    """Convert a toolkit bbox into top-left-origin AOI pixel coordinates."""

    width = float(bbox["w"])
    height = float(bbox["h"])
    x_center = float(bbox["cx"]) + area_x / 2.0
    y_center = area_y / 2.0 - float(bbox["cy"])
    return (
        x_center - width / 2.0,
        y_center - height / 2.0,
        x_center + width / 2.0,
        y_center + height / 2.0,
    )


def contains_point(bbox: Mapping[str, float], x: float, y: float) -> bool:
    """Return whether a centre-origin gaze point falls inside a bbox."""

    half_w = float(bbox["w"]) / 2.0
    half_h = float(bbox["h"]) / 2.0
    return (
        float(bbox["cx"]) - half_w <= float(x) <= float(bbox["cx"]) + half_w
        and float(bbox["cy"]) - half_h <= float(y) <= float(bbox["cy"]) + half_h
    )


def _load_resized_rgb(image_path: str, area_size: Tuple[int, int]) -> np.ndarray:
    """Load an image as RGB pixels resized exactly to the screenshot/AOI size."""

    with Image.open(image_path) as image:
        return np.asarray(image.convert("RGB").resize(area_size, Image.Resampling.BILINEAR), dtype=np.uint8)


def _luminance(rgb: np.ndarray) -> np.ndarray:
    rgb_f = rgb.astype(np.float32)
    return 0.2126 * rgb_f[..., 0] + 0.7152 * rgb_f[..., 1] + 0.0722 * rgb_f[..., 2]


def _bbox_record(label: str, confidence: float, bbox: Mapping[str, float], **extra: Any) -> Dict[str, Any]:
    record: Dict[str, Any] = {"class": label, "conf": float(confidence), "bbox": dict(bbox)}
    record.update(extra)
    return record


class ImageBoundingBoxGenerator:
    """Generate AOI-aligned bboxes for image stimuli.

    Methods
    -------
    ``grid``
        Uniform coverage baseline. Useful as a conservative fallback.
    ``contrast``
        Finds connected components in a local-gradient/contrast mask. This is a
        dependency-safe saliency approximation.
    ``superpixel``
        Splits the image into compact, edge-aware connected regions by combining
        local contrast seeds and grid partitioning. It is intentionally less
        fragile than importing ``skimage.segmentation.slic`` at runtime because
        scikit-image is not a declared project dependency.
    """

    def __init__(self, area_size: Tuple[int, int], method: BBoxMethod = "superpixel"):
        self.area_x = int(area_size[0])
        self.area_y = int(area_size[1])
        self.method: BBoxMethod = method
        if self.area_x <= 0 or self.area_y <= 0:
            raise ValueError("area_size must contain positive width and height")

    def generate(self, image_path: str, method: BBoxMethod | None = None, **kwargs: Any) -> List[Dict[str, Any]]:
        selected = method or self.method
        if selected == "grid":
            return self.grid(**kwargs)
        if selected == "contrast":
            return self.contrast(image_path, **kwargs)
        if selected == "superpixel":
            return self.superpixels(image_path, **kwargs)
        raise ValueError(f"Unknown bbox generation method: {selected}")

    def grid(self, grid_x: int = 3, grid_y: int = 3) -> List[Dict[str, Any]]:
        if grid_x <= 0 or grid_y <= 0:
            raise ValueError("grid_x and grid_y must be positive")
        cell_w = self.area_x / float(grid_x)
        cell_h = self.area_y / float(grid_y)
        boxes: List[Dict[str, Any]] = []
        for row in range(grid_y):
            for col in range(grid_x):
                bbox = bbox_from_top_left(
                    col * cell_w,
                    row * cell_h,
                    (col + 1) * cell_w,
                    (row + 1) * cell_h,
                    self.area_x,
                    self.area_y,
                )
                boxes.append(_bbox_record("grid", 1.0, bbox, row=row, col=col))
        return boxes

    def contrast(
        self,
        image_path: str,
        percentile: float = 88.0,
        min_area_ratio: float = 0.002,
        max_regions: int = 40,
        padding: int = 2,
    ) -> List[Dict[str, Any]]:
        rgb = _load_resized_rgb(image_path, (self.area_x, self.area_y))
        gray = _luminance(rgb)
        score = self._contrast_score(gray)
        threshold = np.percentile(score, percentile)
        mask = score >= threshold
        min_area = max(4, int(self.area_x * self.area_y * min_area_ratio))
        components = self._components_from_mask(mask, score, min_area=min_area, padding=padding)
        components.sort(key=lambda item: item["conf"], reverse=True)
        return components[:max_regions]

    def superpixels(
        self,
        image_path: str,
        target_regions: int = 64,
        min_area_ratio: float = 0.0015,
        edge_percentile: float = 82.0,
        max_regions: int = 80,
    ) -> List[Dict[str, Any]]:
        """Return compact edge-aware regions approximating superpixels.

        The implementation uses the image after AOI resize, computes an edge mask
        from luminance contrast, then runs connected components inside grid cells.
        The resulting boxes are stable, deterministic, and match the screenshot
        coordinate system used for gaze samples.
        """

        rgb = _load_resized_rgb(image_path, (self.area_x, self.area_y))
        gray = _luminance(rgb)
        score = self._contrast_score(gray)
        edge_threshold = np.percentile(score, edge_percentile)
        non_edge = score < edge_threshold

        grid_cols = max(1, int(round(np.sqrt(target_regions * self.area_x / self.area_y))))
        grid_rows = max(1, int(round(target_regions / grid_cols)))
        cell_w = int(np.ceil(self.area_x / grid_cols))
        cell_h = int(np.ceil(self.area_y / grid_rows))
        min_area = max(4, int(self.area_x * self.area_y * min_area_ratio))

        boxes: List[Dict[str, Any]] = []
        region_id = 0
        for row in range(grid_rows):
            for col in range(grid_cols):
                x0 = col * cell_w
                y0 = row * cell_h
                x1 = min(self.area_x, x0 + cell_w)
                y1 = min(self.area_y, y0 + cell_h)
                if x0 >= x1 or y0 >= y1:
                    continue

                local_mask = np.zeros_like(non_edge, dtype=bool)
                local_mask[y0:y1, x0:x1] = non_edge[y0:y1, x0:x1]
                local_components = self._components_from_mask(
                    local_mask,
                    score,
                    min_area=min_area,
                    padding=1,
                    label="superpixel",
                    start_region_id=region_id,
                )
                if local_components:
                    boxes.extend(local_components)
                    region_id += len(local_components)
                else:
                    # Preserve AOI coverage in very textured cells.
                    bbox = bbox_from_top_left(x0, y0, x1, y1, self.area_x, self.area_y)
                    boxes.append(_bbox_record("superpixel", 0.5, bbox, region_id=region_id))
                    region_id += 1

        boxes.sort(key=lambda item: (item["bbox"]["cy"], -item["bbox"]["cx"]), reverse=True)
        return boxes[:max_regions]

    @staticmethod
    def _contrast_score(gray: np.ndarray) -> np.ndarray:
        dy = np.zeros_like(gray, dtype=np.float32)
        dx = np.zeros_like(gray, dtype=np.float32)
        dy[1:, :] = np.abs(gray[1:, :] - gray[:-1, :])
        dx[:, 1:] = np.abs(gray[:, 1:] - gray[:, :-1])
        return dx + dy

    def _components_from_mask(
        self,
        mask: np.ndarray,
        score: np.ndarray,
        min_area: int,
        padding: int = 0,
        label: str = "contrast",
        start_region_id: int = 0,
    ) -> List[Dict[str, Any]]:
        visited = np.zeros(mask.shape, dtype=bool)
        boxes: List[Dict[str, Any]] = []
        region_id = start_region_id
        height, width = mask.shape

        for start_y, start_x in np.argwhere(mask):
            if visited[start_y, start_x]:
                continue
            queue: deque[Tuple[int, int]] = deque([(int(start_y), int(start_x))])
            visited[start_y, start_x] = True
            xs: List[int] = []
            ys: List[int] = []
            values: List[float] = []

            while queue:
                y, x = queue.popleft()
                xs.append(x)
                ys.append(y)
                values.append(float(score[y, x]))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))

            if len(xs) < min_area:
                continue
            x_min = max(0, min(xs) - padding)
            x_max = min(width, max(xs) + 1 + padding)
            y_min = max(0, min(ys) - padding)
            y_max = min(height, max(ys) + 1 + padding)
            bbox = bbox_from_top_left(x_min, y_min, x_max, y_max, self.area_x, self.area_y)
            confidence = float(np.mean(values) / (np.max(score) + 1e-6))
            boxes.append(_bbox_record(label, confidence, bbox, region_id=region_id, area=len(xs)))
            region_id += 1

        return boxes


def _extract_gaze_xy(sample: Any) -> Tuple[float, float] | None:
    """Read a gaze point from common tuple/dict formats used in saved CSV rows."""

    if isinstance(sample, Mapping):
        for x_key, y_key in (("x", "y"), ("gaze_x", "gaze_y"), ("cx", "cy"), ("pos_x", "pos_y")):
            if x_key in sample and y_key in sample:
                return float(sample[x_key]), float(sample[y_key])
        if "position" in sample and isinstance(sample["position"], Sequence) and len(sample["position"]) >= 2:
            return float(sample["position"][0]), float(sample["position"][1])
    elif isinstance(sample, Sequence) and not isinstance(sample, (str, bytes)) and len(sample) >= 2:
        return float(sample[0]), float(sample[1])
    return None


def evaluate_gaze_bbox_mapping(
    gaze_samples: Iterable[Any],
    bbox_records: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Evaluate how well gaze points map to generated bboxes.

    Returns aggregate coverage plus per-box hit counts.  ``coverage`` is the
    fraction of valid gaze samples that landed inside at least one generated box.
    This can be used post-hoc to compare ``grid``, ``contrast``, ``superpixel``
    and any custom model-generated boxes for the same screenshot.
    """

    boxes = [record for record in bbox_records if "bbox" in record]
    per_box_hits = [0 for _ in boxes]
    valid_samples = 0
    mapped_samples = 0

    for raw_sample in gaze_samples:
        xy = _extract_gaze_xy(raw_sample)
        if xy is None:
            continue
        valid_samples += 1
        hit_any = False
        for idx, record in enumerate(boxes):
            if contains_point(record["bbox"], xy[0], xy[1]):
                per_box_hits[idx] += 1
                hit_any = True
        if hit_any:
            mapped_samples += 1

    annotated_boxes: List[Dict[str, Any]] = []
    for record, hits in zip(boxes, per_box_hits):
        annotated = dict(record)
        annotated["gaze_hits"] = hits
        annotated_boxes.append(annotated)

    coverage = mapped_samples / valid_samples if valid_samples else 0.0
    return {
        "valid_samples": valid_samples,
        "mapped_samples": mapped_samples,
        "coverage": coverage,
        "boxes": annotated_boxes,
    }
