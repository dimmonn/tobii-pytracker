# datasets_full.py
import os
import random
import math
import importlib
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from psychopy import visual
from PIL import Image

from tobii_pytracker.utils.bbox_generator import (
    ImageBoundingBoxGenerator,
    bbox_from_top_left,
)
from tobii_pytracker.utils.custom_logger import CustomLogger


# -------------------------
# Helpers
# -------------------------
def _to_centered_bbox_from_tl(x_min_px: float, y_min_px: float, x_max_px: float, y_max_px: float,
                              area_x: int, area_y: int) -> Dict[str, float]:
    """
    Convert bbox in top-left pixel coords (image coordinates where (0,0) is top-left)
    into center-origin pixel coords where (0,0) is center of AOI and y positive is up.

    Returns dict: {"cx":..., "cy":..., "w":..., "h":...} in pixels (not normalized).
    """
    return bbox_from_top_left(x_min_px, y_min_px, x_max_px, y_max_px, area_x, area_y)


def _image_load_size(path: str) -> Tuple[int, int]:
    """Return (width, height) of image at path using PIL."""
    with Image.open(path) as im:
        return im.size  # width, height


# -------------------------
# Base dataset
# -------------------------
class CustomDataset:
    def __init__(self, config: Any, calculate_bboxes: bool = False):
        self.config = config
        self.dataset_path = config.get_dataset_path()
        self.logger = CustomLogger("debug", __name__).logger
        self.classes: List[str] = []
        self.data: List[Dict[str, Any]] = []
        self.calculate_bboxes = calculate_bboxes

    def prepare_data(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def get_classes(self) -> List[str]:
        return self.classes

    @property
    def is_text(self) -> bool:
        return isinstance(self, TextDataset)

    def draw_stimulus(self, window: visual.Window, sample: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


# -------------------------
# TextDataset
# -------------------------
class TextDataset(CustomDataset):
    """
    Text dataset that draws text into a PsychoPy window and returns exact per-line and per-word bboxes
    in center-origin pixel coordinates. The draw_stimulus method draws text and returns:
      {"words": [{"word":str,"conf":1.0,"bbox":{cx,cy,w,h}}...],
       "lines": [{"text":str,"conf":1.0,"bbox":{...}}...]}
    """

    def __init__(self, config: Any, calculate_bboxes: bool = False):
        super().__init__(config, calculate_bboxes)
        self.text_cfg = self.config.get_text_dataset_config()
        self.font_height = int(self.text_cfg.get("font_height", 35))
        self.wrap_frac: float = float(self.text_cfg.get("wrap_fraction", 0.95))
        self._load_data()

    def _load_data(self):
        if not self.dataset_path.endswith(".csv"):
            raise ValueError("TextDataset requires CSV file")
        df = pd.read_csv(self.dataset_path, header=0)
        label_col = self.text_cfg["label_column_name"]
        text_col = self.text_cfg["text_column_name"]
        self.classes = [str(c).lower() for c in df[label_col].unique()]
        self.classes.append("none")
        self.data = [{"class": str(r[label_col]).lower(), "data": str(r[text_col])}
                     for _, r in df.iterrows()]

    def draw_stimulus(self, window: visual.Window, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Draw text into provided PsychoPy window and compute per-line & per-word bboxes."""
        text = str(sample["data"])
        area_x, area_y = self.config.get_area_of_interest_size()
        wrap_width = int(area_x * self.wrap_frac)

        paragraph = visual.TextStim(
            win=window,
            text=text,
            pos=(0, 0),
            height=self.font_height,
            wrapWidth=wrap_width,
            alignText="left",
            color="white"
        )
        paragraph.draw()
        window.flip()

        raw_lines = text.split("\n")
        words_all = []
        for rl in raw_lines:
            for w in rl.split():
                words_all.append(w)

        word_dims: List[Tuple[str, float, float]] = []
        for w in words_all:
            ts = visual.TextStim(win=window, text=w, height=self.font_height, wrapWidth=None)
            w_w, w_h = ts.boundingBox
            word_dims.append((w, float(w_w), float(w_h)))

        space_ts = visual.TextStim(win=window, text=" ", height=self.font_height)
        space_w = float(space_ts.boundingBox[0] or (self.font_height * 0.3))

        lines: List[List[Tuple[str, float, float]]] = []
        cur_line: List[Tuple[str, float, float]] = []
        cur_width = 0.0
        for w, w_w, w_h in word_dims:
            add_w = w_w if cur_width == 0 else (space_w + w_w)
            if cur_width + add_w <= wrap_width or cur_width == 0:
                cur_line.append((w, w_w, w_h))
                cur_width = cur_width + add_w if cur_width != 0 else w_w
            else:
                lines.append(cur_line)
                cur_line = [(w, w_w, w_h)]
                cur_width = w_w
        if cur_line:
            lines.append(cur_line)

        line_heights = [max((h for (_, _, h) in line), default=self.font_height) for line in lines]
        total_text_height = sum(line_heights)
        top_y = (area_y - total_text_height) / 2.0

        line_widths = []
        for line in lines:
            lw = 0.0
            for i, (_, w_w, _) in enumerate(line):
                lw += w_w
                if i < len(line) - 1:
                    lw += space_w
            line_widths.append(lw)

        words_out: List[Dict[str, Any]] = []
        lines_out: List[Dict[str, Any]] = []
        y_cursor = top_y

        for li, line in enumerate(lines):
            lw = line_widths[li]
            line_h = line_heights[li]
            left_x = (area_x - lw) / 2.0

            line_bbox_centered = _to_centered_bbox_from_tl(left_x, y_cursor, left_x + lw, y_cursor + line_h, area_x, area_y)
            lines_out.append({
                "text": " ".join([w for (w, _, _) in line]),
                "conf": 1.0,
                "bbox": line_bbox_centered
            })

            x_cursor = left_x
            for word, w_w, w_h in line:
                w_y_min = y_cursor + (line_h - w_h) / 2.0
                word_bbox_centered = _to_centered_bbox_from_tl(x_cursor, w_y_min, x_cursor + w_w, w_y_min + w_h, area_x, area_y)
                words_out.append({"word": word, "conf": 1.0, "bbox": word_bbox_centered})
                x_cursor += w_w + space_w

            y_cursor += line_h

        return {"words": words_out, "lines": lines_out}


class ImageDataset(CustomDataset):
    """
    Image dataset that draws images and computes AOI-aligned bounding boxes.

    Bboxes are calculated against the same AOI size used for the cropped screenshot,
    so gaze points and object/region boxes share one coordinate system.
    """

    def __init__(self, config: Any, calculate_bboxes: bool = True):
        super().__init__(config, calculate_bboxes)
        self.image_cfg = self.config.get_image_dataset_config()
        self.default_detector = self.image_cfg.get("bbox_model", "superpixel")
        self.model = None

        if self.calculate_bboxes:
            try:
                cfg = self.config.get_bbox_model_config()
                ModelClass = getattr(
                    importlib.import_module(f"{cfg['folder']}.{cfg['module']}"),
                    cfg['class']
                )
                self.model = ModelClass(config, self)
            except Exception as e:
                self.logger.warning(
                    f"No valid custom bbox model found in config; using fallback detector "
                    f"({self.default_detector}). Error: {e}"
                )
                self.model = None

        self._load_data()

    def _load_data(self):
        self.classes = [
            d for d in os.listdir(self.dataset_path)
            if os.path.isdir(os.path.join(self.dataset_path, d))
        ]
        self.classes.append("none")

        samples = []
        for class_name in self.classes:
            class_path = os.path.join(self.dataset_path, class_name)
            if not os.path.isdir(class_path):
                continue

            for root, _, files in os.walk(class_path):
                for f in files:
                    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                        full_path = os.path.join(root, f)
                        sample = {"class": class_name, "data": full_path}
                        if self.calculate_bboxes:
                            sample["bboxes"] = self._compute_image_bboxes(full_path)
                        samples.append(sample)

        random.shuffle(samples)
        self.data = samples

    def _compute_image_bboxes(self, image_path: str) -> List[Dict[str, Any]]:
        if self.model is not None:
            return self._compute_bboxes_with_model(image_path)
        return self._compute_bboxes_fallback(image_path)

    def _compute_bboxes_with_model(self, image_path: str):
        """Handles model inference + AOI rescaling."""
        area_x, area_y = self.config.get_area_of_interest_size()
        detections_out = []

        try:
            preds = self.model.process(image_path)
            img_w, img_h = _image_load_size(image_path)
            sx, sy = area_x / img_w, area_y / img_h

            for class_name, conf, (x_min, y_min, x_max, y_max) in preds:
                detections_out.append({
                    "class": class_name,
                    "conf": float(conf),
                    "bbox": _to_centered_bbox_from_tl(
                        x_min * sx, y_min * sy,
                        x_max * sx, y_max * sy,
                        area_x, area_y
                    )
                })

        except Exception as e:
            self.logger.error(f"Error in model detection: {e}")

        return detections_out

    def _compute_bboxes_fallback(self, image_path: str):
        area_x, area_y = self.config.get_area_of_interest_size()
        generator = ImageBoundingBoxGenerator((area_x, area_y), method=self.default_detector)
        return generator.generate(image_path)

    def _detect_grid(self, image_path: str, grid_x: int = 3, grid_y: int = 3):
        area_x, area_y = self.config.get_area_of_interest_size()
        return ImageBoundingBoxGenerator((area_x, area_y), method="grid").grid(grid_x=grid_x, grid_y=grid_y)

    def _detect_superpixels(self, image_path: str, n_segments: int = 50):
        area_x, area_y = self.config.get_area_of_interest_size()
        return ImageBoundingBoxGenerator((area_x, area_y), method="superpixel").superpixels(image_path, target_regions=n_segments)

    def _detect_saliency(self, image_path: str, threshold: float = 0.6):
        area_x, area_y = self.config.get_area_of_interest_size()
        percentile = max(0.0, min(100.0, threshold * 100.0))
        return ImageBoundingBoxGenerator((area_x, area_y), method="contrast").contrast(image_path, percentile=percentile)

    def draw_stimulus(self, window: visual.Window, sample: Dict[str, Any]) -> Dict[str, Any]:
        area_x, area_y = self.config.get_area_of_interest_size()
        img_path = sample["data"]

        stim = visual.ImageStim(win=window, image=img_path, size=(area_x, area_y), pos=(0, 0))
        stim.draw()
        window.flip()

        bboxes = sample.get("bboxes", [])
        if self.calculate_bboxes and not bboxes:
            bboxes = self._compute_image_bboxes(img_path)

        return {"image_bboxes": bboxes}


# -------------------------
# TimeSeriesDataset
# -------------------------
class TimeSeriesDataset(CustomDataset):
    """
    Time series dataset: CSV rows are: index, v1, v2, ..., vN, class.
    draw_stimulus draws the time series as a polyline in the AOI and returns
    per-timestamp or per-window bboxes in center-origin pixel coordinates.
    """

    def __init__(self, config: Any, calculate_bboxes: bool = False, window_size: int = 1):
        super().__init__(config, calculate_bboxes)
        self.window_size = max(1, int(window_size))
        self._load_data()

    def _load_data(self):
        cfg = self.config.get_time_series_dataset_config()
        file_path = self.dataset_path
        label_col = cfg["label_column_name"]
        df = pd.read_csv(file_path)
        cols = df.columns.tolist()
        index_col = cols[0]
        ts_cols = [c for c in cols[1:] if c != label_col]
        self.classes = df[label_col].unique().tolist()
        self.classes.append("none")
        samples = []
        for _, row in df.iterrows():
            series = row[ts_cols].astype(float).to_numpy()
            samples.append({"class": row[label_col], "data": series, "index": row[index_col]})
        self.data = samples

    def _compute_timeseries_bboxes_from_series(self, series: np.ndarray) -> List[Dict[str, Any]]:
        area_x, area_y = self.config.get_area_of_interest_size()
        n = len(series)
        if n == 0:
            return []

        min_v, max_v = float(np.min(series)), float(np.max(series))
        if math.isclose(min_v, max_v):
            norm_vals = np.full_like(series, 0.5, dtype=float)
        else:
            norm_vals = (series - min_v) / (max_v - min_v)

        xs = np.linspace(0, area_x, n, endpoint=False) + (area_x / (2 * n))
        bboxes_out: List[Dict[str, Any]] = []

        for start in range(0, n, self.window_size):
            end = min(start + self.window_size, n)
            xs_window = xs[start:end]
            ys_window = norm_vals[start:end]
            x_min_px = float(xs_window.min() - (area_x / (2 * n)))
            x_max_px = float(xs_window.max() + (area_x / (2 * n)))
            y_pixels = (1.0 - ys_window) * area_y
            y_min_px = float(y_pixels.min())
            y_max_px = float(y_pixels.max())

            pad_v = max(1.0, 0.02 * area_y)
            y_min_px = max(0.0, y_min_px - pad_v)
            y_max_px = min(area_y - 1.0, y_max_px + pad_v)
            x_min_px = max(0.0, x_min_px)
            x_max_px = min(area_x - 1.0, x_max_px)

            bbox_centered = _to_centered_bbox_from_tl(x_min_px, y_min_px, x_max_px, y_max_px, area_x, area_y)
            bboxes_out.append({
                "start_idx": int(start),
                "end_idx": int(end - 1),
                "bbox": bbox_centered
            })

        return bboxes_out

    def draw_stimulus(self, window: visual.Window, sample: Dict[str, Any]) -> Dict[str, Any]:
        series = np.asarray(sample["data"], dtype=float)
        area_x, area_y = self.config.get_area_of_interest_size()
        n = len(series)
        if n == 0:
            return {"timeseries_bboxes": []}

        min_v, max_v = float(series.min()), float(series.max())
        if math.isclose(min_v, max_v):
            norm_vals = np.full(n, 0.5, dtype=float)
        else:
            norm_vals = (series - min_v) / (max_v - min_v)

        xs = []
        ys = []
        for i, v in enumerate(norm_vals):
            x_px = (i + 0.5) * (area_x / n)
            x_centered = x_px - (area_x / 2.0)
            y_px = (1.0 - v) * area_y
            y_centered = (area_y / 2.0) - y_px
            xs.append(x_centered)
            ys.append(y_centered)

        verts = [(float(x), float(y)) for x, y in zip(xs, ys)]
        line = visual.ShapeStim(win=window, vertices=verts, closeShape=False, lineWidth=2.0, lineColor='black', fillColor=None)
        line.draw()
        window.flip()

        bboxes = self._compute_timeseries_bboxes_from_series(series)
        return {"timeseries_bboxes": bboxes}
