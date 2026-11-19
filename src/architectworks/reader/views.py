"""What the reader sees: crops of the flattened sheet, with a coordinate grid
labelled in full-sheet pixels, and optionally the numbered candidate marks.

Every view says which sheet pixels it covers, so the model can answer in
sheet coordinates whatever the zoom.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from architectworks.reader.detect import Detections

MAX_SIDE = 1568  # keeps one view at about 1,600 image tokens or fewer
JPEG_QUALITY = 90  # a 1568 px view is ~250-450 KB; PNG would be 1-2 MB


@dataclass
class View:
    image: bytes  # JPEG
    x0: int
    y0: int
    x1: int
    y1: int
    scale: float  # view pixels per sheet pixel
    marks: list[dict]

    def caption(self) -> str:
        head = f"Sheet pixels x {self.x0}..{self.x1}, y {self.y0}..{self.y1}; zoom {self.scale:.2f}x. Grid labels are sheet pixels."
        if not self.marks:
            return head
        listing = ", ".join(f"{m['id']}@({m['x']:.0f},{m['y']:.0f})" for m in self.marks)
        return f"{head}\nMarks in view: {listing}"


def _grid_step(span_px: float) -> int:
    for step in (10, 20, 25, 50, 100, 200, 250, 500):
        if span_px / step <= 12:
            return step
    return 1000


def render(flat: np.ndarray, x0: float, y0: float, x1: float, y1: float, det: Detections | None = None, max_side: int = MAX_SIDE, max_zoom: float = 4.0) -> View:
    h, w = flat.shape
    x0, y0 = max(0, int(math.floor(x0))), max(0, int(math.floor(y0)))
    x1, y1 = min(w, int(math.ceil(x1))), min(h, int(math.ceil(y1)))
    if x1 - x0 < 8 or y1 - y0 < 8:
        raise ValueError("region is too small")
    crop = flat[y0:y1, x0:x1]
    scale = min(max_zoom, max_side / max(crop.shape))
    img = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    step = _grid_step(max(x1 - x0, y1 - y0))
    font = 0.4 + 0.1 * min(3, img.shape[1] / 800)
    for gx in range((x0 // step + 1) * step, x1, step):
        X = int((gx - x0) * scale)
        cv2.line(img, (X, 0), (X, img.shape[0]), (150, 210, 150), 1)
        cv2.putText(img, str(gx), (X + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, font, (0, 110, 0), 1, cv2.LINE_AA)
    for gy in range((y0 // step + 1) * step, y1, step):
        Y = int((gy - y0) * scale)
        cv2.line(img, (0, Y), (img.shape[1], Y), (150, 210, 150), 1)
        cv2.putText(img, str(gy), (2, Y - 3), cv2.FONT_HERSHEY_SIMPLEX, font, (0, 110, 0), 1, cv2.LINE_AA)
    marks: list[dict] = []
    if det is not None:
        for m in det.in_region(x0, y0, x1, y1):
            X, Y = int((m.x - x0) * scale), int((m.y - y0) * scale)
            color = (0, 0, 220) if m.kind == "convergence" else (200, 60, 0)
            cv2.circle(img, (X, Y), max(4, int(3 * scale)), color, 1, cv2.LINE_AA)
            cv2.putText(img, m.id, (X + 5, Y - 5), cv2.FONT_HERSHEY_SIMPLEX, font, color, 1, cv2.LINE_AA)
            marks.append(m.to_json())
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("could not encode view")
    return View(buf.tobytes(), x0, y0, x1, y1, scale, marks)
