"""Stage 2a: find candidate marks on a sheet without knowing what they are.

Detection proposes; the reader names. Every candidate gets a short id that
the vision model answers in ("C3 is the center point of the north-west
pavilion"), so positions come from image processing and meaning from the
model.

Two detectors, both tuned on the example foundation plan and measured
against its hand-placed calibration points (see tests/test_reader.py):

* **blobs** (``M`` ids): compact darkness about the size of a column mark,
  found with a center-surround filter. Column marks are hexagon outlines with
  a hatched fill, so they are not solid and plain thresholding misses them.
* **convergences** (``C`` ids): points that many line segments, at many
  different angles, pass through. Pavilion centers are drawn as the meeting
  point of radial lines; so are some column marks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from architectworks.sheets import flatten


@dataclass
class Mark:
    id: str
    kind: str  # blob | convergence
    x: float
    y: float
    score: float

    def to_json(self) -> dict:
        return {"id": self.id, "kind": self.kind, "x": round(self.x, 1), "y": round(self.y, 1), "score": round(self.score, 2)}


@dataclass
class Detections:
    width: int
    height: int
    marks: list[Mark] = field(default_factory=list)
    segments: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))

    def by_id(self) -> dict[str, Mark]:
        return {m.id: m for m in self.marks}

    def in_region(self, x0: float, y0: float, x1: float, y1: float) -> list[Mark]:
        return [m for m in self.marks if x0 <= m.x <= x1 and y0 <= m.y <= y1]

    def to_json(self) -> dict:
        return {"width": self.width, "height": self.height, "marks": [m.to_json() for m in self.marks], "segments": int(len(self.segments))}


def load_flat(path: str) -> np.ndarray:
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(path)
    return flatten(gray)


def _peaks(resp: np.ndarray, threshold: float, nms: int) -> list[tuple[int, int, float]]:
    mx = cv2.dilate(resp, np.ones((nms, nms), np.uint8))
    ys, xs = np.nonzero((resp >= mx) & (resp > threshold))
    return sorted(((int(x), int(y), float(resp[y, x])) for x, y in zip(xs, ys)), key=lambda t: -t[2])


def blobs(flat: np.ndarray, inner: int = 17, outer: int = 35, percentile: float = 99.8) -> list[tuple[int, int, float]]:
    dark = 255.0 - flat.astype(np.float32)
    resp = cv2.boxFilter(dark, -1, (inner, inner)) - cv2.boxFilter(dark, -1, (outer, outer))
    return _peaks(resp, float(np.percentile(resp, percentile)), inner)


def segments(flat: np.ndarray, min_length: float = 20.0) -> np.ndarray:
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    found = lsd.detect(flat)[0]
    if found is None:
        return np.zeros((0, 4))
    s = found.reshape(-1, 4).astype(np.float64)
    keep = np.hypot(s[:, 2] - s[:, 0], s[:, 3] - s[:, 1]) > min_length
    return s[keep]


def convergences(flat: np.ndarray, segs: np.ndarray, min_directions: int = 5, reach: float = 200.0, tol: float = 2.5, step: int = 2) -> list[tuple[int, int, float]]:
    """Points crossed by line segments in at least ``min_directions`` distinct
    12-degree direction bins. Candidates are restricted to inked pixels."""
    if len(segs) == 0:
        return []
    L = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    P0 = segs[:, :2]
    U = (segs[:, 2:] - P0) / L[:, None]
    N = np.stack([-U[:, 1], U[:, 0]], 1)
    bins = ((np.degrees(np.arctan2(U[:, 1], U[:, 0])) % 180.0) // 12).astype(int)
    ink = cv2.dilate((flat < 170).astype(np.uint8), np.ones((5, 5), np.uint8))
    ys, xs = np.nonzero(ink[::step, ::step])
    C = np.stack([xs * step, ys * step], 1).astype(np.float64)
    score = np.zeros(len(C), np.float32)
    for i0 in range(0, len(C), 3000):
        c = C[i0 : i0 + 3000]
        rel = c[:, None, :] - P0[None]
        d = np.abs(np.einsum("ijk,jk->ij", rel, N))
        t = np.einsum("ijk,jk->ij", rel, U)
        near = (np.minimum(np.abs(t), np.abs(t - L[None])) < reach) | ((t > 0) & (t < L[None]))
        ok = (d < tol) & near
        hit = np.zeros((len(c), 15), bool)
        for b in range(15):
            hit[:, b] = (ok & (bins[None] == b)).any(1)
        score[i0 : i0 + 3000] = hit.sum(1)
    grid = np.zeros(((flat.shape[0] + step - 1) // step, (flat.shape[1] + step - 1) // step), np.float32)
    grid[(C[:, 1] // step).astype(int), (C[:, 0] // step).astype(int)] = score
    return [(x * step, y * step, s) for x, y, s in _peaks(grid, min_directions - 0.5, 5)]


def detect(flat: np.ndarray) -> Detections:
    h, w = flat.shape
    segs = segments(flat)
    det = Detections(width=w, height=h, segments=segs)

    def ordered(points, prefix, kind):
        # reading order: top to bottom in 40 px bands, then left to right
        pts = sorted(points, key=lambda p: (round(p[1] / 40), p[0]))
        return [Mark(f"{prefix}{i + 1}", kind, float(x), float(y), float(s)) for i, (x, y, s) in enumerate(pts)]

    det.marks = ordered(blobs(flat), "M", "blob") + ordered(convergences(flat, segs), "C", "convergence")
    return det


def nearest(det: Detections, x: float, y: float, kinds: tuple[str, ...] = ("blob", "convergence")) -> tuple[Mark | None, float]:
    best, dist = None, math.inf
    for m in det.marks:
        if m.kind in kinds:
            d = math.hypot(m.x - x, m.y - y)
            if d < dist:
                best, dist = m, d
    return best, dist
