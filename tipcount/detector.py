"""Real-time pipette tip counter for Opentrons tip racks seen by a fixed deck camera.

Pure OpenCV + NumPy, no neural network, a few milliseconds per 640x360 frame.

What the camera sees (measured on the frames in this folder, OpenCV HSV, H in 0..180):
    yellow 200uL rack   frame/lattice  H~26  S 190-255  V 230-255
                        empty well     H~24  S 230-255  V 190-220   (saturated rim, pale-lit bottom up close)
                        tip collar     H~22  S  60-130  V 170-230   (pale cream)
    blue 1000uL rack    frame/lattice  H~99  S 200-255  V 160-200
                        empty well     H~98  S 220-255  V  60-140
                        tip collar     H~100 S  10-130  V 120-230   (pale blue, dark bore in the middle)

The discriminating feature is *saturation*: a tip collar is pale, the frame and the well rims
are saturated.

Pipeline per frame
------------------
1. Hue-gated loose mask -> rack blobs (frame + walls + tips) = search ROIs.
2. Inside each ROI classify pixels: pale (tip collar) and saturated-dim small blobs (empty
   wells).  Their union, closed across the thin lattice and cut along the bright frame
   border, proposes the 96-well face.  The proposal is only approximate.
3. Snap the proposal to the lattice: the 4 corners are refined by coordinate descent on a
   score that rewards *cell boundaries being more saturated than cell centres* in the
   rectified 12x8 grid.  This holds for empty wells (lattice vs rim/bottom), yellow tips
   (lattice vs collar) and blue tips (seams vs collar), and it flips sign when the grid is
   off by half a cell, so the maximum is sharp.  A bare rack wall has no lattice and scores
   low, which rejects it.
4. Each cell is a tip if its centre window is mostly pale and shows no well rim.
5. Skin-tone blobs next to a rack flag it as occluded so the caller can hold the last good
   count.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

COLS, ROWS = 12, 8
CELL = 8
WARP_W, WARP_H = COLS * CELL, ROWS * CELL
DST = np.array([[0, 0], [WARP_W, 0], [WARP_W, WARP_H], [0, WARP_H]], np.float32)

RACK_COLOURS = {
    "yellow": dict(h=(14, 40), pale_s=150, pale_v=140, hole_s=190, hole_v_ratio=0.90, hole_v_min=120),
    "blue":   dict(h=(86, 112), pale_s=140, pale_v=110, hole_s=190, hole_v_ratio=0.80, hole_v_min=40),
}
SKIN = dict(h=(0, 17), s=(60, 170), v=(110, 255))
SKIN_MIN_BLOB = 1200

MIN_BLOB_AREA = 1200
MIN_FACE_AREA = 500
MIN_LATTICE_SCORE = 18.0      # saturation units; walls and deck score < 10, faces 30-90
K3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
K5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
K7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

_yy, _xx = np.mgrid[0:WARP_H, 0:WARP_W]
BOUNDARY = ((_xx % CELL) == 0) | ((_yy % CELL) == 0)
CENTRE = np.isin(_xx % CELL, (3, 4)) & np.isin(_yy % CELL, (3, 4))

DEBUG = False


@dataclass
class RackResult:
    colour: str
    corners: np.ndarray          # 4x2 float32 image coords, cyclic from the top-left, long edge first
    count: int                   # tips present
    grid: np.ndarray             # 8x12 bool, True = tip
    cell_score: np.ndarray       # 8x12 pale fraction minus rim fraction
    occluded: bool
    truncated: bool              # face touches the image edge; count is a lower bound
    lattice_score: float         # alignment quality (higher is better)
    confidence: float            # 0..1 margin between weakest tip cell and strongest empty cell


@dataclass
class FrameResult:
    racks: list[RackResult] = field(default_factory=list)
    ms: float = 0.0


def _inrange(hsv, h, s, v):
    return cv2.inRange(hsv, (h[0], s[0], v[0]), (h[1], s[1], v[1]))


def _order(pts):
    """Cyclic (clockwise on screen) order starting at the top-most, then left-most corner."""
    pts = pts.reshape(-1, 2).astype(np.float32)
    c = pts.mean(0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]
    start = np.lexsort((pts[:, 0], pts[:, 1]))[0]
    return np.roll(pts, -start, axis=0)


def _quad(contour):
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)
    for eps in (0.02, 0.035, 0.05, 0.07, 0.09):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            return _order(approx)
    return _order(cv2.boxPoints(cv2.minAreaRect(hull)))


def _landscape(q):
    top = np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])
    side = np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])
    return q if top >= side else np.array([q[3], q[0], q[1], q[2]], np.float32)


def lattice_score(sat_img: np.ndarray, q: np.ndarray) -> float:
    M = cv2.getPerspectiveTransform(q, DST)
    w = cv2.warpPerspective(sat_img, M, (WARP_W, WARP_H), flags=cv2.INTER_LINEAR)
    return float(w[BOUNDARY].mean() - w[CENTRE].mean())


def refine_quad(sat_img: np.ndarray, q: np.ndarray, steps=(4, 2, 1), max_pass=6) -> tuple[np.ndarray, float]:
    """Coordinate descent on the 8 corner coordinates to maximise the lattice score."""
    q = q.copy()
    best = lattice_score(sat_img, q)
    for step in steps:
        for _ in range(max_pass):
            improved = False
            for i in range(4):
                for j in range(2):
                    for d in (step, -step):
                        q2 = q.copy(); q2[i, j] += d
                        sc = lattice_score(sat_img, q2)
                        if sc > best + 1e-3:
                            best, q, improved = sc, q2, True
            if not improved:
                break
    return q, best


def detect(bgr: np.ndarray) -> FrameResult:
    t0 = time.perf_counter()
    H, W = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    S_full = hsv[:, :, 1]
    out = FrameResult()

    skin = cv2.morphologyEx(_inrange(hsv, SKIN["h"], SKIN["s"], SKIN["v"]), cv2.MORPH_OPEN, K3)
    ns, _, sst, _ = cv2.connectedComponentsWithStats(skin, connectivity=8)
    skin_boxes = [sst[i] for i in range(1, ns) if sst[i, cv2.CC_STAT_AREA] >= SKIN_MIN_BLOB]

    for colour, spec in RACK_COLOURS.items():
        loose = cv2.morphologyEx(_inrange(hsv, spec["h"], (60, 255), (60, 255)), cv2.MORPH_CLOSE, K7)
        nb, blab, bst, _ = cv2.connectedComponentsWithStats(loose, connectivity=8)
        for b in range(1, nb):
            if bst[b, cv2.CC_STAT_AREA] < MIN_BLOB_AREA:
                continue
            bx, by, bw, bh = (int(bst[b, k]) for k in (0, 1, 2, 3))
            pad = 6
            x0, y0 = max(0, bx - pad), max(0, by - pad)
            x1, y1 = min(W, bx + bw + pad), min(H, by + bh + pad)
            roi = hsv[y0:y1, x0:x1]
            s = roi[:, :, 1]; v = roi[:, :, 2]
            hue_ok = cv2.inRange(roi[:, :, 0], spec["h"][0], spec["h"][1]) > 0
            blob = blab[y0:y1, x0:x1] == b
            hull_m = np.zeros(blob.shape, np.uint8)
            cs, _ = cv2.findContours(blob.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.fillConvexPoly(hull_m, cv2.convexHull(max(cs, key=cv2.contourArea)), 255)
            inside = cv2.dilate(hull_m, K3) > 0

            sat = inside & hue_ok & (s >= spec["hole_s"])
            if sat.sum() < 50:
                continue
            lattice_v = float(np.percentile(v[sat], 95))
            pale = inside & hue_ok & (s >= 45) & (s < spec["pale_s"]) & (v > spec["pale_v"])
            hole_px = sat & (v < spec["hole_v_ratio"] * lattice_v) & (v >= spec["hole_v_min"])

            px_per_mm = float(np.clip(bw / 128.0, 0.7, 1.3))
            hole_area = np.pi / 4 * (7.0 * px_per_mm) ** 2
            nh, hlab, hst, _ = cv2.connectedComponentsWithStats(hole_px.astype(np.uint8), connectivity=4)
            keep = np.zeros(nh, bool)
            for j in range(1, nh):
                a = hst[j, cv2.CC_STAT_AREA]
                keep[j] = 0.12 * hole_area <= a <= 3.0 * hole_area
            holes = keep[hlab]

            content = ((pale | holes).astype(np.uint8)) * 255
            content = cv2.morphologyEx(content, cv2.MORPH_CLOSE, K5)
            frame_bright = (sat & (v >= 0.85 * lattice_v)).astype(np.uint8) * 255
            frame_bright = cv2.erode(cv2.morphologyEx(frame_bright, cv2.MORPH_CLOSE, K3), K3)
            content = cv2.bitwise_and(content, cv2.bitwise_not(frame_bright))
            content = cv2.morphologyEx(content, cv2.MORPH_OPEN, K3)

            nc, clab, cst, _ = cv2.connectedComponentsWithStats(content, connectivity=8)
            for c in range(1, nc):
                area = int(cst[c, cv2.CC_STAT_AREA])
                if area < MIN_FACE_AREA:
                    continue
                comp = (clab == c).astype(np.uint8)
                ccs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                q = _landscape(_quad(max(ccs, key=cv2.contourArea)))
                long_e = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
                short_e = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
                if short_e < 10 or not (1.05 <= long_e / short_e <= 3.5):
                    if DEBUG: print(f"  reject {colour} area={area} aspect={long_e/max(short_e,1e-3):.2f} short={short_e:.1f}")
                    continue
                qa = cv2.contourArea(q)
                if qa <= 0 or area / qa < 0.55:
                    if DEBUG: print(f"  reject {colour} area={area} fill={area/max(qa,1):.2f}")
                    continue

                q_img = q + np.array([x0, y0], np.float32)
                q_img, lsc = refine_quad(S_full, q_img)
                if lsc < MIN_LATTICE_SCORE:
                    if DEBUG: print(f"  reject {colour} area={area} lattice={lsc:.1f}")
                    continue

                M = cv2.getPerspectiveTransform(q_img, DST)
                roi_full = np.zeros((H, W), np.uint8)
                pale_full = np.zeros((H, W), np.uint8); pale_full[y0:y1, x0:x1] = pale
                hole_full = np.zeros((H, W), np.uint8); hole_full[y0:y1, x0:x1] = hole_px
                pale_w = cv2.warpPerspective(pale_full, M, (WARP_W, WARP_H), flags=cv2.INTER_NEAREST)
                hole_w = cv2.warpPerspective(hole_full, M, (WARP_W, WARP_H), flags=cv2.INTER_NEAREST)
                pf = pale_w.reshape(ROWS, CELL, COLS, CELL)[:, 1:7, :, 1:7].mean(axis=(1, 3))
                hf = hole_w.reshape(ROWS, CELL, COLS, CELL)[:, 1:7, :, 1:7].mean(axis=(1, 3))
                cells = pf - hf
                grid = cells >= 0.45
                tips, empt = cells[grid], cells[~grid]
                conf = float(np.clip((tips.min() - empt.max()) / 0.4, 0, 1)) if len(tips) and len(empt) else 1.0

                truncated = bool(q_img[:, 0].min() <= 1 or q_img[:, 1].min() <= 1 or
                                 q_img[:, 0].max() >= W - 2 or q_img[:, 1].max() >= H - 2)
                qx0, qy0 = q_img.min(0); qx1, qy1 = q_img.max(0)
                occluded = any(sx < qx1 + 15 and sx + sw > qx0 - 15 and sy < qy1 + 15 and sy + sh > qy0 - 15
                               for sx, sy, sw, sh, _a in skin_boxes)
                out.racks.append(RackResult(colour, q_img, int(grid.sum()), grid, cells, occluded,
                                            truncated, lsc, conf))
    out.racks = _dedupe(out.racks)
    out.ms = (time.perf_counter() - t0) * 1000
    return out


def _dedupe(racks: list[RackResult]) -> list[RackResult]:
    """Two proposals that snapped to the same face: keep the better-aligned one."""
    kept: list[RackResult] = []
    for r in sorted(racks, key=lambda r: -r.lattice_score):
        c = r.corners.mean(0)
        if all(np.linalg.norm(c - k.corners.mean(0)) > 20 for k in kept):
            kept.append(r)
    return kept


def draw(bgr: np.ndarray, res: FrameResult) -> np.ndarray:
    vis = bgr.copy()
    for r in res.racks:
        col = (0, 0, 255) if r.occluded else (0, 255, 0)
        cv2.polylines(vis, [r.corners.astype(np.int32)], True, col, 1)
        x, y = r.corners[0].astype(int)
        cv2.putText(vis, f"{r.colour[0].upper()} {r.count}{'+' if r.truncated else ''}",
                    (int(x), max(10, int(y) - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
    cv2.putText(vis, f"{res.ms:.1f} ms", (5, bgr.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def grid_image(r: RackResult, scale: int = 10) -> np.ndarray:
    img = np.zeros((ROWS * scale, COLS * scale, 3), np.uint8)
    for rr in range(ROWS):
        for cc in range(COLS):
            v = int(np.clip(r.cell_score[rr, cc] * 255, 0, 255))
            colour = (0, 200, 0) if r.grid[rr, cc] else (0, 0, 180)
            cv2.rectangle(img, (cc * scale, rr * scale), ((cc + 1) * scale - 1, (rr + 1) * scale - 1), colour, -1)
            cv2.rectangle(img, (cc * scale + 3, rr * scale + 3), ((cc + 1) * scale - 4, (rr + 1) * scale - 4), (v, v, v), -1)
    return img
