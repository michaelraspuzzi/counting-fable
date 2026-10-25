"""Slot-locked real-time tip counter.

The deck camera is fixed and tip racks sit in deck slots, so the 96-well face of a rack in a
given slot is always the same quadrilateral in the image.  Calibrate that quad once per slot
(``slots.json``), then per frame the work is one small perspective warp per slot plus a few
array reductions: well under a millisecond per slot.

Per-cell classification (measured on this camera, OpenCV HSV):
    pale  = rack hue, 45 <= S < pale_s, V > pale_v            -> tip collar
    rim   = rack hue, S >= hole_s, V between hole_v_min and hole_v_ratio * lattice brightness
                                                              -> empty-well rim / shadowed lattice
    frame = rack hue, S >= hole_s, bright                     -> frame / lattice
A cell is a tip when its centre window is mostly pale and shows no rim.  Two counts are
produced: tips seen directly, and 96 minus empty wells seen directly; they agree when the
grid is aligned, and the gap is reported as a health signal.
"""
from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

COLS, ROWS, CELL = 12, 8, 8
WARP_W, WARP_H = COLS * CELL, ROWS * CELL
DST = np.array([[0, 0], [WARP_W, 0], [WARP_W, WARP_H], [0, WARP_H]], np.float32)
WIN = slice(1, 7)   # central 6x6 of each 8x8 cell

# Thresholds are relative to the rack's own frame (lattice brightness and frame saturation) so
# that auto-exposure and shadows move everything together.  Absolute equivalents at the
# calibration exposure: yellow pale_s~150 pale_v~140, blue pale_s~140 pale_v~110.
RACK_COLOURS = {
    "yellow": dict(h=(14, 40), pale_s_rel=0.68, pale_v_rel=0.57, hole_s_rel=0.86, hole_v_ratio=0.90, hole_v_min_rel=0.49, tip_pf=0.5),
    "blue":   dict(h=(86, 112), pale_s_rel=0.63, pale_v_rel=0.50, hole_s_rel=0.86, hole_v_ratio=0.80, hole_v_min_rel=0.18, tip_pf=0.3),
}
SKIN = dict(h=(0, 17), s=(60, 170), v=(110, 255))
SKIN_MIN_BLOB = 1200


@dataclass
class SlotReading:
    slot: str
    present: bool
    colour: str | None = None
    count: int | None = None            # best estimate: tips seen (yellow) / 96 - holes seen (blue)
    count_by_holes: int | None = None   # the other estimator, as a cross-check
    grid: np.ndarray | None = None      # 8x12 bool
    occluded: bool = False
    coverage: float = 0.0               # fraction of cells that look like rack at all
    misaligned: bool = False            # frame-dominated cells inside the grid
    unknown: int = 0                    # cells whose centre falls outside the image; `count` assumes
                                        # they are intact, so the true count is in [count - unknown, count]
    quality: float = 0.0                # coverage minus frame fraction; used to arbitrate overlaps


@dataclass
class FrameReading:
    slots: list[SlotReading] = field(default_factory=list)
    ms: float = 0.0


def _cells(mask: np.ndarray) -> np.ndarray:
    return mask.reshape(ROWS, CELL, COLS, CELL)[:, WIN, :, WIN].mean(axis=(1, 3))


def pixel_classes(warp_hsv: np.ndarray, colour: str):
    """pale (tip collar), rim (empty-well rim), frame (bright lattice/border) masks of a warp."""
    spec = RACK_COLOURS[colour]
    h, s, v = warp_hsv[:, :, 0], warp_hsv[:, :, 1], warp_hsv[:, :, 2]
    hue_ok = (h >= spec["h"][0]) & (h <= spec["h"][1])
    # Near-black pixels get arbitrary hue and saturation in HSV, so everything below is gated on
    # brightness as well, and a "rack" whose frame is not bright is no rack.
    strong = hue_ok & (s >= 190) & (v >= 80)
    frame_s = float(np.median(s[strong])) if strong.sum() > 20 else 225.0
    sat = hue_ok & (s >= spec["hole_s_rel"] * frame_s) & (v >= 25)
    lattice_v = float(np.percentile(v[sat], 95)) if sat.sum() > 20 else 255.0
    if lattice_v < 110:
        z = np.zeros(h.shape, bool)
        return z, z, z
    pale = hue_ok & (s >= 45) & (s < spec["pale_s_rel"] * frame_s) & (v > spec["pale_v_rel"] * lattice_v)
    rim = sat & (v < spec["hole_v_ratio"] * lattice_v) & (v >= spec["hole_v_min_rel"] * lattice_v)
    frame = sat & (v >= spec["hole_v_ratio"] * lattice_v)
    return pale, rim, frame


# Expanded warp with a one-cell margin, used to score how well a quad fits the rack face.
MARGIN = 1
EW, EH = (COLS + 2 * MARGIN) * CELL, (ROWS + 2 * MARGIN) * CELL
M0 = MARGIN * CELL
EDST = np.array([[M0, M0], [M0 + COLS * CELL, M0], [M0 + COLS * CELL, M0 + ROWS * CELL], [M0, M0 + ROWS * CELL]], np.float32)
_yy, _xx = np.mgrid[0:EH, 0:EW]
_RING = ~((_xx >= M0) & (_xx < EW - M0) & (_yy >= M0) & (_yy < EH - M0)) & (_yy >= M0)   # not the back margin


def fit_score(hsv: np.ndarray, q: np.ndarray, colour: str) -> float:
    """Higher is better: cell centres on rack content, no frame inside cells, no content in the
    margin just outside the grid, grid inside the picture."""
    H, W = hsv.shape[:2]
    out = np.clip(-q[:, 0], 0, None).sum() + np.clip(q[:, 0] - (W - 1), 0, None).sum() + \
          np.clip(-q[:, 1], 0, None).sum() + np.clip(q[:, 1] - (H - 1), 0, None).sum()
    M = cv2.getPerspectiveTransform(q, EDST)
    warp = cv2.warpPerspective(hsv, M, (EW, EH), flags=cv2.INTER_LINEAR)
    pale, rim, frame = pixel_classes(warp, colour)
    content = pale | rim
    inner = content[M0:EH - M0, M0:EW - M0].reshape(ROWS, CELL, COLS, CELL)
    fr = frame[M0:EH - M0, M0:EW - M0].reshape(ROWS, CELL, COLS, CELL)
    centre = inner[:, 2:6, :, 2:6].mean()
    frame_in = fr[:, 1:7, :, 1:7].mean()
    spill = content[_RING].mean()
    return float(centre - 1.5 * frame_in - 1.0 * spill - 0.05 * out)


class SlotMonitor:
    def __init__(self, slots_path: str | Path, history: int = 3, track: bool = True, track_px: int = 3):
        self.track, self.track_px = track, track_px   # keep < half a cell or the search can jump a well
        cfg = json.loads(Path(slots_path).read_text())
        self.slots = {}
        for name, s in cfg["slots"].items():
            q = np.array(s["quad"], np.float32)
            M = cv2.getPerspectiveTransform(q, DST)
            cx, cy = np.meshgrid((np.arange(COLS) + 0.5) * CELL, (np.arange(ROWS) + 0.5) * CELL)
            centres = np.stack([cx, cy], -1).reshape(1, -1, 2).astype(np.float32)
            img_pts = cv2.perspectiveTransform(centres, np.linalg.inv(M)).reshape(ROWS, COLS, 2)
            W, H = cfg.get("image_size", [640, 360])
            valid = (img_pts[..., 0] >= 2) & (img_pts[..., 0] <= W - 3) & (img_pts[..., 1] >= 2) & (img_pts[..., 1] <= H - 3)
            self.slots[name] = dict(quad=q, M=M, bbox=(q.min(0) - 12, q.max(0) + 12), valid=valid, centres=img_pts,
                                    partial=bool(s.get("partial", False)))
        self.history = {name: deque(maxlen=history) for name in self.slots}
        self.track_state = {name: dict(offset=np.zeros(2, np.float32), score=None) for name in self.slots}

    def _classify(self, warp_hsv: np.ndarray, colour: str):
        pale, rim, frame = pixel_classes(warp_hsv, colour)
        P = pale.reshape(ROWS, CELL, COLS, CELL); R = rim.reshape(ROWS, CELL, COLS, CELL)
        pf = P[:, 2:6, :, 2:6].mean(axis=(1, 3))                      # pale in the 4x4 centre
        ring = (R.sum(axis=(1, 3)) - R[:, 2:6, :, 2:6].sum(axis=(1, 3))) / (CELL * CELL - 16)
        rf = _cells(rim); ff = _cells(frame)
        return pf, rf, ff, ring

    def read(self, bgr: np.ndarray) -> FrameReading:
        t0 = time.perf_counter()
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        skin = cv2.inRange(hsv, (SKIN["h"][0], SKIN["s"][0], SKIN["v"][0]), (SKIN["h"][1], SKIN["s"][1], SKIN["v"][1]))
        skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        ns, _, sst, _ = cv2.connectedComponentsWithStats(skin, connectivity=8)
        skin_boxes = [sst[i] for i in range(1, ns) if sst[i, cv2.CC_STAT_AREA] >= SKIN_MIN_BLOB]

        out = FrameReading()
        for name, sl in self.slots.items():
            warp = cv2.warpPerspective(hsv, sl["M"], (WARP_W, WARP_H), flags=cv2.INTER_LINEAR)
            best = None
            for colour in RACK_COLOURS:
                pf, rf, ff, ring = self._classify(warp, colour)
                coverage = float(((pf + rf + ff) >= 0.5).mean())
                if best is None or coverage > best[0]:
                    best = (coverage, colour, pf, rf, ff, ring)
            coverage, colour, pf, rf, ff, ring = best
            (bx0, by0), (bx1, by1) = sl["bbox"]
            occluded = any(sx < bx1 and sx + sw > bx0 and sy < by1 and sy + sh > by0
                           for sx, sy, sw, sh, _a in skin_boxes)
            if coverage < 0.75:
                out.slots.append(SlotReading(name, False, None, None, None, None, occluded, coverage))
                self.history[name].clear()
                continue
            if self.track:
                # Hand or gripper placement varies by a few px between runs.  Search a small
                # translation of the calibrated quad for the most *decisive* grid: cell centres
                # that are clearly pale or clearly not, with no frame inside cells.  Stateless,
                # ~17 small warps per visible rack.
                q0 = sl["quad"]
                def decisive(dx, dy):
                    Mq = cv2.getPerspectiveTransform(q0 + np.array([dx, dy], np.float32), DST)
                    w = cv2.warpPerspective(hsv, Mq, (WARP_W, WARP_H), flags=cv2.INTER_LINEAR)
                    pf_, rf_, ff_, ring_ = self._classify(w, colour)
                    return float(np.abs(pf_ - 0.5).mean() - ff_.mean()), (pf_, rf_, ff_, ring_)
                best_sc, best_d, best_maps = decisive(0, 0)[0], (0, 0), None
                for step, centre in ((2, (0, 0)), (1, None)):
                    c0 = best_d if centre is None else centre
                    for dx in (-step, 0, step):
                        for dy in (-step, 0, step):
                            d = (c0[0] + dx, c0[1] + dy)
                            if d == (0, 0) and best_maps is None and step == 2:
                                continue
                            if max(abs(d[0]), abs(d[1])) > self.track_px:
                                continue
                            sc, maps = decisive(*d)
                            if sc > best_sc + 1e-6:
                                best_sc, best_d, best_maps = sc, d, maps
                if best_maps is not None:
                    pf, rf, ff, ring = best_maps
                    coverage = float(((pf + rf + ff) >= 0.5).mean())
                self.track_state[name]["offset"] = np.array(best_d, np.float32)
            # A real well face: nearly every cell is a collar, a well disc or (at the edges) frame,
            # and cells are decisive.  Rack walls and deck seen through a vacant slot are not.
            face = float(((pf >= 0.3) | ((rf >= 0.25) & (ff < 0.5)) | (ff >= 0.4)).mean())
            decisive = float(np.abs(pf - 0.5).mean())
            if face < 0.80 or decisive < 0.25:
                out.slots.append(SlotReading(name, False, None, None, None, None, occluded, coverage))
                self.history[name].clear()
                continue
            spec_tip = RACK_COLOURS[colour]["tip_pf"]
            framey = ff >= 0.4                      # the cell is mostly frame: grid edge misfit, not a well
            if colour == "yellow":
                # an empty 200uL well shows a saturated rim around a (sometimes lit) centre; a tip
                # collar covers the whole cell so its ring is never rim-coloured
                tips = (pf >= 0.5) & (ring <= 0.10)
                empties = ((ring > 0.10) | (pf < 0.3)) & ~framey
            else:
                # an empty 1000uL well is a dark saturated disc in the centre of the cell
                tips = pf >= spec_tip
                empties = (rf >= 0.25) & (pf < spec_tip) & ~framey
            unknown = ~sl["valid"]
            tips &= ~unknown; empties &= ~unknown
            misaligned = bool(framey.sum() > 4)
            if sl["partial"] or unknown.sum() > 12:
                # most of the face is outside the picture: say so instead of guessing
                out.slots.append(SlotReading(name, True, colour, None, None, tips, occluded, coverage,
                                             misaligned, int(unknown.sum()), float(coverage - ff.mean())))
                continue
            misaligned = bool((ff >= 0.5).sum() > 4)
            # Which object is the distinctive one depends on the rack: a 200uL collar is a clean
            # pale disc with no rim (count tips), an empty 1000uL well is a clean dark disc
            # (count holes).  The other count is kept as a cross-check.
            n_unknown = int(unknown.sum())
            n_tips, n_holes = int(tips.sum()) + n_unknown, int(96 - empties.sum())
            count, cross = (n_tips, n_holes) if colour == "yellow" else (n_holes, n_tips)
            r = SlotReading(name, True, colour, count, cross, tips,
                            occluded, coverage, misaligned, int(unknown.sum()), float(coverage - ff.mean()))
            if not occluded:
                self.history[name].append(r.count)
            out.slots.append(r)
        out.slots = self._suppress_overlaps(out.slots)
        out.ms = (time.perf_counter() - t0) * 1000
        return out

    def _suppress_overlaps(self, readings):
        """Hand-placed racks can straddle two calibrated positions; keep the better reading."""
        present = [r for r in readings if r.present]
        drop = set()
        for i, a in enumerate(present):
            for b in present[i + 1:]:
                if self._overlap(a.slot, b.slot) > 0.3:
                    drop.add(b.slot if a.quality >= b.quality else a.slot)
        for r in readings:
            if r.slot in drop:
                r.present = False; r.count = None; r.count_by_holes = None
                self.history[r.slot].clear()
        return readings

    def _overlap(self, a, b):
        pa = self.slots[a]["quad"].astype(np.float32); pb = self.slots[b]["quad"].astype(np.float32)
        inter, _ = cv2.intersectConvexConvex(pa, pb)
        return inter / min(cv2.contourArea(pa), cv2.contourArea(pb))

    def smoothed(self, name: str) -> int | None:
        h = self.history[name]
        return int(np.median(h)) if h else None


def draw(bgr: np.ndarray, mon: SlotMonitor, fr: FrameReading) -> np.ndarray:
    vis = bgr.copy()
    for r in fr.slots:
        q = mon.slots[r.slot]["quad"].astype(np.int32)
        if not r.present:
            cv2.polylines(vis, [q], True, (90, 90, 90), 1)
            continue
        col = (0, 0, 255) if r.occluded else ((0, 200, 255) if r.misaligned else (0, 255, 0))
        cv2.polylines(vis, [q], True, col, 1)
        x, y = q[1]
        cv2.putText(vis, f"{r.slot[0]}:{r.count}", (int(x) - 10, max(10, int(y) - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    cv2.putText(vis, f"{fr.ms:.2f} ms", (5, bgr.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return vis
