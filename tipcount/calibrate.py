"""One-time per-slot calibration of the 96-well face quad.

Usage:  python -m tipcount.calibrate FRAME.jpg SLOT COLOUR [--write]

With a rack sitting in the slot, the quad is fitted by coordinate descent so that:
  * every cell centre lands on rack content (tip collar or empty well),
  * no bright frame lands inside any cell,
  * no content spills into the one-cell margin just outside the grid.
The optimum is sharp and does not depend on how full the rack is.  Per evaluation this is one
small perspective warp, so a full calibration takes ~50 ms.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from .monitor import fit_score as objective_fn


def objective(hsv, q, colour):
    return objective_fn(hsv, q, colour)


def fit(hsv: np.ndarray, q0: np.ndarray, colour: str, steps=(4, 2, 1, 0.5), max_pass=8):
    q = q0.astype(np.float32).copy()
    best = objective(hsv, q, colour)
    for step in steps:
        for _ in range(max_pass):
            improved = False
            for i in range(4):
                for j in range(2):
                    for d in (step, -step):
                        q2 = q.copy(); q2[i, j] += d
                        sc = objective(hsv, q2, colour)
                        if sc > best + 1e-4:
                            best, q, improved = sc, q2, True
            if not improved:
                break
    return q, best


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    write = "--write" in sys.argv
    frame, slot, colour = args[0], args[1], args[2]
    here = Path(__file__).resolve().parent
    cfg = json.loads((here / "slots.json").read_text())
    q0 = np.array(cfg["slots"][slot]["quad"], np.float32)
    hsv = cv2.cvtColor(cv2.imread(frame), cv2.COLOR_BGR2HSV)
    j0 = objective(hsv, q0, colour)
    q, j = fit(hsv, q0, colour)
    print(f"{slot} {colour}: objective {j0:.3f} -> {j:.3f}\n   from {q0.round(1).tolist()}\n   to   {q.round(1).tolist()}")
    if write:
        cfg["slots"][slot]["quad"] = [[round(float(x), 1), round(float(y), 1)] for x, y in q]
        cfg["slots"][slot]["calibrated_from"] = Path(frame).name
        cfg["slots"][slot]["objective"] = round(j, 3)
        (here / "slots.json").write_text(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    main()
