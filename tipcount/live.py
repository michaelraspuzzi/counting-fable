"""Live tip-count monitor.

Examples
  python -m tipcount.live --source 0                       # webcam index
  python -m tipcount.live --source rtsp://... | http://host/stream.mjpg
  python -m tipcount.live --source http://ROBOT_IP:31950/camera/picture --snapshot   # OT-2 style still endpoint
  python -m tipcount.live --source frames/                 # a folder of JPGs, replayed in name order
  python -m tipcount.live --source clip.mp4 --show --out readings.jsonl

Each frame prints one JSON line per slot:
  {"t": 1730000000.123, "slot": "G_near_mid", "present": true, "colour": "yellow", "count": 64,
   "count_low": 64, "cross_check": 76, "smoothed": 64, "occluded": false, "latency_ms": 18.2}
`smoothed` is the median of the last 3 unoccluded readings and is the value a digital twin
should display; `count` is this frame's raw reading.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .monitor import SlotMonitor, draw


def frames(source: str, snapshot: bool):
    if Path(source).is_dir():
        for f in sorted(glob.glob(str(Path(source) / "[0-9][0-9]-[0-9][0-9]-[0-9][0-9]-[0-9][0-9].jpg")) or glob.glob(str(Path(source) / "*.jpg")), key=lambda f: Path(f).name.split("-", 1)[-1]):
            yield cv2.imread(f)
        return
    if snapshot:
        import urllib.request
        while True:
            data = urllib.request.urlopen(source, timeout=5).read()
            im = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if im is not None:
                yield im
        return
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)          # always read the newest frame
    while True:
        ok, im = cap.read()
        if not ok:
            return
        yield im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--slots", default=str(Path(__file__).with_name("slots.json")))
    ap.add_argument("--snapshot", action="store_true", help="source is a still-image URL to poll")
    ap.add_argument("--show", action="store_true", help="show the overlay window")
    ap.add_argument("--out", help="append JSON lines here")
    ap.add_argument("--no-track", action="store_true")
    args = ap.parse_args()

    mon = SlotMonitor(args.slots, track=not args.no_track)
    sink = open(args.out, "a") if args.out else None
    for im in frames(args.source, args.snapshot):
        t_cap = time.time()
        if im.shape[1] != 640:
            im = cv2.resize(im, (640, int(im.shape[0] * 640 / im.shape[1])))
        fr = mon.read(im)
        for r in fr.slots:
            rec = dict(t=round(t_cap, 3), slot=r.slot, present=r.present, colour=r.colour, count=r.count,
                       count_low=(None if r.count is None else r.count - r.unknown), cross_check=r.count_by_holes,
                       smoothed=mon.smoothed(r.slot), occluded=r.occluded, unknown_cells=r.unknown,
                       latency_ms=round(fr.ms, 2))
            line = json.dumps(rec)
            print(line)
            if sink:
                sink.write(line + "\n")
        if sink:
            sink.flush()
        if args.show:
            cv2.imshow("tipcount", draw(im, mon, fr))
            if cv2.waitKey(1) & 0xFF == 27:
                break


if __name__ == "__main__":
    main()
