"""Benchmark the slot monitor on the labelled frames: per-frame latency and count accuracy."""
import json, sys, time, glob
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tipcount.monitor import SlotMonitor, draw

here = Path(__file__).resolve().parent
gt = json.loads((here / "ground_truth.json").read_text())["frames"]
import os
mon = SlotMonitor(here / "slots.json", track=os.environ.get("TRACK", "1") == "1")
files = sorted(glob.glob(str(here.parent / "frames" / "[0-9][0-9]-[0-9][0-9]-[0-9][0-9]-[0-9][0-9].jpg")), key=lambda f: Path(f).name.split("-", 1)[1])
out_dir = here.parent / "scratch"; out_dir.mkdir(exist_ok=True)
errs, lat, rows = [], [], []
false_presence = 0; missed = 0
for f in files:
    name = Path(f).name
    im = cv2.imread(f)
    t = time.perf_counter(); fr = mon.read(im); lat.append((time.perf_counter() - t) * 1000)
    cv2.imwrite(str(out_dir / f"slot_{name}"), draw(im, mon, fr))
    labels = gt.get(name, {})
    line = []
    for r in fr.slots:
        want = labels.get(r.slot)
        if r.present and r.count is None:
            line.append(f"{r.slot[0]}=partial({r.unknown} cells out of view)"); continue
        if r.present:
            tag = f"{r.slot[0]}={r.count}/{r.count_by_holes}"
            if r.occluded: tag += "occ"
            if r.misaligned: tag += "!"
            if want is None: false_presence += 1; tag += "(FALSE)"
            elif want == "occluded": tag += "(occ-gt)"
            else:
                e = r.count - want; errs.append(abs(e)); tag += f"(gt{want},{e:+d})"
            line.append(tag)
        else:
            if want not in (None, "occluded"): missed += 1; line.append(f"{r.slot[0]}=MISSED(gt{want})")
    rows.append((name, fr.ms, line))
for name, ms, line in rows:
    print(f"{name} {ms:6.2f}ms  " + "  ".join(line))
errs = np.array(errs); lat = np.array(lat)
print("\nscored readings:", len(errs), " exact:", int((errs == 0).sum()), " within 1:", int((errs <= 1).sum()),
      " within 2:", int((errs <= 2).sum()), " mean abs err: %.2f" % errs.mean(), " max: %d" % errs.max())
print("false presence:", false_presence, " missed racks:", missed)
print("latency ms: median %.2f  p95 %.2f  max %.2f (incl. first-call warmup)" % (np.median(lat), np.percentile(lat, 95), lat.max()))
