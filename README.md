# Real-time pipette-tip counter (Opentrons Flex deck camera)

Goal: count the tips left in a 96-tip rack from the fixed deck camera in under 500 ms, accurately,
with the least compute. Built 2026-10-24 from the 28 frames in `frames/` (640x360).

## Result in one table

| Config | Latency per frame (M4 Pro, 1 core) | Scored readings within ±2 tips | Exact |
|---|---|---|---|
| slot-locked, per-frame tracking (default) | median 22 ms, p95 34 ms | 37 / 51 | 16 |
| slot-locked, no tracking | median 3 ms, p95 6 ms | 24 / 51 | 11 |

Latency is 20x inside the 500 ms budget either way. The gap between the two rows is what the
±3 px per-frame alignment buys when racks are hand-placed.

Per rack, default config (ground truth in brackets):

| Rack and position | Frames | Readings within ±2 | Notes |
|---|---|---|---|
| blue 1000 µL, far right, full [96] | 9 | 9 / 9 | reads 96 in every frame |
| blue 1000 µL, near left, full [96] | 12 | 11 / 12 | one −4 under a hand shadow |
| yellow 200 µL "IN USE", near middle [64] | 6 | 4 / 6 | two −6 frames right after a hand nudged the rack |
| yellow 200 µL "IN USE", far left [64] | 11 | 9 / 11 | one −3, one +6 |
| blue 1000 µL, near middle, 2 pulled [94] | 2 | 2 / 2 | the two empty wells are seen directly |
| blue 1000 µL, back right, back row cut off by image edge [94] | 2 | 2 / 2 | reported as a range, 7 cells out of view |
| yellow "IN USE" rotated so its empty columns face away [64] | 9 | 0 / 9 | reads 81–87: the empty wells are hidden behind tips. Not solvable from this camera angle. |

The last row is the important caveat: at this oblique viewpoint, an empty well behind a row of
tips is invisible. Camera placement decides accuracy before any algorithm does.

## How it works (no neural network, pure OpenCV)

1. **Slot-locked geometry.** The camera is fixed and racks sit in deck slots, so the 96-well face
   of a rack in a given slot is always the same quadrilateral in the image. `tipcount/slots.json`
   holds one calibrated quad per slot. Per frame the work for a slot is one 96x64 perspective warp
   plus a few array reductions.
2. **Pixel classes measured on this camera.** The discriminating feature is saturation, not
   brightness: a tip collar is pale (yellow rack S≈100, blue rack S≈115), while the frame, the
   lattice and an empty well's rim are deeply saturated (S≥190). Thresholds are relative to each
   rack's own frame so auto-exposure and shadows move everything together.
3. **Per-cell decision.** The centre of each cell is scored for "pale" and the ring around it for
   "empty-well rim". A 200 µL tip is a clean pale disc with no rim; an empty 1000 µL well is a clean
   dark disc. So yellow racks are counted by tips seen and blue racks by 96 minus empty wells seen,
   and the other count is kept as a cross-check.
4. **Per-frame tracking.** A ±3 px translation search of the calibrated quad picks the most
   *decisive* grid (cells clearly pale or clearly not, no frame inside cells). Reach is kept under
   half a well pitch on purpose; a wider search can lock one well over.
5. **Health signals.** Skin-tone blobs next to a rack flag it occluded (the twin holds the last good
   count); cells whose centre falls outside the picture are reported as unknown and the count as a
   range; cells that look like frame flag a misaligned grid.

What did not work at 640x360, for the record: free-form rack detection with grid fitting from the
rack outline, from saturation contrast on cell boundaries, from gradient projections, and from
collar-centre point sets. All of them were off by a fraction of a well, which is a whole tip column
in the count. The frames are simply too small (a well is 6–9 px) for outline-driven alignment.

## Run it

```bash
cd ~/Documents/quick-sketches/500ms-count-fable
.venv/bin/python tipcount/bench.py                       # accuracy + latency on the 28 frames
.venv/bin/python -m tipcount.live --source frames             # replay the frames, JSON lines per slot
.venv/bin/python -m tipcount.live --source 0 --show      # webcam
.venv/bin/python -m tipcount.live --source http://ROBOT/stream.mjpg --out readings.jsonl
.venv/bin/python -m tipcount.live --source http://ROBOT:31950/camera/picture --snapshot
```

Calibrating a new slot (one-time, with any rack sitting in it): put a rough quad for the slot in
`tipcount/slots.json` (four image points: front-left, back-left, back-right, front-right), then

```bash
.venv/bin/python -m tipcount.calibrate frames/FRAME.jpg SLOT_NAME yellow --write
```

The optimizer fits the quad so every cell centre lands on rack content with no frame inside cells.
`scratch/calib_view.py` renders a 6x crop with image-coordinate gridlines to pick the rough quad.

Files: `tipcount/monitor.py` (the counter), `tipcount/calibrate.py` (slot calibration),
`tipcount/live.py` (camera / stream / folder entry point), `tipcount/bench.py` (evaluation),
`tipcount/ground_truth.json` (hand labels), `results_overlay_sheet.jpg` (every frame with
overlays), `tipcount/detector.py` (the free-form detector, kept as a proposal generator).

## What would move accuracy to ±1

- **Higher resolution.** At 1280x720 a well is 12–18 px and the lattice is resolvable; the far
  racks here are at the limit (two pulled tips in a far rack read as a full rack).
- **Camera placement.** Point the camera so the rack's long axis faces it, or mount it closer to
  overhead, so empty wells are never behind tips.
- **Gripper placement.** Robot-placed racks are repeatable to under a pixel; the two −6 frames here
  are a hand nudge the ±3 px tracker could not absorb.
- **Known labware map.** The Flex run API reports which rack is in which slot; feeding that in
  removes the colour guess and the overlap arbitration.
