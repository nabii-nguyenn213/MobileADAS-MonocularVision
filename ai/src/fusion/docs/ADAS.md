# Straight-road ADAS experiment

## Implementation

New modules:

- `ai/src/fusion/adas_pipeline.py`: `ADASPipeline`, one lane and vehicle call
  per frame, classification, inside-only distance calls, reset and timings.
- `ai/src/fusion/ego_lane_classifier.py`: independent RANSAC boundary geometry
  and conservative inside/outside/unknown decisions.
- `ai/src/fusion/adas_visualizer.py`: points-only lanes, green inside boxes with
  meters or N/A, gray outside/unknown boxes with no distance label.
- `ai/src/fusion/tests/test_adas_pipeline.py`: video runner and JSONL debugging.
- `ai/src/fusion/tests/test_adas_pipeline_unit.py`: membership, gating,
  calibration, coordinate and visualization regression tests.
- This document.

No existing pipelines, model architectures, weights, distance formulas or tests
were changed. No dependencies were added. Runtime uses existing PyTorch,
Ultralytics, NumPy, OpenCV and Python standard-library modules.

The existing `TwinLiteNetLaneModel` loads official Medium BDD100K weights at
`ai/checkpoints/lane_detection/twinlitenet/medium.pth`. The existing
`VehicleModel` is explicitly given `ai/checkpoints/vehicle_detection/26n_best.pt`,
not its different default checkpoint. Existing class mapping is preserved.
No old lane network or additional distance/vehicle network is imported.

## What roi_640_0.10_robust actually means

Inspection of `distance_estimation/model.py`, the image test runner and saved
experiment summary found **no method named roi_640_0.10_robust in the distance
estimator**, and no robust distance aggregation. This is a detector experiment
name: normalized ROI `[0.25, 0.25, 0.75, 0.65]`, YOLO size 640, confidence 0.10,
and robust selection of a single target car for still-image evaluation.

The new `--distance-method roi_640_0.10_robust` selects those saved detector/ROI
settings, loaded from
`dataset/distance/results/_experiments/roi_640_0.10_robust/summary.json`.
ROI detection and original-coordinate mapping reuse the existing `infer()`
helper from `distance_estimation/tests/test_distance_images.py` verbatim. This
helper performs exactly one `VehicleModel.predict()` call. Importing that module
has no CLI side effects. It imports no old lane or fusion network.

The original robust **single-target** selector remains unchanged. It is not run
on video: it intentionally discards other vehicles and selects only cars, whereas
ADAS must classify every detected road vehicle. Distance estimation is still the
exact existing `DistanceEstimator.estimate(bbox, (width,height), class_name)`.
No robust depth formula or aggregation was invented.

The still-image ROI is unsuitable for the provided driving video: at frame 600
it ends at row 936, while lane support starts around rows 963/975. Cropped vehicle
contact points cannot lie inside that observed lane region. For this video the
recorded command explicitly uses `--full-frame-vehicles`, replacing only the
inference crop with the full frame. YOLO still runs once at size 640 and the
preset's 0.10 confidence. Omit this flag to reproduce the exact saved ROI crop.
The benchmark records the actual ROI (`null` for full frame) and confidence;
the full-frame run is not an exact replay of the historical detector experiment.

## Calibration and distance reuse

`DistanceEstimator` is initialized once. It has no detector and no neural network.
Bounding boxes and image dimensions passed to it are original-frame pixels;
YOLO's internal 640-pixel size and lane tensor sizes are never camera geometry.
Its existing `effective_focal_length()` handles reference-resolution scaling.

The user explicitly confirmed that `dataset/tests/lane_detector/test.MP4` and
`dataset/distance/5m.jpeg` use the same camera, lens, zoom and image mode.
Accordingly, the final run explicitly loads the existing saved summary through
`--camera-config` (alias `--calibration-config`):

- focal length: 616.6666666666666 pixels at 1280x961;
- existing vehicle-width assumptions: car 1.8 m, bus/truck 2.5 m,
  motorcycle 0.8 m, pickup truck 2.0 m;
- full-video effective focal length: 925 pixels, calculated by the existing API.

`load_camera_config()` accepts existing constructor JSON or an experiment summary,
extracts only supported constructor keys, and validates them before network loads.
No per-frame calibration or calibration-image detector pass occurs. The current
image runner removed `--calibration-image`; this CLI therefore uses its supported
camera configuration API rather than resurrecting an old calibration workflow.

For arbitrary videos, omit camera configuration when applicability is unknown:
inside vehicles show N/A and the frame displays a calibration-unavailable notice.
An explicit config is the caller's assertion that the camera setup matches.
No focal length or FOV is silently guessed. Nonfinite/nonpositive predictions,
exceptions, unknown vehicle classes and invalid boxes produce N/A. ROI-truncated
boxes are classified unknown. Outside/unknown vehicles never call the estimator.

Distances remain approximate optical-axis depth, not bumper clearance or
Euclidean range. Class widths are assumptions, not per-vehicle measurements;
yaw, occlusion and imperfect boxes can bias estimates even with matching camera
calibration. The historical evaluation itself reported nontrivial errors.

## Membership geometry

Each boundary is robustly refitted as x = a*y+b using normalized coordinates and
48 deterministic RANSAC trials, then least squares on inliers. Require at least
six points, 70% support and 10% image-height vertical span; reject out-of-image,
nonfinite and excessively sloped inputs. Intersect the two support regions,
reject crossing/narrow/overwide pairs, then evaluate at the vehicle's bottom
center `(0.5*(x1+x2), y2)`.

Default evaluation extension is tightly bounded to 1.5% image height beyond
shared support (21.6 pixels in this video); pair geometry is validated at those
extended endpoints too. `EgoLaneClassifier(max_extrapolation=0)` disables it.
No unbounded horizon extrapolation is performed. Lane drawing still uses only
the model's original point lists, independent of fitted evaluation geometry.

The configurable `--lane-margin` is an uncertain band in original-frame pixels:
inside requires bottom center strictly between inward margins; farther outside
is outside; near either boundary is unknown. Missing boundaries, invalid fits,
partial boxes and contact points outside the supported region are unknown.
Bounding-box overlap alone does not imply inside status.

With `--smooth`, lane coefficient smoothing uses the existing implementation.
Membership additionally requires two consecutive inside observations matched by
class and IoU >=0.5 using one-to-one matches. Unknown/outside evidence and missing
tracks clear inside confirmation immediately; no inside label is carried across
a missing-lane frame. This is lightweight confirmation, not a persistent tracker.

## Commands

From the project root in the existing capstone environment:

```bash
python -m unittest ai.src.fusion.tests.test_adas_pipeline_unit ai.src.lane_detection.tests.test_twinlitenet -v
python -m unittest discover -s ai/src/distance_estimation/tests -v

python -m ai.src.fusion.tests.test_adas_pipeline \
  --input dataset/tests/lane_detector/test.MP4 \
  --output ai/outputs/adas/full.mp4 \
  --full-frame-vehicles --smooth \
  --camera-config dataset/distance/results/_experiments/roi_640_0.10_robust/summary.json \
  --save-every 300 \
  --debug-jsonl ai/outputs/adas/full.frames.jsonl
```

The default device auto-selects CUDA when available, otherwise CPU. Use `--device
cpu` to force CPU. Use `--show` for an interactive window (Q quits); `--max-frames`
for a short run; `--warmup` defaults to 10. Other options include lane/vehicle
weights, vehicle confidence, lane threshold, point radius, lane margin, distance
preset and CPU thread count. Original resolution and FPS are preserved. OpenCV
writes video without audio. JSONL records provide every frame's lanes, vehicle
boxes/statuses/distances, distance-call count and stage timings. Sample JPEGs
are saved at the requested interval.

## Measurement and verification

Timing uses explicit CUDA synchronization where applicable, after unmeasured
warmup and reset of smoothing/counters. Lane latency includes preprocessing,
network inference and lane extraction. Vehicle latency includes the existing
predict API's preprocessing, inference, NMS and dictionary conversion. Membership,
distance and drawing are separate. Pipeline latency excludes encoding and input
video decoding; end-to-end includes encoding, input decoding and optional debug
writes/display. Peak RSS includes model loading and warmup, not model-only memory.

Unit checks cover inside/outside/touching/overlapping/partial boxes; missing,
nonfinite and crossing lanes; unsupported contact rows; conservative temporal
confirmation; single lane/vehicle calls; inside-only distance calls; exact ROI
coordinate translation; invalid predictions/exceptions; no automatic calibration;
saved calibration loading; green/gray colors, points and no polygon filling.
The existing lane and distance regression suites are also run. No GPU, mobile
latency or quantitative lane-membership accuracy is claimed.

## Final measured run

Intel Core i5-12450H CPU, four PyTorch threads, CUDA unavailable.
Full source processed: 1,800 frames, 1920x1440 at 30 FPS, 10 warmup passes.
Video metadata and final frame decoding verified after writer release.

- lane_ms: 146.2872 ms per frame
- vehicle_ms: 38.6286 ms per frame
- membership_ms: 0.5683 ms per frame
- distance_ms: 0.0116 ms per frame
- visualization_ms: 1.5179 ms per frame
- pipeline_ms: 187.0135 ms per frame
- encoding_ms: 14.0927 ms per frame
- end_to_end_ms: 203.5412 ms per frame
- Processing FPS: 5.347
- End-to-end FPS: 4.913
- Peak process RSS: 1062.5 MiB
- Vehicle observations: 244 inside, 519 outside, 6,642 unknown.
- Distance calls: 244; valid returned distances: 244.

All 1,800 debug records were audited: calls equal inside observations in
every frame; outside/unknown distance fields are always null; every numeric
prediction exactly matches the existing estimator with the saved calibration.
No forbidden previous lane/fusion modules were loaded. Forty unique tests
passed (19 new ADAS, six existing TwinLiteNet checks, 15 existing distance
checks); the two configuration checks also passed after removing a redundant
startup validation instance. Missing video and lane weights fail clearly.

Artifacts: `ai/outputs/adas/full.mp4`, `full.benchmark.json`,
`full.frames.jsonl`, and `full.frame{0,300,600,900,1200,1500}.jpg`.
Sample frames were inspected for point-only boundaries, green boxes with
meters, gray boxes without distances, and FPS. Interactive display and CUDA
were not tested on this CPU/headless run.

Lane availability and conservative support limits cause many unknown
observations. Counts are per-frame observations, not unique tracked vehicles
or accuracy scores. Video distances and lane membership have no ground-truth
evaluation. Performance is desktop CPU performance, not mobile FPS.
