# Persistent vehicle membership and distance filtering

## Root cause and inspected behavior

`VehicleModel.predict()` returns boxes, class and confidence, with no persistent IDs. No other persistent vehicle tracker existed in `ai/src`. `EgoLaneClassifier.classify_all()` classifies each detection independently by default. Its optional smoothing only confirms inside observations against the previous frame using IoU; it immediately accepts outside/unknown observations and drops history on missing evidence. This makes both green status and permission to estimate distance sensitive to a single noisy frame. There was no per-vehicle distance filter. The distance estimator itself is stateless and does not reset: the pipeline simply stops calling it and supplies no label value when membership changes.

The lane scheduler and coordinate handling are already correct: scheduled frames run TwinLiteNet+, skipped frames reuse confidence/age-gated temporal geometry, and the classifier receives smoothed fitted lines. The ROI helper maps YOLO boxes back to original-frame pixels; bottom-center `(x1+x2)/2, y2` is evaluated against lane lines in those same coordinates. Lane tracking resets for explicit resets, scene changes, time discontinuities or resolution changes, not every frame. The legacy classifier allowed at most 1.5% image-height extrapolation; the new observation path uses only overlapping observed support.

The available clip does **not reproduce green/gray flicker** with the current saved ROI: the before run already produces only unknown memberships. New diagnostics explain why: 4,907 observations are ROI-clipped, 377 have missing/invalid boundary pairs, 59 have bottom centers outside the supported y range, 35 coincide with lane-track expiration, and one fails plausible lane width. Those validity gates are preserved. Smoothing must not manufacture inside membership from these detections.

## Implementation

- `vehicle_lane_tracker.py`: one-to-one, class-gated motion/IoU association with active tracks preferred over inactive hypotheses. Close ambiguous matches retire old hypotheses and create fresh IDs. IDs are never recycled within an instance. Tracks expire after the configured timeout. A detection reacquired after any missed frame keeps its ID when safely associated but resets membership and distance evidence. No missing detections are rendered as predicted boxes.
- Per-ID hysteresis confirms inside after three observations and outside after six. At higher frame rates it also requires the corresponding 30 FPS duration; low FPS still requires the configured observation count. Isolated disagreeing/uncertain observations hold color for at most 0.3 seconds from the last supporting observation. Consecutive opposing evidence switches membership. ROI clipping, invalid boxes, expired lane tracks, scene resets and implausible inside associations invalidate immediately.
- `ego_lane_classifier.py`: `observe()` reports normalized position `u`, current geometric evidence, supported-y and width validity, and diagnostic reasons. Default inside band is `[0.10, 0.90]`; outside requires `u < -0.05` or `u > 1.05`. Existing eight-pixel uncertainty is retained as a minimum band. Camera offset is already accounted for by the lane tracker's boundary identity selection. Legacy classification remains available.
- `distance_smoother.py`: per-ID rolling median (three samples), then EMA (alpha 0.35), applied after the existing estimator. Only positive finite values enter history. An extreme ratio greater than four is rejected only when the absolute jump also exceeds a generous time-based motion allowance; repeated consistent discontinuities reacquire instead of freezing forever. This is a numerical gate, not an accuracy or collision-safety guarantee.
- `adas_pipeline.py`: preserves lane inference scheduling and tracking, the exact ROI helper, VehicleModel and checkpoint, calibration and distance formula. New distance calls require both stable inside status and **current** reliable inside evidence. A briefly held green box gets `N/A` rather than an old range when evidence is uncertain. No stale range is displayed. Results expose `raw_distance`, filtered `distance`, `distance_fresh`, `distance_for_alerts`, counters, IDs and rejection reasons. The latter flags do not add an alert system.
- `tests/test_adas_pipeline.py`: CLI parameters and per-frame JSON diagnostics. `--debug-lane-classification` logs lane confidence/age, current observations, normalized position, IDs, stable states and distance rejection reasons. Existing point-only lane, FPS and green/gray rendering remain unchanged.

The installed Ultralytics version is 8.4.174; the environment lacks `scipy` and `lap`, so its ByteTrack path cannot run with the installed dependencies. The lightweight tracker uses existing NumPy only and makes no model calls. It has no appearance embedding; crossings and severe occlusion can still fragment or misassociate identities. Ambiguity deliberately favors losing state over transferring another vehicle's range history.

## Full-video before/after

Both runs use all 1,800 frames of `dataset/tests/lane_detector/test.MP4` (60 seconds, 1920×1440, 30 FPS), CPU, four Torch threads, N=5 and five excluded warmup frames. The saved ROI and camera configuration are unchanged. No applicable calibration was supplied.

| Metric | Before | After |
|---|---:|---:|
| Frames | 1800 | 1800 |
| TwinLiteNet+ calls | 360 | 360 |
| Vehicle observations | 5379 | 5379 |
| Green → gray transitions | 0 | 0 |
| Gray → green transitions | 0 | 0 |
| Unknown vehicle observations | 100% | 100% |
| Frames with any unknown vehicle | 99.11% | 99.11% |
| Distance calls / valid ranges | 0 / 0 | 0 / 0 |
| Membership stage ms/frame | 1.081 | 1.473 |
| Pipeline FPS | 16.04 | 13.77 |

The membership-stage increase is 0.392 ms/frame. The overall FPS difference is -14.1%; it is not an isolated tracker cost measurement: total pipeline time changed by 10.26 ms/frame, considerably more than the membership increase. CPU model timings vary between runs. Debug JSON and encoding are outside the pipeline FPS measurement.

Lane visualization points match on **all 1,800 frames**. The final association produced 4,591 continuous matches, 437 reacquisitions with state reset, 312 new IDs and 39 ambiguous new IDs (351 unique IDs overall). These are fragmentation/association counts, **not measured ID switches**. No annotated vehicle identities are available to measure actual ID switches or tracking accuracy. An identical offline association is applied to before and after logs for transition counts, and births/disappearances are not counted as color changes.

Real-video raw-versus-smoothed distance variability is **unavailable**, because no range calls were allowed. The all-gray result is neither an improvement claim nor evidence that real lane changes are handled correctly. That behavior is tested separately below. To demonstrate actual inside-lane distance behavior on camera footage, a video/ROI combination with fully observed vehicles and usable ego boundaries plus applicable calibration is required. The default ROI was not enlarged and historical calibration was not silently reused.

## Controlled tests and evidence sequence

86 unittest regressions pass, covering multiple nearby vehicles and detection-order changes, separate distance histories, track misses/expiry/reassignment, ambiguous matches, initial confirmation, isolated and consecutive errors, uncertain boundary regions, a gradual geometric lane change with the same ID, missing/expired geometry, current tracked geometry on skipped inference frames, invalid/extreme distances, plausible rapid closing motion, stale history expiration, calibration reuse, green/gray pixels, point rendering and no polygon fills.

A reproducible 100-frame **synthetic evidence sequence** adds isolated classification errors, a genuine inside-to-outside change at frame 30, return at frame 60, and geometry loss at frame 80:

| Metric | Instantaneous | Persistent state |
|---|---:|---:|
| Green → gray | 5 | 2 |
| Gray → green | 4 | 2 |
| Distance step standard deviation | 2.836 m | 0.313 m |

The outside transition confirms at frame 35 (six observations); inside confirms at frame 62 (three observations). Missing geometry produces unknown at frame 89, after the 0.3-second hold. The single known synthetic vehicle retains its ID. These numbers are **not video accuracy measurements** and do not prove range accuracy. Median/EMA adds lag; aggressive closing behavior still needs application-specific evaluation before safety use.

## Reproduce

Run from repository root. The defaults enable vehicle tracking. The explicit command used for the after video is:

```bash
/home/nabinguyen/yes/envs/capstone/bin/python -m ai.src.fusion.tests.test_adas_pipeline \
  --input dataset/tests/lane_detector/test.MP4 \
  --output ai/outputs/adas/vehicle_after.mp4 \
  --lane-detection-interval 5 --lane-tracking --vehicle-tracking \
  --device cpu --threads 4 --warmup 5 --max-frames 1800 \
  --debug-jsonl ai/outputs/adas/vehicle_after.jsonl
```

For the before behavior, replace `--vehicle-tracking` with `--no-vehicle-tracking` and use `vehicle_before.mp4` / `vehicle_before.jsonl`. This keeps the working lane tracker enabled while restoring per-frame legacy membership and raw distance output. Snapshots of pre-change source are in `ai/outputs/vehicle_tracking_previous/`.

Optional tuning flags (defaults shown):

```text
--inside-confirm-frames 3
--outside-confirm-frames 6
--lane-inside-margin 0.10
--lane-outside-margin 0.05
--max-uncertain-hold-seconds 0.3
--vehicle-track-timeout-seconds 0.5
--distance-median-window 3
--distance-ema-alpha 0.35
--distance-max-age-seconds 0.5
--debug-lane-classification
```

Use source timestamps in seconds when calling the API. The existing video CLI supplies timestamps at the actual source FPS. No stale ranges are displayed during holds. `--camera-config` may be supplied with calibration that matches the video; absence means distance is unavailable.

```bash
/home/nabinguyen/yes/envs/capstone/bin/python -m unittest \
  ai.src.fusion.tests.test_vehicle_tracking \
  ai.src.lane_detection.tests.test_temporal_tracker \
  ai.src.fusion.tests.test_adas_pipeline_unit \
  ai.src.lane_detection.tests.test_twinlitenet \
  ai.src.fusion.tests.test_lane_intervals \
  ai.src.distance_estimation.tests.test_model \
  ai.src.distance_estimation.tests.test_image_pipeline -q

/home/nabinguyen/yes/envs/capstone/bin/python -m ai.src.fusion.tests.benchmark_vehicle_tracking \
  --before ai/outputs/adas/vehicle_before.jsonl \
  --after ai/outputs/adas/vehicle_after.jsonl \
  --output ai/outputs/adas/vehicle_comparison.json
```

Outputs: `ai/outputs/adas/vehicle_after.mp4`, per-frame `.jsonl`, `.benchmark.json`, and `vehicle_comparison.json`. Existing unrelated working-tree changes, older pipelines, TwinLiteNet+ implementation and temporal lane smoothing, VehicleModel and YOLO architecture, and distance-estimation source were preserved. Existing test fixtures explicitly disable the new vehicle layer when testing legacy behavior.
