# Lane interval tracking verification

Compared the first 300 frames (10 seconds, 1920×1440, 30 FPS) of `dataset/tests/lane_detector/test.MP4` on CPU with four Torch threads. Five warmup frames are excluded; each run resets state and scheduling before measurement.

| N | Frames / tracker updates | TwinLiteNet+ calls | Any boundary visible | Both visible | Mean lane predict call (ms) | Pipeline FPS | Visibility transitions |
|---|---|---|---|---|---|---|---|
| 1 | 300 / 300 | 300 | 89.3% | 33.3% | 132.1 | 5.99 | 11 |
| 5 | 300 / 300 | 60 | 72.7% | 25.7% | 150.0 | 13.72 | 11 |
| 10 | 300 / 300 | 30 | 66.7% | 24.0% | 130.8 | 21.85 | 15 |

Lane predict latency includes preprocessing, forward inference, decoding, and the existing model extractor; it is averaged over actual calls, not all frames. Pipeline FPS includes lane tracking, vehicle detection, membership, distance gating, and rendering; decoding/encoding are reported separately in benchmark JSON. Hardware load can affect these measurements.

## Continuity and limitations

Per-frame JSON audit: {"1": {"skipped_frame_boundary_disappearances": 0, "explained_by_expiry": 0}, "5": {"skipped_frame_boundary_disappearances": 5, "explained_by_expiry": 5}, "10": {"skipped_frame_boundary_disappearances": 7, "explained_by_expiry": 7}}. All skipped-frame boundary disappearances were due to the configured observation timeout. None disappeared merely because inference was skipped. Visibility transitions count each left/right change and include initial acquisition and timeout/reacquisition. Raw visibility statistics describe scheduled observations only and should not be interpreted as a per-frame baseline when N > 1.

The reviewed frame at index 100 is a turning intersection with weak or occluded boundaries. All runs produced 693 vehicle detections, all classified unknown, with zero distance calls. This clip segment therefore validates conservative abstention, but does not demonstrate successful inside-lane distance predictions. No matching camera calibration was supplied. Unit tests verify inside/outside/unknown behavior, green/gray colors, and one existing estimator call per inside vehicle, including reliable skipped frames.

EMA holds a fitted target between observations; no optical flow or motion compensation was added. Positional lag/drift during turns remains possible, particularly at N=10. There is no annotated ground truth to quantify drift. Rejected jumps and missing detections still cause expiry and later reacquisition; these results do not claim flicker-free rendering when the model cannot supply reliable geometry. N=5 balances efficiency against observation frequency in this comparison.

## Reproduce

From repository root, run each N separately (1, 5, and 10):

```bash
/home/nabinguyen/yes/envs/capstone/bin/python -m ai.src.fusion.tests.test_adas_pipeline \
  --input dataset/tests/lane_detector/test.MP4 \
  --output ai/outputs/adas/interval5.mp4 \
  --lane-detection-interval 5 --lane-tracking \
  --lane-smoothing-alpha 0.25 --lane-max-hold-seconds 0.5 \
  --device cpu --threads 4 --warmup 5 --max-frames 300 \
  --debug-jsonl ai/outputs/adas/interval5.jsonl
```

Change interval and output filenames for N=1 and N=10. Remove `--max-frames 300` for the full video. Add `--lane-debug` to log tracking decisions. Supply `--camera-config` only with calibration applicable to this video.

```bash
/home/nabinguyen/yes/envs/capstone/bin/python -m unittest \
  ai.src.lane_detection.tests.test_temporal_tracker \
  ai.src.fusion.tests.test_adas_pipeline_unit \
  ai.src.lane_detection.tests.test_twinlitenet \
  ai.src.fusion.tests.test_lane_intervals -q
```

36 tests pass. Pytest is unavailable in this environment; the tests are unittest-compatible.

## Configuration and implementation

The pipeline defaults to N=5, alpha=0.25, hold=0.5 seconds, and tracking enabled. It detects on frame indices 0,N,2N. YOLO and rendering run every frame. The existing TemporalLaneTracker now keeps last accepted geometry, target geometry, smoothed parameters, observation time, confidence, and status per side. It advances EMA every frame without refreshing observation age on skips. Scheduled missing/rejected observations immediately revoke classification trust while retaining display geometry until timeout.

The CLI supplies actual source FPS. Direct API users should supply `video_fps` (default 30) and source timestamps in seconds; absent timestamps are derived from frame index and FPS. Hold time must cover N/FPS. Default classification age is one detection interval capped by hold time; confidence, valid pair geometry, and observation validity also gate classification. `--lane-classification-max-age 0` restores fresh-observation-only classification. Scene changes, resolution changes, and timestamp discontinuities reset tracking.

TwinLiteNet+, YOLO architecture, distance algorithm, ROI preset, and points-only renderer are unchanged. Existing LaneSmoother is left intact for its older consumers. The original extractor clears missing-side history, the older LaneSmoother resets on missing input, and the prior tracker only advanced EMA on detections with default zero classification age. The fusion pipeline previously invoked lane inference every frame; no interval scheduler existed.

Pre-change source snapshots are in `ai/outputs/lane_tracking_previous/`. The original per-frame untracked path remains available through `--lane-detection-interval 1 --no-lane-tracking`. Existing unrelated working-tree edits were preserved.
