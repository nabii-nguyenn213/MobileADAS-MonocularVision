# Pixel-width vehicle depth

`DistanceEstimator` uses `fx * real_width_m / bbox_width_px`. It has no ML,
NumPy, OpenCV, or detector dependencies. Unknown classes, absent calibration,
nonfinite/degenerate boxes, and boxes touching or exceeding any image boundary
return `None`. Boundary rejection is conservative because clipping can hide the
physical width. Occlusion away from the boundary cannot be detected here.

The existing `LaneVehicleFusion` constructs the estimator once and calls it
immediately after its single `self.vehicle.predict(work_frame)` call. Original
vehicle dictionaries, class names, confidence, lane switching/smoothing, and lead
candidate selection are preserved. `result['vehicle_distances']` contains
`{'detection_index': i, 'distance_m': float_or_None}` aligned to `vehicles`.
Coordinates refer to the working frame, including when fusion cropping is enabled.

Fusion options: `distance_enabled=True`, `horizontal_fov_deg=None`,
`focal_length_px=None`, `distance_threshold_m=50.0`, `vehicle_widths=None`,
`show_distance=True`, `distance_reference_resolution=None` (width, height).
Supply either focal length or FOV, not both. Without either, depth is unavailable.
Default widths in meters: car 1.8, truck/bus 2.5, motorcycle/motorbike 0.8,
pickup_truck 2.0. These are configurable placeholders, not object measurements.
Visualization keeps confidence and lane status, appends meters, and uses red below
50 m, green at/above, gray unavailable. `show_distance=False` restores lane colors.
The threshold is a demonstration setting, not a safety-certified collision warning.

## Camera geometry and calibration

All resolutions are `(width, height)`; boxes are `xyxy`. For a reference focal
length, set `reference_resolution` to its calibration image size. With no
reference size, supplied focal length is in current uncropped source pixels.
FOV describes the full uncropped source (or reference when supplied), not the
YOLO network's internal resized/letterboxed tensor. VehicleModel already returns
boxes in input-frame coordinates, so no YOLO `img_size` scaling is needed.

For external crop/resize, call `estimate(..., source_resolution=(w,h),
crop_box=(x1,y1,x2,y2))`, where the crop is in source pixels and
`image_resolution` is the final resized crop. Horizontal focal scaling is
`fx_reference * source_width/reference_width * output_width/crop_width`.
Cropping alone preserves fx; padding must be removed before this API is used.
Fusion automatically supplies geometry for its existing crop. Changing camera
zoom, stabilization crop, lens, or camera mode requires updated calibration.

Calibrate with an unobstructed, rear/front-facing vehicle at a measured optical
axis distance. Measure its actual width and its full box in the uncropped image:

```python
from ai.src.distance_estimation import DistanceEstimator
est = DistanceEstimator()
fx = est.calibrate(bbox, (image_width, image_height), known_distance_m,
                   real_width_m=measured_vehicle_width_m)
# Persist fx and est.reference_resolution; pass them to LaneVehicleFusion as
# focal_length_px=fx, distance_reference_resolution=est.reference_resolution.
```

Calibration sets `fx = distance * pixel_width / real_width`. Check several known
distances; one sample does not establish accuracy. Real class dimensions vary,
yaw exposes vehicle length, detector box error/occlusion changes apparent width,
and lens distortion or incorrect intrinsics biases depth. Results approximate
optical-axis camera-to-object depth, not Euclidean range or bumper clearance.
No particular smartphone or iPhone FOV is assumed.

## Run and verify

From the repository root, using the environment with project dependencies:

```bash
python -m unittest discover -s ai/src/distance_estimation/tests -v
python -m unittest discover -s ai/src/fusion/tests -p test_distance_integration.py -v
python -m ai.main --video dataset/tests/lane_detector/test.MP4 --vehicle-weights ai/checkpoints/vehicle_detection/26n_best.pt --focal-length-px YOUR_CALIBRATED_FX --distance-reference-resolution WIDTH HEIGHT --no-crop --display --output ai/outputs/lane_vehicle_detector/distance_calibrated.mp4
```

`--display` enables viewing and q to quit; omit it on headless hosts. `--no-crop`
preserves source resolution and aspect ratio. Source FPS is retained (30 FPS
fallback for invalid metadata). MP4 saving uses the existing OpenCV writer.
`--max-frames` allows bounded smoke tests. `--hide-distance`, `--no-distance`,
`--distance-threshold-m`, and `--vehicle-widths widths.json` are supported. The
existing `python -m ai.src.fusion.tests.test_lane_vehicle` entry point delegates
to the same CLI. Lane weights are required at
`ai/checkpoints/lane_detection/{tusimple,curvelanes}_res34.pth`.

A 30-frame smoke output is generated at
`ai/outputs/lane_vehicle_detector/distance_demo.mp4` using **illustrative 90° FOV**,
not measured camera metadata. It verifies execution, not metric accuracy.

Verification in the project's `capstone` environment: 11 estimator tests and 3
fusion/rendering tests passed with unittest. Pytest was unavailable. The real
30-frame inference run loaded both ResNet34 checkpoints and `26n_best.pt`,
completed on CPU at approximately 0.2 processing FPS, and its MP4 decoded back
to 30 frames at 1920x1440 and 30 FPS (4,261,636 bytes). A decoded frame was
visually inspected for lane points, boxes, confidence, and distance labels.
Interactive display was not exercised on the headless host. All required
weights and the sample video were present; no checkpoints were changed.

## Still-image evaluation with configured camera geometry

`tests/test_distance_images.py` reuses `VehicleModel` and `DistanceEstimator`.
It requires explicit camera geometry and an approximate target width; it does
not calibrate on a labeled distance image. The former `--calibration-image`
argument has been removed. Filenames supply ground truth only after prediction.

From the repository root, replace the shell variables with your camera/vehicle
measurements before running:

```bash
python -m ai.src.distance_estimation.tests.test_distance_images \
  --input-dir dataset/distance \
  --output-dir dataset/distance/results/camera_configured \
  --horizontal-fov-deg "$CAMERA_HORIZONTAL_FOV_DEG" \
  --real-width-m "$VEHICLE_WIDTH_M" \
  --imgsz 960 --conf 0.25 --selection-method robust \
  --roi 0.25 0.25 0.75 0.65 --debug-detections
```

Alternatively use `--focal-length-px "$CAMERA_FX_PX" --reference-resolution
"$CAMERA_WIDTH_PX" "$CAMERA_HEIGHT_PX"` instead of `--horizontal-fov-deg`.
Pixel focal length means **fx in pixels**, not EXIF focal length in millimeters.
Obtain fx from a camera intrinsics calibration (for example, a checkerboard
captured with the same lens, zoom, resolution and camera mode), or use a documented
horizontal FOV for that exact image mode. A diagonal FOV is not a horizontal FOV.
If physical focal length and active sensor width are known, fx in pixels is
physical focal length / active sensor width * reference image width; sensor crop
and digital zoom must be accounted for. Do not guess the phone camera FOV.

Measure the target width or explicitly supply an approximate width. Width errors
scale distance estimates proportionally. `--camera-config camera.json` can supply
`focal_length_px` or `horizontal_fov_deg`, `reference_resolution: [width, height]`,
and `vehicle_widths: {"car": width_in_meters}` instead. CLI geometry overrides
JSON geometry; CLI width and reference resolution override corresponding JSON
values. All configuration values are recorded in `summary.json` and the CSV.

YOLO crop boxes are translated back into original-image pixels before distance
estimation. The estimator receives the **full original image resolution**;
neither YOLO inference size nor ROI size is the camera reference resolution.
Use images from the same lens/zoom/image mode for a shared configuration.

`--sweep` compares sizes 640/960/1280, confidences .25/.15/.10 and both selection
methods, plus full-image and cropped inference if `--roi` is supplied. It uses
one loaded detector and the same configured estimator throughout. Each trial
writes annotations, all post-NMS candidates with selection reasons, diagnostics,
metrics and a CSV; the sweep writes `comparison.csv`. These are detector outputs
after confidence filtering/NMS, not raw neural-network logits. The normal path
runs YOLO exactly once per image. The robust selector ranks plausible cars by
location and confidence, applies broad dimension/boundary gates, and reports
ambiguity when scores are too close. It does not recognize vehicle color or brand;
review diagnostics or supply `--target-points` / `--target-overrides` when needed.
Conflicting duplicate labels remain excluded from aggregate accuracy.

Earlier `results/` and `results/experiments/` files were generated using the 5m
reference-image calibration; they are historical results, not camera-configured
predictions. Write new runs to a separate output directory to retain provenance.
