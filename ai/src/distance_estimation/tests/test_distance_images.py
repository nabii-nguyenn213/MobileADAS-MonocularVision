"""Evaluate existing vehicle and distance models on labeled still images.

Camera geometry and approximate car width must be explicitly configured.
Filename distances are used only for evaluation, never calibration or prediction.
--camera-config accepts DistanceEstimator constructor arguments as JSON.
--target-points accepts JSON mapping filenames to normalized [x, y] points;
expected location defaults to the image center. See --help for selection and ROI options.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re

import cv2

from ai.src.vehicle_detection.model import VehicleModel
from ai.src.distance_estimation.model import DistanceEstimator

ROOT = Path(__file__).resolve().parents[4]


def select_target(detections, resolution, point, method="robust", override=None):
    """Rank plausible cars; require a clear score margin, never rank by size alone.

    Spatial radius is normalized to image dimensions. Dimensions are a broad
    validity gate, not a proxy for target distance. Manual IDs refer to candidates
    from this exact configuration, so they must be reviewed in diagnostics.
    """
    w, h = resolution
    eligible = []
    for index, d in enumerate(detections):
        x1, y1, x2, y2 = d['bbox']
        bw, bh = x2-x1, y2-y1
        cx, cy = (x1+x2)/2, (y1+y2)/2
        proximity = math.hypot(cx/w-point[0], cy/h-point[1])
        reason = ('unexpected_class' if d['class_name'] != 'car' else
                  'clipped_box' if d.get('crop_clipped') or not (0 < x1 < x2 < w and 0 < y1 < y2 < h) else
                  'implausible_dimensions' if min(bw, bh) < 4 or not .4 <= bw/bh <= 3 else
                  'outside_target_neighborhood' if proximity > .22 else '')
        if method == 'center' and not (x1 < point[0]*w < x2 and y1 < point[1]*h < y2):
            reason = reason or 'does_not_contain_target_point'
        d.update(candidate_id=index, width_px=bw, height_px=bh, center=[cx, cy],
                 score=round(.7*(1-min(proximity/.22, 1))+.3*d['confidence'], 6),
                 passed_selection=False, rejection_reason=reason)
        if not reason:
            eligible.append(d)
    eligible.sort(key=lambda d: (-d['score'], d['candidate_id']))
    chosen, status = None, 'not_detected' if not detections else 'not_selected'
    if override is not None:
        if not isinstance(override, int) or not 0 <= override < len(detections):
            raise ValueError('Manual candidate ID is invalid for this configuration')
        chosen = detections[override]
        # Manual review may override proximity or class, but never invalid geometry.
        if chosen['rejection_reason'] in {'clipped_box', 'implausible_dimensions'}:
            raise ValueError('Manual target has invalid geometry')
    elif eligible:
        if len(eligible) > 1 and (method == 'center' or eligible[0]['score']-eligible[1]['score'] < .12):
            status = 'ambiguous'
        else:
            chosen = eligible[0]
    for d in detections:
        if d is chosen:
            d.update(passed_selection=True, rejection_reason='')
        elif not d['rejection_reason']:
            d['rejection_reason'] = 'ambiguous' if status == 'ambiguous' else 'lower_combined_score'
    return chosen, 'selected' if chosen else status


def infer(detector, frame, roi=None):
    """One detector call; translate crop detections to original pixels."""
    h, w = frame.shape[:2]
    left, top, right, bottom = (0, 0, w, h)
    if roi:
        left, top, right, bottom = [int(v*n) for v, n in zip(roi, (w, h, w, h))]
        if right <= left or bottom <= top:
            raise ValueError('ROI has no pixels')
    detections = detector.predict(frame[top:bottom, left:right])
    for d in detections:
        x1, y1, x2, y2 = d['bbox']
        d['crop_clipped'] = bool(roi and (x1 <= 0 or y1 <= 0 or x2 >= right-left-1 or y2 >= bottom-top-1))
        d['bbox'] = [x1+left, y1+top, x2+left, y2+top]
    return detections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=ROOT / 'dataset/distance')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'dataset/distance/results')
    parser.add_argument('--weights', type=Path, default=ROOT / 'ai/checkpoints/vehicle_detection/26n_best.pt')
    parser.add_argument('--camera-config', type=Path)
    geometry = parser.add_mutually_exclusive_group()
    geometry.add_argument('--horizontal-fov-deg', type=float, help='Horizontal FOV of the original full image, in degrees')
    geometry.add_argument('--focal-length-px', type=float, help='Camera fx in pixels at --reference-resolution')
    parser.add_argument('--reference-resolution', type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'),
                        help='Original camera resolution for fx; required with pixel focal length')
    parser.add_argument('--real-width-m', type=float, help='Approximate physical target width in meters; required unless configured in JSON')
    parser.add_argument('--target-points', type=Path)
    parser.add_argument('--confidence', '--conf', dest='confidence', type=float, default=0.25)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--selection-method', choices=['center', 'robust'], default='robust')
    parser.add_argument('--roi', type=float, nargs=4, metavar=('X1','Y1','X2','Y2'),
                        help='Normalized inference crop; boxes mapped to original pixels')
    parser.add_argument('--debug-detections', action='store_true')
    parser.add_argument('--target-overrides', type=Path, help='JSON filename -> candidate ID, reviewed per configuration')
    parser.add_argument('--sweep', action='store_true', help='Compare 3 sizes x 3 thresholds x 2 selectors, full image and optional ROI')
    args = parser.parse_args()
    if not 0 < args.confidence <= 1 or args.imgsz <= 0:
        parser.error('Confidence must be in (0,1] and imgsz positive')
    if args.roi and not (0 <= args.roi[0] < args.roi[2] <= 1 and 0 <= args.roi[1] < args.roi[3] <= 1):
        parser.error('ROI must be normalized xyxy within the image')
    config = json.loads(args.camera_config.read_text()) if args.camera_config else {}
    if args.horizontal_fov_deg is not None or args.focal_length_px is not None:
        config.pop('horizontal_fov_deg', None)
        config.pop('focal_length_px', None)
        config['horizontal_fov_deg'] = args.horizontal_fov_deg
        config['focal_length_px'] = args.focal_length_px
    if args.reference_resolution is not None:
        config['reference_resolution'] = args.reference_resolution
    if args.real_width_m is not None:
        config.setdefault('vehicle_widths', {})['car'] = args.real_width_m
    if config.get('focal_length_px') is None and config.get('horizontal_fov_deg') is None:
        parser.error('Camera parameters unknown. Supply --horizontal-fov-deg or --focal-length-px '
                     '(with --reference-resolution), or --camera-config. Obtain horizontal FOV for '
                     'the actual lens/image mode or fx from a camera intrinsics calibration; '
                     'do not infer these from distance-labeled filenames.')
    if config.get('focal_length_px') is not None and config.get('reference_resolution') is None:
        parser.error('Pixel focal length requires --reference-resolution WIDTH HEIGHT (or reference_resolution in JSON)')
    if config.get('vehicle_widths', {}).get('car') is None:
        parser.error('Supply --real-width-m or vehicle_widths.car in --camera-config (approximate target width in meters)')
    try:
        estimator = DistanceEstimator(**config)
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    detector = VehicleModel(model_path=str(args.weights), img_size=args.imgsz,
                            confidence_threshold=args.confidence)
    if args.sweep:
        import copy
        all_rows = []
        base = args.output_dir
        for roi in ([None, args.roi] if args.roi else [None]):
            for size in (640, 960, 1280):
                for conf in (.25, .15, .10):
                    for method in ('center', 'robust'):
                        trial = copy.copy(args)
                        trial.imgsz, trial.confidence, trial.selection_method, trial.roi = size, conf, method, roi
                        trial.output_dir = base / f"{'roi' if roi else 'full'}_{size}_{conf:.2f}_{method}"
                        all_rows.extend(run(trial, detector, estimator))
        base.mkdir(parents=True, exist_ok=True)
        with (base / 'comparison.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)
    else:
        run(args, detector, estimator)


def run(args, detector, estimator):
    detector.img_size, detector.confidence_threshold = args.imgsz, args.confidence
    points = json.loads(args.target_points.read_text()) if args.target_points else {}
    overrides = json.loads(args.target_overrides.read_text()) if args.target_overrides else {}
    records = []
    for path in args.input_dir.expanduser().iterdir():
        if path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.webp'}:
            continue
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f'Cannot decode {path}')
        height, width = frame.shape[:2]
        digest = hashlib.sha256(str(frame.shape).encode() + frame.tobytes()).hexdigest()
        detections = infer(detector, frame, args.roi)
        point = points.get(path.name, [0.5, 0.5])
        if len(point) != 2 or not all(0 <= v <= 1 for v in point):
            raise ValueError(f'Invalid normalized target point for {path.name}')
        selected, selection_status = select_target(detections, (width,height), point,
                                                   args.selection_method, overrides.get(path.name))
        # Predict before reading the ground-truth filename label.
        predicted = estimator.estimate(selected['bbox'], (width, height), 'car') if selected else None
        match = re.fullmatch(r'(\d+(?:\.\d+)?)m', path.stem)
        if not match or float(match[1]) <= 0:
            raise ValueError(f'Expected positive distance filename: {path.name}')
        records.append(dict(path=path, frame=frame, resolution=(width, height),
                            digest=digest, actual=float(match[1]), predicted=predicted, selected=selected, selection_status=selection_status,
                            detections=detections))
    records.sort(key=lambda r: (r['actual'], r['path'].name))
    if not records:
        raise ValueError('No labeled images found')
    groups = {}
    for r in records:
        groups.setdefault(r['digest'], []).append(r)
    conflicts = {key for key, group in groups.items() if len({r['actual'] for r in group}) > 1}
    for key in conflicts:
        print('CONFLICTING DUPLICATE LABELS:', ', '.join(r['path'].name for r in groups[key]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, errors, percentages, seen = [], [], [], set()
    for r in records:
        d = r['selected']
        predicted = r['predicted']
        error = abs(predicted - r['actual']) if predicted is not None else None
        percentage = 100 * error / r['actual'] if error is not None else None
        exclusion = ('conflicting_duplicate_labels' if r['digest'] in conflicts else
                     'duplicate_content' if r['digest'] in seen else '')
        seen.add(r['digest'])
        status = (r['selection_status'] if d is None else 'ok' if predicted is not None else
                  'uncalibrated' if estimator.effective_focal_length(r['resolution']) is None else 'invalid_box')
        if status == 'ok' and not exclusion:
            errors.append(error)
            percentages.append(percentage)
        row = dict(imgsz=args.imgsz, conf_threshold=args.confidence, selection_method=args.selection_method, roi=json.dumps(args.roi), focal_length_px=estimator.focal_length_px, horizontal_fov_deg=estimator.horizontal_fov_deg, real_width_m=estimator.vehicle_widths['car'], reference_resolution=json.dumps(estimator.reference_resolution), image=r['path'].name, actual_m=r['actual'], predicted_m=predicted,
                   absolute_error_m=error, percentage_error=percentage,
                   bbox_width_px=d['bbox'][2] - d['bbox'][0] if d else None,
                   confidence=d['confidence'] if d else None, bbox_xyxy=json.dumps(d['bbox']) if d else '',
                   status=status, metric_exclusion=exclusion, content_sha256=r['digest'])
        rows.append(row)
        frame = r['frame'].copy()
        diagnostic = frame.copy()
        for candidate in r['detections']:
            x1,y1,x2,y2 = candidate['bbox']
            color = (0,255,0) if candidate['passed_selection'] else (0,165,255)
            cv2.rectangle(diagnostic, (x1,y1), (x2,y2), color, 2)
            label = f"#{candidate['candidate_id']} {candidate['class_name']} {candidate['confidence']:.3f}"
            cv2.putText(diagnostic, label, (max(0,x1-20),max(15,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
            if args.debug_detections:
                print(r['path'].name, json.dumps(candidate))
        if not cv2.imwrite(str(args.output_dir / (r['path'].stem + '_diagnostic.jpg')), diagnostic):
            raise OSError('Cannot write diagnostic image')
        if d:
            x1, y1, x2, y2 = d['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        lines = [f"Car | Confidence: {d['confidence']:.2f}" if d else f"Car | {status}",
                 f"Bounding-box width: {d['bbox'][2]-d['bbox'][0]} px" if d else 'Bounding-box width: unavailable',
                 f'Predicted: {predicted:.2f} m' if predicted is not None else f'Predicted: unavailable ({status})',
                 f"Actual: {r['actual']:.1f} m",
                 f'Error: {error:.2f} m' if error is not None else 'Error: unavailable']
        if exclusion:
            lines.append('Excluded: ' + exclusion)
        scale = max(0.55, r['resolution'][0] / 1800)
        step = int(36 * scale)
        for i, line in enumerate(lines):
            pos = (20, 30 + i * step)
            cv2.putText(frame, line, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5)
            cv2.putText(frame, line, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2)
        output = args.output_dir / (r['path'].stem + '_predicted.jpg')
        if not cv2.imwrite(str(output), frame):
            raise OSError(f'Cannot write {output}')
    with (args.output_dir / 'predictions.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(evaluation_count=len(errors), mae_m=sum(errors)/len(errors) if errors else None,
                   rmse_m=math.sqrt(sum(e*e for e in errors)/len(errors)) if errors else None,
                   mape_percent=sum(percentages)/len(percentages) if errors else None,
                   prediction_source="configured_camera_geometry", focal_length_px=estimator.focal_length_px,
                   horizontal_fov_deg=estimator.horizontal_fov_deg,
                   reference_resolution=estimator.reference_resolution,
                   vehicle_widths=estimator.vehicle_widths, weights=str(args.weights),
                   selection=args.selection_method, imgsz=args.imgsz, confidence_threshold=args.confidence, roi=args.roi,
                   camera_config=str(args.camera_config) if args.camera_config else None)
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    # Preserve detector candidates for selection review without rerunning YOLO.
    (args.output_dir / 'detections.json').write_text(json.dumps({r['path'].name: r['detections'] for r in records}, indent=2) + '\n')
    print('| Image | Actual (m) | Predicted (m) | Error (m) | Status / exclusion |')
    print('|---|---:|---:|---:|---|')
    for row in rows:
        values = [f'{row[key]:.2f}' if row[key] is not None else 'N/A' for key in ('actual_m', 'predicted_m', 'absolute_error_m')]
        print('| ' + ' | '.join([row['image'], *values, row['status'] + ' / ' + row['metric_exclusion']]) + ' |')
    print(json.dumps(summary, indent=2))
    return rows


if __name__ == '__main__':
    main()
