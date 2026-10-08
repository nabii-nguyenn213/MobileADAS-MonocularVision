import cv2
import numpy as np


def _clean_points(points):
    """Return finite (x, y) points; tolerate absent/invalid lane points."""
    clean = []
    if points is None:
        return np.empty((0, 2), dtype=np.float32)

    for point in points:
        if point is None:
            continue
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError, IndexError):
            continue
        if np.isfinite(x) and np.isfinite(y):
            clean.append((x, y))

    return np.asarray(clean, dtype=np.float32).reshape(-1, 2)


def lane_x_at_y(points, y):
    """Interpolate a boundary's x at image row y; never extrapolate."""
    pts = _clean_points(points)
    if len(pts) < 2 or not np.isfinite(y):
        return None

    # np.unique sorts y values and merges duplicate rows.
    unique_y, inverse = np.unique(pts[:, 1], return_inverse=True)
    if len(unique_y) < 2 or not unique_y[0] <= y <= unique_y[-1]:
        return None

    mean_x = (
        np.bincount(inverse, weights=pts[:, 0])
        / np.bincount(inverse)
    )
    return float(np.interp(y, unique_y, mean_x))


def classify_vehicle_lane(bbox, left_lane, right_lane, margin=0):
    """Return 'IN_LANE', 'OTHER_LANE' or 'UNKNOWN'.

    The vehicle ground-contact position is approximated by the bounding box's
    bottom center. This is a heuristic, not a 3D distance measurement.
    """
    if margin < 0:
        raise ValueError("margin must be non-negative")

    try:
        x1, y1, x2, y2 = map(float, bbox)
    except (TypeError, ValueError):
        return "UNKNOWN"

    if not all(map(np.isfinite, (x1, y1, x2, y2))):
        return "UNKNOWN"
    if x2 <= x1 or y2 <= y1:
        return "UNKNOWN"

    center_x = (x1 + x2) / 2.0
    bottom_y = y2
    left_x = lane_x_at_y(left_lane, bottom_y)
    right_x = lane_x_at_y(right_lane, bottom_y)

    if left_x is None or right_x is None or left_x >= right_x:
        return "UNKNOWN"
    if left_x + margin > right_x - margin:
        return "UNKNOWN"

    if left_x + margin <= center_x <= right_x - margin:
        return "IN_LANE"
    return "OTHER_LANE"


def is_vehicle_in_lane(bbox, left_lane, right_lane, margin=0):
    return (
        classify_vehicle_lane(bbox, left_lane, right_lane, margin)
        == "IN_LANE"
    )


def filter_vehicles_in_lane(detections, left_lane, right_lane, margin=0):
    """Return the original detection dictionaries for in-lane vehicles."""
    return [
        d for d in detections
        if is_vehicle_in_lane(d["bbox"], left_lane, right_lane, margin)
    ]


def select_lead_vehicle(vehicles):
    """Choose a provisional lead candidate by the lowest box edge in image.

    This is NOT reliably the physically closest vehicle. Replace this heuristic
    with distance estimates when the distance model is integrated.
    """
    return max(vehicles, key=lambda d: d["bbox"][3], default=None)


def draw_lane_points(frame, left_lane, right_lane, radius=5):
    """Draw isolated lane points only (no lines, triangles or filled areas)."""
    output = frame.copy()
    # Same BGR colors as HybridLaneTest: left=green, right=red.
    for lane, color in (
        (left_lane, (0, 255, 0)),
        (right_lane, (0, 0, 255)),
    ):
        for x, y in _clean_points(lane):
            cv2.circle(
                output, (int(x), int(y)), radius, color, -1, cv2.LINE_AA
            )
    return output


def draw_lane_vehicles(
    frame,
    detections,
    left_lane,
    right_lane,
    margin=0,
    show_lane_points=True,
):
    """Draw colored boxes and return (annotated_frame, in_lane, lead).

    Red box: provisional lead candidate
    Green: other in-lane vehicles
    Gray: other-lane vehicles
    Yellow: unknown lane membership

    Set show_lane_points=False if HybridLaneTest._visualize() already drew them.
    """
    if show_lane_points:
        output = draw_lane_points(frame, left_lane, right_lane)
    else:
        output = frame.copy()

    statuses = [
        classify_vehicle_lane(d["bbox"], left_lane, right_lane, margin)
        for d in detections
    ]
    in_lane_indices = [i for i, status in enumerate(statuses) if status == "IN_LANE"]
    in_lane = [detections[i] for i in in_lane_indices]

    lead_index = (
        max(in_lane_indices, key=lambda i: detections[i]["bbox"][3])
        if in_lane_indices else None
    )
    lead = detections[lead_index] if lead_index is not None else None

    colors = {
        "IN_LANE": (0, 255, 0),      # Green
        "OTHER_LANE": (150, 150, 150),  # Gray
        "UNKNOWN": (0, 255, 255),     # Yellow
    }

    for index, (detection, status) in enumerate(zip(detections, statuses)):
        x1, y1, x2, y2 = map(int, detection["bbox"])
        if index == lead_index:
            label_status = "LEAD CANDIDATE"
            color = (0, 0, 255)   # Red
        else:
            label_status = status.replace("_", " ")
            color = colors[status]

        class_name = detection.get("class_name", "vehicle")
        confidence = float(detection.get("confidence", 0.0))
        label = f"{label_status} | {class_name} {confidence:.2f}"

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            output,
            label,
            (max(0, x1), max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    return output, in_lane, lead
