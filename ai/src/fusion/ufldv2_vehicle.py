"""Independent UFLDv2 + existing VehicleModel visualization, without distance."""
from pathlib import Path

import cv2

from ai.src.lane_detection.ufldv2 import DEFAULT_LANE_WEIGHTS, ROOT, UFLDv2, parse_device
from ai.src.vehicle_detection.model import VehicleModel
from ai.src.vehicle_detection.visualizer import draw_detections

DEFAULT_VEHICLE_WEIGHTS = ROOT / "ai/checkpoints/vehicle_detection/26n_best.pt"
LANE_COLORS = ((255, 200, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255))


class UFLDv2VehicleFusion:
    def __init__(self, lane_weights=DEFAULT_LANE_WEIGHTS,
                 vehicle_weights=DEFAULT_VEHICLE_WEIGHTS,
                 vehicle_conf=0.25, device="auto"):
        if not Path(vehicle_weights).is_file():
            raise FileNotFoundError(f"Vehicle weights not found: {vehicle_weights}")
        if not 0 <= vehicle_conf <= 1:
            raise ValueError("vehicle_conf must be between 0 and 1")
        kind, index = parse_device(device)
        self.lane_model = UFLDv2(lane_weights, device=device)
        self.vehicle_model = VehicleModel(
            model_path=str(vehicle_weights), confidence_threshold=vehicle_conf
        )
        # Use Ultralytics' existing device adapter without changing VehicleModel.
        if kind != "auto":
            self.vehicle_model.model.to("cpu" if kind == "cpu" else f"cuda:{index}")
        self.last_lanes = []
        self.last_vehicles = []

    def predict(self, frame):
        self.last_lanes = self.lane_model.predict(frame)
        self.last_vehicles = self.vehicle_model.predict(frame)
        output = draw_detections(frame, self.last_vehicles)
        # Model slots 1 and 2 are ego boundaries; no previous selection logic.
        ego = [lane for lane in self.last_lanes if lane.lane_id in (1, 2) and lane.detected]
        visible = ego or [lane for lane in self.last_lanes if lane.detected]
        for lane in visible:
            for point in lane.points:
                cv2.circle(output, point, 4, LANE_COLORS[lane.lane_id], -1, cv2.LINE_AA)
        return output
