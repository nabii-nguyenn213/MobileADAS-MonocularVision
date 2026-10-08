import inspect
from pathlib import Path

import numpy as np

from ai.src.lane_detection.tests.test_hybrid_lane import HybridLaneTest
from ai.src.vehicle_detection.model import VehicleModel
from ai.src.fusion.lane_vehicle import draw_lane_vehicles


class LaneVehicleFusion:
    """Run existing HybridLaneTest (two lane models) + VehicleModel (YOLO26n)."""

    DEFAULT_WEIGHTS = (
        Path(__file__).resolve().parents[2]
        / "checkpoints" / "vehicle_detection" / "26n_best.pt"
    )

    def __init__(
        self,
        *,
        vehicle_weights=None,
        vehicle_conf=0.25,
        img_size=640,
        tusimple_threshold=0.60,
        curvelanes_threshold=0.60,
        switch_margin=0.08,
        required_frames=3,
        point_radius=5,
        lane_margin=0,
        crop_16_9=True,
    ):
        # HybridLaneTest already builds both ResNet34-based lane detectors,
        # loads checkpoints and controls confidence/hysteresis selection.
        self.lane = HybridLaneTest(
            tusimple_threshold=tusimple_threshold,
            curvelanes_threshold=curvelanes_threshold,
            switch_margin=switch_margin,
            required_frames=required_frames,
            point_radius=point_radius,
        )

        # Support either constructor spelling found in project versions:
        # uploaded test: confidence_threshold; earlier model: conf_threshold.
        kwargs = {
            "model_path": str(vehicle_weights or self.DEFAULT_WEIGHTS),
            "img_size": img_size,
        }
        params = inspect.signature(VehicleModel).parameters
        if "confidence_threshold" in params:
            kwargs["confidence_threshold"] = vehicle_conf
        elif "conf_threshold" in params:
            kwargs["conf_threshold"] = vehicle_conf
        else:
            raise TypeError(
                "VehicleModel constructor has no confidence_threshold "
                "or conf_threshold parameter. Check its real API."
            )
        self.vehicle = VehicleModel(**kwargs)

        self.lane_margin = lane_margin
        self.crop_16_9 = crop_16_9

    def reset(self):
        """Call before a new independent video to reset hybrid model selection."""
        self.lane.reset()

    def process_frame(self, frame):
        """Return (annotated_frame, result) for a single BGR video frame."""
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("frame must be a non-empty numpy array")

        # Apply the SAME crop before BOTH models, so all pixel positions agree.
        work_frame = (
            self.lane.crop_to_16_9(frame)
            if self.crop_16_9 else frame
        )

        # Reuse existing hybrid inference and its hysteresis selector.
        lane_result = self.lane._predict_both(work_frame)
        ego_lane = lane_result.get("ego_lane") or {}
        left_lane = ego_lane.get("left")
        right_lane = ego_lane.get("right")

        # Reuse existing YOLO model; no new vehicle model logic.
        detections = self.vehicle.predict(work_frame)

        # Reuse the hybrid's original point-only lane visualization.
        lane_frame = self.lane._visualize(work_frame, lane_result)

        # Overlay vehicle boxes without drawing the lane points twice.
        output, in_lane, lead = draw_lane_vehicles(
            lane_frame,
            detections,
            left_lane,
            right_lane,
            margin=self.lane_margin,
            show_lane_points=False,
        )

        result = {
            "selected_lane_model": lane_result["model"],
            "tusimple_score": lane_result["tusimple_score"],
            "curvelanes_score": lane_result["curvelanes_score"],
            "ego_lane": ego_lane,
            "vehicles": detections,
            "in_lane_vehicles": in_lane,
            "lead_vehicle": lead,
        }
        return output, result
