"""TwinLiteNet+ and the existing VehicleModel, with no distance estimation."""
from time import perf_counter
from pathlib import Path
import cv2
from ai.src.lane_detection.twinlitenet import TwinLiteNetLaneModel, ROOT
from ai.src.vehicle_detection.model import VehicleModel

DEFAULT_VEHICLE_WEIGHTS = ROOT / 'ai/checkpoints/vehicle_detection/26n_best.pt'


class TwinLiteNetVehicleFusion:
    def __init__(self, lane_weights=None, vehicle_weights=DEFAULT_VEHICLE_WEIGHTS,
                 vehicle_conf=.25, lane_threshold=.5, device='auto', smooth=False,
                 point_radius=4, geometry=None):
        if point_radius < 1 or not 0 <= vehicle_conf <= 1:
            raise ValueError('Invalid point radius or vehicle confidence')
        if not vehicle_weights or not Path(vehicle_weights).is_file():
            raise FileNotFoundError(f'Vehicle checkpoint missing: {vehicle_weights}')
        kwargs = {} if lane_weights is None else {'weights': lane_weights}
        self.lane = TwinLiteNetLaneModel(**kwargs, threshold=lane_threshold, device=device, smooth=smooth, geometry=geometry)
        self.vehicle = VehicleModel(str(vehicle_weights), confidence_threshold=vehicle_conf)
        self.vehicle.model.to(str(self.lane.device))
        self.point_radius = point_radius
        self.timings = {}
        self.lanes = {}
        self.detections = []

    def process(self, frame):
        start = perf_counter()
        self.mask, self.lanes = self.lane.predict(frame)
        self.lane.sync()
        before = perf_counter()
        self.detections = self.vehicle.predict(frame)
        self.lane.sync()
        after = perf_counter()
        output = frame.copy()
        for name, color in [('left_ego', (0,255,0)), ('right_ego', (0,0,255))]:
            for point in self.lanes[name]:
                cv2.circle(output, point, self.point_radius, color, -1, cv2.LINE_AA)
        for d in self.detections:
            x1,y1,x2,y2 = d['bbox']
            color = (0,255,0)
            cv2.rectangle(output, (x1,y1), (x2,y2), color, 2)
            cv2.putText(output, f"{d['class_name']} {d['confidence']:.2f}",
                        (max(0,x1), max(20,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 2, cv2.LINE_AA)
        end = perf_counter()
        self.timings = dict(self.lane.timings, vehicle_inference_ms=(after-before)*1000,
                            visualization_ms=(end-after)*1000, pipeline_ms=(end-start)*1000)
        return output
