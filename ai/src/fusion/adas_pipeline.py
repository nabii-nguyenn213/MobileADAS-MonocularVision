"""TwinLiteNet+ / VehicleModel / conservative membership / existing distance API."""
import json
import math
import logging
from dataclasses import replace
from pathlib import Path
from time import perf_counter
import torch

from ai.src.lane_detection.twinlitenet import TwinLiteNetLaneModel, DEFAULT_LANE_WEIGHTS, ROOT, EgoLaneExtractor, LaneGeometry
from ai.src.vehicle_detection.model import VehicleModel
from ai.src.distance_estimation.model import DistanceEstimator
# Reuse the existing ROI translation implementation exactly; it invokes predict once.
from ai.src.distance_estimation.tests.test_distance_images import infer
from .ego_lane_classifier import EgoLaneClassifier
from ai.src.lane_detection.temporal_tracker import TemporalLaneTracker
from .adas_visualizer import draw_results
from .vehicle_lane_tracker import VehicleLaneTracker

DEFAULT_VEHICLE_WEIGHTS = ROOT / 'ai/checkpoints/vehicle_detection/26n_best.pt'
PRESET = ROOT / 'dataset/distance/results/_experiments/roi_640_0.10_robust/summary.json'


def load_camera_config(path):
    """Accept existing constructor JSON or a persisted experiment summary.

    Explicit selection asserts camera applicability; historical files are not
    automatically treated as calibration for arbitrary videos.
    """
    if path is None:
        return {}
    config = json.loads(Path(path).read_text())
    values = {k:config[k] for k in ('focal_length_px','horizontal_fov_deg','reference_resolution','vehicle_widths') if k in config}
    if values.get('focal_length_px') is None and values.get('horizontal_fov_deg') is None:
        raise ValueError('Camera config has no focal length or horizontal FOV')
    if values.get('focal_length_px') is not None and not values.get('reference_resolution'):
        raise ValueError('Pixel focal length requires reference_resolution')
    # ADASPipeline constructs and validates the sole estimator before networks.
    return values


class ADASPipeline:
    def __init__(self, lane_weights=DEFAULT_LANE_WEIGHTS, vehicle_weights=DEFAULT_VEHICLE_WEIGHTS,
                 vehicle_conf=None, lane_threshold=.5, device='auto', smooth=False,
                 point_radius=4, lane_margin=8., distance_method='roi_640_0.10_robust',
                 camera_config=None, geometry=None, full_frame_vehicles=False, lane_model=None, vehicle_model=None,
                 distance_estimator=None, lane_smoothing_alpha=.25, lane_max_hold_seconds=.5,
                 lane_confidence_threshold=None, lane_tracking_enabled=True, lane_debug=False,
                 lane_compare_baseline=False, lane_classification_max_age=None,
                 lane_detection_interval=5, video_fps=30., vehicle_tracking_enabled=True,
                 inside_confirm_frames=3, outside_confirm_frames=6,
                 lane_inside_margin=.10, lane_outside_margin=.05,
                 max_uncertain_hold_seconds=.3, vehicle_track_timeout_seconds=.5,
                 distance_median_window=3, distance_ema_alpha=.35, distance_max_age_seconds=.5,
                 debug_lane_classification=False):
        if (not isinstance(lane_detection_interval, int) or lane_detection_interval < 1
                or not math.isfinite(video_fps) or video_fps <= 0):
            raise ValueError('Lane interval must be a positive integer and video FPS positive')
        if lane_detection_interval > 1 and not lane_tracking_enabled:
            raise ValueError('Detection intervals greater than one require lane tracking')
        if lane_tracking_enabled and lane_detection_interval > 1 and lane_max_hold_seconds < lane_detection_interval/video_fps:
            raise ValueError('Lane hold time must cover the detection interval / video FPS')
        self.lane_detection_interval = lane_detection_interval
        self.video_fps = video_fps
        if lane_classification_max_age is None:
            lane_classification_max_age = min(lane_max_hold_seconds, lane_detection_interval/video_fps)
        if distance_method != 'roi_640_0.10_robust':
            raise ValueError('Supported distance experiment preset: roi_640_0.10_robust')
        preset = json.loads(PRESET.read_text())
        self.roi = None if full_frame_vehicles else preset['roi']
        if preset['selection'] != 'robust' or preset['imgsz'] != 640:
            raise ValueError('Unexpected saved ROI experiment configuration')
        self.distance_method = distance_method
        self.distance_estimator = distance_estimator if distance_estimator is not None else DistanceEstimator(**load_camera_config(camera_config))
        self.camera_config = str(camera_config) if camera_config else None
        confidence = preset['confidence_threshold'] if vehicle_conf is None else vehicle_conf
        if not 0 < confidence <= 1 or point_radius < 1:
            raise ValueError('Invalid confidence or point radius')
        if not (0 <= lane_inside_margin < .5 and 0 <= lane_outside_margin < .5):
            raise ValueError('Normalized lane margins must be in [0, .5)')
        self.vehicle_tracking_enabled = vehicle_tracking_enabled
        self.debug_lane_classification = debug_lane_classification
        self.lane_inside_margin = lane_inside_margin
        self.lane_outside_margin = lane_outside_margin
        self.vehicle_tracker = VehicleLaneTracker(inside_confirm_frames,outside_confirm_frames,
                max_uncertain_hold_seconds,vehicle_track_timeout_seconds,
                distance_median_window,distance_ema_alpha,distance_max_age_seconds,video_fps)
        self.classifier = EgoLaneClassifier(lane_margin,smooth)
        self.lane_tracking_enabled = lane_tracking_enabled
        self.lane_debug = lane_debug
        self.lane_tracker = TemporalLaneTracker(alpha=lane_smoothing_alpha,
                            max_hold_seconds=lane_max_hold_seconds, acquisition_hits=1,
                            classification_max_age=lane_classification_max_age,
                            camera_center=(geometry or LaneGeometry()).camera_center)
        # Short dashed observations can support a tracked line; retain legacy
        # extraction defaults for all previous pipelines and the comparison.
        tracking_geometry = replace(geometry or LaneGeometry(), min_span=.08, min_points=6)
        self.tracking_extractor = EgoLaneExtractor(tracking_geometry, validate_pair=False)
        self.baseline_extractor = EgoLaneExtractor(geometry, smooth=smooth) if lane_compare_baseline else None
        self.point_radius = point_radius
        self.lane = lane_model
        self.vehicle = vehicle_model
        if self.lane is None:
            threshold = lane_threshold if lane_confidence_threshold is None else lane_confidence_threshold
            self.lane = TwinLiteNetLaneModel(lane_weights,device,threshold,
                                            smooth and not lane_tracking_enabled,geometry)
        if self.vehicle is None:
            if not Path(vehicle_weights).is_file():
                raise FileNotFoundError(f'Vehicle weights missing: {vehicle_weights}')
            self.vehicle = VehicleModel(str(vehicle_weights),img_size=preset['imgsz'],confidence_threshold=confidence)
            self.vehicle.model.to(str(self.lane.device))
        self.reset()

    def reset(self):
        self.classifier.reset()
        self.vehicle_tracker.reset()
        self.lane_tracker.reset()
        self.tracking_extractor.reset()
        if self.baseline_extractor is not None:
            self.baseline_extractor.reset()
        if hasattr(self.lane,'extractor'):
            self.lane.extractor.reset()
        self.frame_index = 0
        self.lane_inference_calls = 0
        self.distance_calls = 0
        self.timings = {}
        self.resolution = None

    def sync(self):
        if hasattr(self.lane,'sync'):
            self.lane.sync()

    @torch.inference_mode()
    def process_frame(self, frame, timestamp=None):
        height,width = frame.shape[:2]
        if self.resolution is not None and self.resolution != (width,height):
            self.reset()
        self.resolution = (width,height)
        self.sync();start = perf_counter()
        detection_ran = self.frame_index % self.lane_detection_interval == 0
        raw_lanes = {side: [] for side in self.lane_tracker.SIDES}
        mask = None
        if detection_ran:
            mask,raw_lanes = self.lane.predict(frame)
            self.lane_inference_calls += 1
        self.frame_index += 1
        self.sync();inferred_done = perf_counter()
        timestamp = (self.frame_index-1)/self.video_fps if timestamp is None else timestamp
        baseline = self.baseline_extractor.extract(mask) if self.baseline_extractor is not None and detection_ran else raw_lanes
        diagnostics = getattr(getattr(self.lane,'extractor',None),'diagnostics',{})
        diagnostics = dict(diagnostics) if detection_ran and isinstance(diagnostics,dict) else {}
        tracking = None
        if self.lane_tracking_enabled:
            observations = self.tracking_extractor.extract(mask) if detection_ran else raw_lanes
            if detection_ran:
                diagnostics = dict(self.tracking_extractor.diagnostics)
            lanes,classification_lanes,tracking = self.lane_tracker.update(observations,(width,height),timestamp,frame,detection_ran=detection_ran)
            if tracking['reset_reason']:
                self.classifier.reset()
                self.vehicle_tracker.reset()
        else:
            observations = raw_lanes
            lanes = classification_lanes = raw_lanes
        lane_done = perf_counter()
        if self.lane_debug:
            logging.getLogger(__name__).info('lane %s',json.dumps(dict(timestamp=timestamp,
                diagnostics=diagnostics,tracking=tracking)))
        detections = infer(self.vehicle,frame,self.roi)
        self.sync();vehicle_done = perf_counter()
        if self.vehicle_tracking_enabled:
            tracked_vehicles = self.vehicle_tracker.associate(detections,timestamp)
            self.classifier.prepare(classification_lanes,(width,height))
            vehicles = []
            for item in tracked_vehicles:
                observation = self.classifier.observe(item.get('bbox'),(width,height),
                        self.lane_inside_margin,self.lane_outside_margin,item.get('crop_clipped',False))
                if tracking and any(v.get('timed_out') for v in tracking['sides'].values()):
                    observation.update(hard_invalid=True,lane_reason='lane_track_expired')
                vehicles.append(self.vehicle_tracker.classify(item,observation,timestamp))
        else:
            vehicles = self.classifier.classify_all(detections,classification_lanes,(width,height))
        for vehicle in vehicles:
            if vehicle.get('crop_clipped'):
                vehicle['lane_status'] = 'unknown'
        classified = perf_counter()
        calls = 0
        calibrated = self.distance_estimator.effective_focal_length((width,height)) is not None
        for vehicle in vehicles:
            if vehicle['lane_status'] != 'inside' or (self.vehicle_tracking_enabled and not vehicle['distance_eligible']):
                if self.vehicle_tracking_enabled:
                    self.vehicle_tracker.smooth_distance(vehicle,None,timestamp)
                continue
            calls += 1
            try:
                # A truncated ROI detection cannot provide a full physical width.
                value = self.distance_estimator.estimate(vehicle['bbox'],(width,height),vehicle['class_name'])
                if vehicle.get('crop_clipped'):
                    value = None
                if value is not None and math.isfinite(value) and value > 0:
                    if self.vehicle_tracking_enabled:
                        vehicle['raw_distance'] = float(value)
                        vehicle['distance'] = self.vehicle_tracker.smooth_distance(vehicle,value,timestamp)
                        vehicle['distance_fresh'] = vehicle['distance'] is not None
                        vehicle['distance_for_alerts'] = vehicle['distance_fresh']
                    else:
                        vehicle['distance'] = float(value)
                elif self.vehicle_tracking_enabled:
                    self.vehicle_tracker.smooth_distance(vehicle,value,timestamp)
            except (ValueError,TypeError,RuntimeError,ArithmeticError):
                vehicle['distance'] = None
                if self.vehicle_tracking_enabled:
                    self.vehicle_tracker.smooth_distance(vehicle,None,timestamp)
        distance_done = perf_counter()
        if self.vehicle_tracking_enabled:
            for vehicle in vehicles:
                track = self.vehicle_tracker.tracks.get(vehicle.get('track_id'))
                vehicle['distance_rejection'] = track.distance.rejection if track else 'no_track'
        if self.debug_lane_classification:
            logging.getLogger(__name__).info('vehicle_lane %s',json.dumps(dict(
                timestamp=timestamp,lane_detection_ran=detection_ran,lane_tracking=tracking,
                geometry=self.classifier.diagnostics,vehicles=vehicles)))
        result = dict(lanes=lanes,raw_lanes=observations,baseline_lanes=baseline,
                      classification_lanes=classification_lanes,lane_tracking=tracking,
                      lane_diagnostics=diagnostics, classification_diagnostics=dict(self.classifier.diagnostics), timestamp_seconds=timestamp, lane_detection_ran=detection_ran,
                      vehicles=vehicles,calibrated=calibrated,
                      distance_calls=calls,processing_fps=1/max(distance_done-start,1e-9))
        output = self.draw_results(frame,result)
        end = perf_counter()
        self.timings = dict(lane_ms=(lane_done-start)*1000,lane_model_ms=(inferred_done-start)*1000 if detection_ran else 0.,
                            lane_tracking_ms=(lane_done-inferred_done)*1000,vehicle_ms=(vehicle_done-lane_done)*1000,
                            membership_ms=(classified-vehicle_done)*1000,distance_ms=(distance_done-classified)*1000,
                            visualization_ms=(end-distance_done)*1000,pipeline_ms=(end-start)*1000)
        result['processing_fps'] = 1000/self.timings['pipeline_ms']
        result['timings'] = dict(self.timings)
        self.distance_calls += calls
        return output,result

    def draw_results(self,frame,results):
        return draw_results(frame,results,self.point_radius)
