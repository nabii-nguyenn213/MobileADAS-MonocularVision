import unittest
from unittest.mock import patch, Mock

import numpy as np

from ai.src.fusion.model import LaneVehicleFusion
from ai.src.distance_estimation import DistanceEstimator


class IntegrationTests(unittest.TestCase):
    @patch('ai.src.fusion.model.HybridLaneTest')
    @patch('ai.src.fusion.model.VehicleModel', autospec=True)
    def test_reuses_detector(self, vehicle_class, lane_class):
        detections = [{'bbox': [100, 100, 200, 200], 'class_name': 'car',
                       'class_id': 0, 'confidence': .9}]
        vehicle_class.return_value.predict.return_value = detections
        lane_class.return_value._predict_both.return_value = {
            'model': 'tusimple', 'ego_lane': None,
            'tusimple_score': .7, 'curvelanes_score': .6}
        lane_class.return_value._visualize.side_effect = lambda frame, result: frame.copy()
        fusion = LaneVehicleFusion(focal_length_px=1000, crop_16_9=False)
        fusion.distance_estimator.estimate = Mock(wraps=fusion.distance_estimator.estimate)
        output, result = fusion.process_frame(np.zeros((600, 1000, 3), dtype=np.uint8))
        vehicle_class.assert_called_once()
        vehicle_class.return_value.predict.assert_called_once()
        fusion.distance_estimator.estimate.assert_called_once_with(
            detections[0]['bbox'], (1000, 600), 'car',
            source_resolution=(1000, 600), crop_box=(0, 0, 1000, 600))
        self.assertIs(result['vehicles'], detections)
        self.assertNotIn('distance_m', detections[0])
        self.assertEqual(result['vehicle_distances'], [{'detection_index': 0, 'distance_m': 18}])
        self.assertEqual(output.shape, (600, 1000, 3))

    @patch('ai.src.fusion.model.HybridLaneTest')
    @patch('ai.src.fusion.model.VehicleModel', autospec=True)
    def test_crop_and_disabled(self, vehicle_class, lane_class):
        lane = lane_class.return_value
        lane.crop_to_16_9.side_effect = lambda frame: frame[:, 100:900]
        lane._predict_both.return_value = {
            'model': 'curvelanes', 'ego_lane': None,
            'tusimple_score': .2, 'curvelanes_score': .8}
        lane._visualize.side_effect = lambda frame, result: frame.copy()
        vehicle_class.return_value.predict.return_value = [
            {'bbox': [100, 100, 200, 200], 'class_name': 'car', 'confidence': .9}]
        fusion = LaneVehicleFusion(horizontal_fov_deg=90)
        frame = np.zeros((450, 1000, 3), dtype=np.uint8)
        _, result = fusion.process_frame(frame)
        # Full-source FOV gives fx=500, not the cropped width/2=400.
        self.assertAlmostEqual(result['vehicle_distances'][0]['distance_m'], 9)
        fusion.distance_enabled = False
        fusion.distance_estimator.estimate = Mock(side_effect=AssertionError)
        _, result = fusion.process_frame(frame)
        self.assertIsNone(result['vehicle_distances'][0]['distance_m'])
        fusion.distance_estimator.estimate.assert_not_called()

    def test_distance_colors_and_toggle(self):
        from ai.src.fusion.lane_vehicle import draw_lane_vehicles
        frame = np.zeros((300, 400, 3), dtype=np.uint8)
        detections = [{'bbox': [100, 100, 200, 200], 'class_name': 'car', 'confidence': .9}]
        for distance, color in [(49, (0, 0, 255)), (50, (0, 255, 0)),
                                (None, (150, 150, 150))]:
            output, _, _ = draw_lane_vehicles(
                frame, detections, None, None, show_distance=True,
                distance_estimates=[{'distance_m': distance}])
            self.assertEqual(tuple(output[150, 100]), color)
        output, _, _ = draw_lane_vehicles(frame, detections, None, None, show_distance=False)
        self.assertEqual(tuple(output[150, 100]), (0, 255, 255))
