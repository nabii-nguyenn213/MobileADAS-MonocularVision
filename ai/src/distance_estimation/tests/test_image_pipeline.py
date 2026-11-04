"""Regression checks for label independence, ROI geometry and explicit config."""
import contextlib
import io
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from ai.src.distance_estimation.model import DistanceEstimator
from ai.src.distance_estimation.tests import test_distance_images as script


class ImagePipelineTests(unittest.TestCase):
    def test_missing_camera_fails_before_loading_yolo(self):
        with patch('sys.argv', ['test_distance_images']), patch.object(script, 'VehicleModel') as detector:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                script.main()
            self.assertEqual(raised.exception.code, 2)
            detector.assert_not_called()

    def test_labels_do_not_change_predictions(self):
        class Detector:
            calls = 0

            def predict(self, frame):
                self.calls += 1
                return [dict(class_name='car', class_id=0, confidence=.8,
                             bbox=[40, 30, 60, 50])]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'images'
            source.mkdir()
            detector = Detector()
            estimator = DistanceEstimator(focal_length_px=100,
                                          reference_resolution=(100, 80), vehicle_widths={'car': 2})
            for name in ('5m.jpeg', '50m.jpeg'):
                cv2.imwrite(str(source / name), np.zeros((80, 100, 3), dtype=np.uint8))
            args = Namespace(input_dir=source, output_dir=root / 'out', target_points=None,
                             target_overrides=None, imgsz=640, confidence=.25, roi=None,
                             selection_method='robust', debug_detections=False,
                             weights=Path('unused.pt'), camera_config=None)
            with contextlib.redirect_stdout(io.StringIO()):
                rows = script.run(args, detector, estimator)
            self.assertEqual(detector.calls, 2)
            self.assertEqual([r['predicted_m'] for r in rows], [10, 10])
            self.assertTrue(all(r['metric_exclusion'] == 'conflicting_duplicate_labels' for r in rows))
            self.assertTrue((args.output_dir / 'predictions.csv').exists())

    def test_roi_maps_to_original_geometry(self):
        class Detector:
            def predict(self, frame):
                self.shape = frame.shape
                return [dict(class_name='car', class_id=0, confidence=.8, bbox=[10, 10, 30, 30])]

        detector = Detector()
        detections = script.infer(detector, np.zeros((100, 200, 3), dtype=np.uint8), [.25, .2, .75, .8])
        self.assertEqual(detector.shape[:2], (60, 100))
        self.assertEqual(detections[0]['bbox'], [60, 30, 80, 50])
        estimator = DistanceEstimator(focal_length_px=200, reference_resolution=(200, 100), vehicle_widths={'car': 2})
        self.assertEqual(estimator.estimate(detections[0]['bbox'], (200, 100), 'car'), 20)

    def test_selection_accepts_offset_target_and_reports_ambiguity(self):
        def candidate(box):
            return dict(class_name='car', class_id=0, confidence=.8, bbox=box)
        chosen, _ = script.select_target([candidate([45, 35, 55, 45])], (100, 100), [.5,.5])
        self.assertIsNotNone(chosen)
        chosen, status = script.select_target([candidate([40,35,50,45]), candidate([50,35,60,45])], (100,100), [.5,.5])
        self.assertIsNone(chosen)
        self.assertEqual(status, 'ambiguous')


if __name__ == '__main__':
    unittest.main()
