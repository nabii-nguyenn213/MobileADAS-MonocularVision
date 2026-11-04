"""Fusion drawing, device adapter and video lifecycle regression tests."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

import cv2
import numpy as np

from ai.src.fusion.ufldv2_vehicle import UFLDv2VehicleFusion, LANE_COLORS
from ai.src.fusion.tests.test_ufldv2_vehicle import process_video
from ai.src.lane_detection.ufldv2 import LaneDetection


class TestFusion(unittest.TestCase):
    def pipeline(self):
        lane_model, vehicle_model = Mock(), Mock()
        lane_model.predict.return_value = [
            LaneDetection(0, "left-side", [(10, y) for y in (60, 80, 100)]),
            LaneDetection(1, "left-ego", [(40, y) for y in (60, 80, 100)]),
            LaneDetection(2, "right-ego", [(120, y) for y in (60, 80, 100)]),
            LaneDetection(3, "right-side", []),
        ]
        vehicle_model.predict.return_value = [dict(
            class_id=0, class_name="car", confidence=0.87, bbox=[60, 40, 100, 65]
        )]
        with patch("ai.src.fusion.ufldv2_vehicle.UFLDv2", return_value=lane_model), patch(
            "ai.src.fusion.ufldv2_vehicle.VehicleModel", return_value=vehicle_model
        ) as factory, tempfile.NamedTemporaryFile() as weights:
            fusion = UFLDv2VehicleFusion(vehicle_weights=weights.name, device="cpu")
            factory.assert_called_once_with(model_path=weights.name, confidence_threshold=0.25)
        vehicle_model.model.to.assert_called_once_with("cpu")
        return fusion

    def test_points_and_existing_vehicle_visualizer(self):
        pipeline = self.pipeline()
        frame = np.zeros((120, 160, 3), np.uint8)
        output = pipeline.predict(frame)
        self.assertTrue(np.all(frame == 0))
        self.assertEqual(tuple(output[80, 40]), LANE_COLORS[1])
        self.assertEqual(tuple(output[80, 120]), LANE_COLORS[2])
        self.assertEqual(tuple(output[80, 10]), (0, 0, 0))  # Ego boundaries preferred.
        self.assertEqual(tuple(output[70, 40]), (0, 0, 0))  # No connecting lines.
        self.assertEqual(tuple(output[60, 60]), (0, 255, 0))  # Vehicle box.
        self.assertEqual(pipeline.last_vehicles[0]["confidence"], 0.87)

    def test_empty_predictions(self):
        pipeline = self.pipeline()
        pipeline.lane_model.predict.return_value = []
        pipeline.vehicle_model.predict.return_value = []
        frame = np.zeros((120, 160, 3), np.uint8)
        np.testing.assert_array_equal(pipeline.predict(frame), frame)

    def test_video_metadata_and_missing_input(self):
        pipeline = self.pipeline()
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "input.mp4", Path(directory) / "output.mp4"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 12, (160, 120))
            self.assertTrue(writer.isOpened())
            for _ in range(3):
                writer.write(np.zeros((120, 160, 3), np.uint8))
            writer.release()
            stats = process_video(source, target, pipeline)
            self.assertEqual(stats["both_frames"], 3)
            video = cv2.VideoCapture(str(target))
            try:
                self.assertEqual(video.get(cv2.CAP_PROP_FRAME_COUNT), 3)
                self.assertEqual(video.get(cv2.CAP_PROP_FPS), 12)
                self.assertEqual(video.read()[1].shape, (120, 160, 3))
            finally:
                video.release()
            with self.assertRaisesRegex(ValueError, "must differ"):
                process_video(source, source, pipeline)
            with self.assertRaisesRegex(FileNotFoundError, "Input video not found"):
                process_video(Path(directory) / "missing.mp4", target, pipeline)

    def test_cleanup_on_inference_error(self):
        capture, writer = Mock(), Mock()
        capture.get.return_value = 30
        capture.read.return_value = (True, np.zeros((120, 160, 3), np.uint8))
        pipeline = Mock()
        pipeline.predict.side_effect = RuntimeError("Inference failed")
        with tempfile.NamedTemporaryFile() as source, tempfile.TemporaryDirectory() as directory:
            with patch("cv2.VideoCapture", return_value=capture), patch("cv2.VideoWriter", return_value=writer):
                with self.assertRaisesRegex(RuntimeError, "Inference failed"):
                    process_video(source.name, Path(directory) / "out.mp4", pipeline)
        capture.release.assert_called_once()
        writer.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
