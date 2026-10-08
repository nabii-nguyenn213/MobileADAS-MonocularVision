import argparse
import os
import time

import cv2

from ai.src.lane_detection.config import LaneDetectionConfig
from ai.src.lane_detection.detector import LaneDetector
from ai.src.lane_detection.processor import (
    LanePreprocessor,
    LanePostprocessor,
    CurveLanesPostprocessor,
)
from ai.src.lane_detection.models.ufldv2 import UFLDv2
from ai.src.lane_detection.models.ufldv2_curvelanes import UFLDv2CurveLanes


class HybridModelSelector:

    def __init__(
        self,
        switch_margin: float = 0.08,
        required_frames: int = 3,
    ):
        self.switch_margin = switch_margin
        self.required_frames = required_frames

        self.current_model = None
        self.candidate_model = None
        self.candidate_count = 0

    @staticmethod
    def _is_valid(ego_lane) -> bool:
        if ego_lane is None:
            return False

        left = ego_lane.get("left")
        right = ego_lane.get("right")

        return (
            left is not None
            and right is not None
            and len(left) >= 2
            and len(right) >= 2
        )

    def select(
        self,
        tusimple_ego,
        curvelanes_ego,
        tusimple_score: float,
        curvelanes_score: float,
    ) -> str:

        tusimple_valid = self._is_valid(tusimple_ego)
        curvelanes_valid = self._is_valid(curvelanes_ego)

        if tusimple_valid and not curvelanes_valid:
            self.current_model = "tusimple"
            self.candidate_model = None
            self.candidate_count = 0
            return self.current_model

        if curvelanes_valid and not tusimple_valid:
            self.current_model = "curvelanes"
            self.candidate_model = None
            self.candidate_count = 0
            return self.current_model

        if not tusimple_valid and not curvelanes_valid:
            return self.current_model or "tusimple"

        scores = {
            "tusimple": float(tusimple_score),
            "curvelanes": float(curvelanes_score),
        }

        if self.current_model is None:
            self.current_model = max(
                scores,
                key=scores.get,
            )
            return self.current_model

        other_model = (
            "curvelanes"
            if self.current_model == "tusimple"
            else "tusimple"
        )

        current_score = scores[self.current_model]
        other_score = scores[other_model]

        if (
            other_score
            > current_score + self.switch_margin
        ):
            if self.candidate_model == other_model:
                self.candidate_count += 1
            else:
                self.candidate_model = other_model
                self.candidate_count = 1

            if self.candidate_count >= self.required_frames:
                self.current_model = other_model
                self.candidate_model = None
                self.candidate_count = 0
        else:
            self.candidate_model = None
            self.candidate_count = 0

        return self.current_model

    def reset(self):
        self.current_model = None
        self.candidate_model = None
        self.candidate_count = 0


class HybridLaneTest:

    TUSIMPLE = {
        "backbone": "resnet34",
        "checkpoint": (
            "ai/checkpoints/lane_detection/"
            "tusimple_res34.pth"
        ),
        "num_grid_row": 100,
        "num_grid_col": 100,
        "num_row": 56,
        "num_col": 41,
        "num_lanes": 4,
        "input_width": 800,
        "input_height": 320,
        "crop_ratio": 0.8,
        "expected_points": 56,
    }

    CURVELANES = {
        "backbone": "resnet34",
        "checkpoint": (
            "ai/checkpoints/lane_detection/"
            "curvelanes_res34.pth"
        ),
        "num_grid_row": 200,
        "num_grid_col": 100,
        "num_row": 72,
        "num_col": 81,
        "num_lanes": 10,
        "input_width": 1600,
        "input_height": 800,
        "crop_ratio": 0.8,
        "expected_points": 72,
    }

    def __init__(
        self,
        tusimple_threshold: float = 0.60,
        curvelanes_threshold: float = 0.60,
        switch_margin: float = 0.08,
        required_frames: int = 3,
        point_radius: int = 5,
    ):
        self.tusimple_detector = self._build_tusimple_detector(
            confidence_threshold=tusimple_threshold
        )

        self.curvelanes_detector = self._build_curvelanes_detector(
            confidence_threshold=curvelanes_threshold
        )

        self.selector = HybridModelSelector(
            switch_margin=switch_margin,
            required_frames=required_frames,
        )

        self.point_radius = point_radius

        print("=" * 70)
        print("Hybrid lane test")
        print("TuSimple model:   ResNet34")
        print("CurveLanes model: ResNet34")
        print(f"TuSimple threshold:   {tusimple_threshold:.2f}")
        print(f"CurveLanes threshold: {curvelanes_threshold:.2f}")
        print(f"Switch margin:        {switch_margin:.2f}")
        print(f"Required frames:      {required_frames}")
        print(f"Point radius:         {point_radius}")
        print("=" * 70)

    def _build_tusimple_detector(
        self,
        confidence_threshold: float,
    ):
        cfg = self.TUSIMPLE

        config = LaneDetectionConfig()
        config.input_width = cfg["input_width"]
        config.input_height = cfg["input_height"]
        config.checkpoint_path = cfg["checkpoint"]

        model = UFLDv2(
            backbone=cfg["backbone"],
            num_grid_row=cfg["num_grid_row"],
            num_grid_col=cfg["num_grid_col"],
            num_row=cfg["num_row"],
            num_col=cfg["num_col"],
            num_lanes=cfg["num_lanes"],
            input_height=cfg["input_height"],
            input_width=cfg["input_width"],
        )

        detector = LaneDetector(
            model=model,
            config=config,
        )

        detector.preprocessor = LanePreprocessor(
            input_width=cfg["input_width"],
            input_height=cfg["input_height"],
            crop_ratio=cfg["crop_ratio"],
        )

        detector.postprocessor = LanePostprocessor(
            num_row=cfg["num_row"],
            num_col=cfg["num_col"],
            confidence_threshold=confidence_threshold,
        )

        print("Loading TuSimple ResNet34...")
        detector.load_weights()
        print("TuSimple checkpoint loaded.")

        return detector

    def _build_curvelanes_detector(
        self,
        confidence_threshold: float,
    ):
        cfg = self.CURVELANES

        config = LaneDetectionConfig()
        config.input_width = cfg["input_width"]
        config.input_height = cfg["input_height"]
        config.checkpoint_path = cfg["checkpoint"]

        model = UFLDv2CurveLanes(
            backbone=cfg["backbone"],
            num_grid_row=cfg["num_grid_row"],
            num_grid_col=cfg["num_grid_col"],
            num_row=cfg["num_row"],
            num_col=cfg["num_col"],
            num_lanes=cfg["num_lanes"],
            input_height=cfg["input_height"],
            input_width=cfg["input_width"],
        )

        detector = LaneDetector(
            model=model,
            config=config,
        )

        detector.preprocessor = LanePreprocessor(
            input_width=cfg["input_width"],
            input_height=cfg["input_height"],
            crop_ratio=cfg["crop_ratio"],
        )

        try:
            detector.postprocessor = CurveLanesPostprocessor(
                num_row=cfg["num_row"],
                num_col=cfg["num_col"],
                num_lanes=cfg["num_lanes"],
                confidence_threshold=confidence_threshold,
            )
        except TypeError:
            detector.postprocessor = CurveLanesPostprocessor(
                num_row=cfg["num_row"],
                num_col=cfg["num_col"],
                num_lanes=cfg["num_lanes"],
            )

        print("Loading CurveLanes ResNet34...")
        detector.load_weights()
        print("CurveLanes checkpoint loaded.")

        return detector

    @staticmethod
    def crop_to_16_9(frame):
        height, width = frame.shape[:2]
        current_ratio = width / height
        target_ratio = 16 / 9

        if current_ratio < target_ratio:
            target_height = int(width / target_ratio)
            start_y = height - target_height
            frame = frame[start_y:, :]

        elif current_ratio > target_ratio:
            target_width = int(height * target_ratio)
            start_x = (width - target_width) // 2
            frame = frame[
                :,
                start_x:start_x + target_width,
            ]

        return frame

    @staticmethod
    def _clamp_score(score) -> float:
        return max(
            0.0,
            min(
                1.0,
                float(score),
            ),
        )

    def _ego_score(
        self,
        ego_lane,
        expected_points: int,
    ) -> float:
        """
        Preferred:
            Use ego_lane["confidence"] when available.

        Fallback:
            Use left/right boundary coverage as a temporary
            proxy so the test can run before confidence
            propagation exists for both postprocessors.
        """
        if ego_lane is None:
            return 0.0

        confidence = ego_lane.get("confidence")

        if confidence is not None:
            return self._clamp_score(confidence)

        left_confidence = ego_lane.get("left_confidence")
        right_confidence = ego_lane.get("right_confidence")

        available = [
            value
            for value in [
                left_confidence,
                right_confidence,
            ]
            if value is not None
        ]

        if available:
            return self._clamp_score(
                sum(available)
                / len(available)
            )

        left = ego_lane.get("left")
        right = ego_lane.get("right")

        left_score = 0.0
        right_score = 0.0

        if left is not None:
            left_score = min(
                len(left) / expected_points,
                1.0,
            )

        if right is not None:
            right_score = min(
                len(right) / expected_points,
                1.0,
            )

        return self._clamp_score(
            0.5 * (
                left_score
                + right_score
            )
        )

    def _predict_both(
        self,
        frame,
    ):
        tusimple_ego = (
            self.tusimple_detector
            .predict_ego_lane(frame)
        )

        curvelanes_ego = (
            self.curvelanes_detector
            .predict_ego_lane(frame)
        )

        tusimple_score = self._ego_score(
            tusimple_ego,
            self.TUSIMPLE["expected_points"],
        )

        curvelanes_score = self._ego_score(
            curvelanes_ego,
            self.CURVELANES["expected_points"],
        )

        selected_model = self.selector.select(
            tusimple_ego=tusimple_ego,
            curvelanes_ego=curvelanes_ego,
            tusimple_score=tusimple_score,
            curvelanes_score=curvelanes_score,
        )

        if selected_model == "curvelanes":
            selected_ego = curvelanes_ego
        else:
            selected_ego = tusimple_ego

        return {
            "model": selected_model,
            "ego_lane": selected_ego,
            "tusimple_score": tusimple_score,
            "curvelanes_score": curvelanes_score,
        }

    def _draw_lane_points(
        self,
        image,
        lane,
        color,
    ):
        if lane is None:
            return

        for point in lane:
            if point is None or len(point) < 2:
                continue

            x, y = point

            cv2.circle(
                image,
                (int(x), int(y)),
                self.point_radius,
                color,
                -1,
                cv2.LINE_AA,
            )

    def _visualize(
        self,
        frame,
        result,
    ):
        # Points only: no lines, no triangle, no fill.
        output = frame.copy()

        ego_lane = result["ego_lane"]

        if ego_lane is not None:
            left_lane = ego_lane.get("left")
            right_lane = ego_lane.get("right")

            # Left boundary: green
            self._draw_lane_points(
                image=output,
                lane=left_lane,
                color=(0, 255, 0),
            )

            # Right boundary: red
            self._draw_lane_points(
                image=output,
                lane=right_lane,
                color=(0, 0, 255),
            )

        cv2.putText(
            output,
            f"Selected: {result['model']}",
            (30, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            output,
            (
                f"TuSimple: "
                f"{result['tusimple_score']:.3f}"
            ),
            (30, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            output,
            (
                f"CurveLanes: "
                f"{result['curvelanes_score']:.3f}"
            ),
            (30, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return output

    def reset(self):
        self.selector.reset()

    def test_image(
        self,
        image_path: str,
        output_path: str,
        crop_16_9: bool = True,
    ):
        self.reset()

        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(
                f"Could not load image: {image_path}"
            )

        if crop_16_9:
            image = self.crop_to_16_9(image)

        result = self._predict_both(image)
        visualization = self._visualize(
            image,
            result,
        )

        output_dir = os.path.dirname(output_path)

        if output_dir:
            os.makedirs(
                output_dir,
                exist_ok=True,
            )

        if not cv2.imwrite(
            output_path,
            visualization,
        ):
            raise RuntimeError(
                f"Could not save image: {output_path}"
            )

        print(
            f"TuSimple score:   "
            f"{result['tusimple_score']:.4f}"
        )
        print(
            f"CurveLanes score: "
            f"{result['curvelanes_score']:.4f}"
        )
        print(
            f"Selected model:   "
            f"{result['model']}"
        )
        print(
            f"Saved to: "
            f"{output_path}"
        )

    def test_video(
        self,
        video_path: str,
        output_path: str,
        crop_16_9: bool = True,
        progress_interval: int = 30,
    ):
        self.reset()

        capture = cv2.VideoCapture(video_path)

        if not capture.isOpened():
            raise FileNotFoundError(
                f"Could not open video: {video_path}"
            )

        input_fps = capture.get(
            cv2.CAP_PROP_FPS
        )

        if input_fps <= 0:
            input_fps = 30.0

        total_frames = int(
            capture.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        output_dir = os.path.dirname(output_path)

        if output_dir:
            os.makedirs(
                output_dir,
                exist_ok=True,
            )

        writer = None
        frame_index = 0
        start_time = time.perf_counter()

        selected_counts = {
            "tusimple": 0,
            "curvelanes": 0,
        }

        try:
            while True:
                success, frame = capture.read()

                if not success:
                    break

                if crop_16_9:
                    frame = self.crop_to_16_9(frame)

                result = self._predict_both(frame)

                visualization = self._visualize(
                    frame,
                    result,
                )

                selected_counts[result["model"]] += 1

                if writer is None:
                    (
                        output_height,
                        output_width,
                    ) = visualization.shape[:2]

                    writer = cv2.VideoWriter(
                        output_path,
                        cv2.VideoWriter_fourcc(
                            *"mp4v"
                        ),
                        input_fps,
                        (
                            output_width,
                            output_height,
                        ),
                    )

                    if not writer.isOpened():
                        raise RuntimeError(
                            f"Could not create output video: "
                            f"{output_path}"
                        )

                writer.write(visualization)
                frame_index += 1

                if (
                    progress_interval > 0
                    and frame_index % progress_interval == 0
                ):
                    elapsed = (
                        time.perf_counter()
                        - start_time
                    )

                    processing_fps = (
                        frame_index / elapsed
                    )

                    print(
                        f"Frame "
                        f"{frame_index}/"
                        f"{total_frames} | "
                        f"TuSimple="
                        f"{result['tusimple_score']:.3f} | "
                        f"Curve="
                        f"{result['curvelanes_score']:.3f} | "
                        f"Selected="
                        f"{result['model']} | "
                        f"FPS="
                        f"{processing_fps:.2f}"
                    )

        finally:
            capture.release()

            if writer is not None:
                writer.release()

        elapsed = (
            time.perf_counter()
            - start_time
        )

        average_fps = (
            frame_index / elapsed
            if elapsed > 0
            else 0.0
        )

        print()
        print(f"Processed frames: {frame_index}")
        print(f"Average processing FPS: {average_fps:.2f}")
        print(
            f"TuSimple selected: "
            f"{selected_counts['tusimple']} frames"
        )
        print(
            f"CurveLanes selected: "
            f"{selected_counts['curvelanes']} frames"
        )
        print(f"Saved to: {output_path}")


def build_parser():

    parser = argparse.ArgumentParser(
        description=(
            "Run TuSimple ResNet34 and "
            "CurveLanes ResNet34 together."
        )
    )

    parser.add_argument(
        "--mode",
        choices=[
            "image",
            "video",
        ],
        required=True,
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--tusimple-threshold",
        type=float,
        default=0.60,
    )

    parser.add_argument(
        "--curvelanes-threshold",
        type=float,
        default=0.60,
    )

    parser.add_argument(
        "--switch-margin",
        type=float,
        default=0.08,
    )

    parser.add_argument(
        "--required-frames",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--point-radius",
        type=int,
        default=5,
        help="Radius of each visualized lane point.",
    )

    parser.add_argument(
        "--progress-interval",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--no-crop",
        action="store_true",
    )

    return parser


def main():

    parser = build_parser()
    args = parser.parse_args()

    if not 0.0 <= args.tusimple_threshold <= 1.0:
        parser.error(
            "--tusimple-threshold must be between 0 and 1."
        )

    if not 0.0 <= args.curvelanes_threshold <= 1.0:
        parser.error(
            "--curvelanes-threshold must be between 0 and 1."
        )

    if args.switch_margin < 0:
        parser.error(
            "--switch-margin cannot be negative."
        )

    if args.required_frames < 1:
        parser.error(
            "--required-frames must be at least 1."
        )

    if args.point_radius < 1:
        parser.error(
            "--point-radius must be at least 1."
        )

    tester = HybridLaneTest(
        tusimple_threshold=(
            args.tusimple_threshold
        ),
        curvelanes_threshold=(
            args.curvelanes_threshold
        ),
        switch_margin=(
            args.switch_margin
        ),
        required_frames=(
            args.required_frames
        ),
        point_radius=(
            args.point_radius
        ),
    )

    output_path = args.output

    if output_path is None:
        extension = (
            "png"
            if args.mode == "image"
            else "mp4"
        )

        output_path = (
            "ai/outputs/lane_detector/"
            f"hybrid_lane_result.{extension}"
        )

    if args.mode == "image":
        tester.test_image(
            image_path=args.input,
            output_path=output_path,
            crop_16_9=(
                not args.no_crop
            ),
        )

    else:
        tester.test_video(
            video_path=args.input,
            output_path=output_path,
            crop_16_9=(
                not args.no_crop
            ),
            progress_interval=(
                args.progress_interval
            ),
        )


if __name__ == "__main__":
    main()
