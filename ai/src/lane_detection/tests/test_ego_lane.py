import argparse
import os
import time

import cv2

from ai.src.lane_detection.config import (
    LaneDetectionConfig,
)

from ai.src.lane_detection.detector import (
    LaneDetector,
)

from ai.src.lane_detection.models.ufldv2 import (
    UFLDv2,
)


class EgoLaneTest:

    MODEL_CONFIGS = {
        "tusimple_res18": {
            "backbone": "resnet18",

            "checkpoint": (
                "ai/checkpoints/"
                "lane_detection/"
                "tusimple_res18.pth"
            ),

            "num_grid_row": 100,
            "num_grid_col": 100,

            "num_row": 56,
            "num_col": 41,

            "num_lanes": 4,

            "input_width": 800,
            "input_height": 320,
        },

        "tusimple_res34": {
            "backbone": "resnet34",

            "checkpoint": (
                "ai/checkpoints/"
                "lane_detection/"
                "tusimple_res34.pth"
            ),

            "num_grid_row": 100,
            "num_grid_col": 100,

            "num_row": 56,
            "num_col": 41,

            "num_lanes": 4,

            "input_width": 800,
            "input_height": 320,
        },
    }

    def __init__(
        self,
        model_name: str = "tusimple_res34",
        confidence_threshold: float = 0.60,
        smoothing_alpha: float = 0.15,
        hold_frames: int = 5,
    ):
        if model_name not in self.MODEL_CONFIGS:

            raise ValueError(
                f"Unsupported model: "
                f"{model_name}"
            )

        if not (
            0.0
            <= confidence_threshold
            <= 1.0
        ):
            raise ValueError(
                "confidence_threshold must "
                "be between 0 and 1."
            )

        if not (
            0.0
            < smoothing_alpha
            <= 1.0
        ):
            raise ValueError(
                "smoothing_alpha must "
                "be greater than 0 "
                "and <= 1."
            )

        if hold_frames < 0:
            raise ValueError(
                "hold_frames cannot "
                "be negative."
            )

        self.model_name = (
            model_name
        )

        self.model_config = (
            self.MODEL_CONFIGS[
                model_name
            ]
        )

        # =================================
        # Lane configuration
        # =================================

        self.config = (
            LaneDetectionConfig()
        )

        self.config.input_width = (
            self.model_config[
                "input_width"
            ]
        )

        self.config.input_height = (
            self.model_config[
                "input_height"
            ]
        )

        self.config.checkpoint_path = (
            self.model_config[
                "checkpoint"
            ]
        )

        # =================================
        # Build UFLDv2
        # =================================

        model = UFLDv2(
            backbone=(
                self.model_config[
                    "backbone"
                ]
            ),

            num_grid_row=(
                self.model_config[
                    "num_grid_row"
                ]
            ),

            num_grid_col=(
                self.model_config[
                    "num_grid_col"
                ]
            ),

            num_row=(
                self.model_config[
                    "num_row"
                ]
            ),

            num_col=(
                self.model_config[
                    "num_col"
                ]
            ),

            num_lanes=(
                self.model_config[
                    "num_lanes"
                ]
            ),

            input_height=(
                self.model_config[
                    "input_height"
                ]
            ),

            input_width=(
                self.model_config[
                    "input_width"
                ]
            ),
        )

        # =================================
        # Detector
        # =================================

        self.detector = LaneDetector(
            model=model,
            config=self.config,

            confidence_threshold=(
                confidence_threshold
            ),
        )

        # =================================
        # Visualization settings
        # =================================

        self.detector.visualizer.smoothing_alpha = (
            smoothing_alpha
        )

        self.detector.visualizer.hold_frames = (
            hold_frames
        )

        # =================================
        # Print configuration
        # =================================

        print(
            "=" * 60
        )

        print(
            f"Model: "
            f"{self.model_name}"
        )

        print(
            "Backbone: "
            f"{self.model_config['backbone']}"
        )

        print(
            "Checkpoint: "
            f"{self.config.checkpoint_path}"
        )

        print(
            "Confidence threshold: "
            f"{confidence_threshold:.2f}"
        )

        print(
            "Smoothing alpha: "
            f"{smoothing_alpha:.2f}"
        )

        print(
            "Hold frames: "
            f"{hold_frames}"
        )

        print(
            "=" * 60
        )

        # =================================
        # Load checkpoint
        # =================================

        self.detector.load_weights()

        print(
            "Checkpoint loaded successfully."
        )

    def crop_to_16_9(
        self,
        frame,
    ):
        height, width = (
            frame.shape[:2]
        )

        current_ratio = (
            width / height
        )

        target_ratio = (
            16 / 9
        )

        # ---------------------------------
        # Too tall
        # ---------------------------------

        if current_ratio < target_ratio:

            target_height = int(
                width
                / target_ratio
            )

            start_y = (
                height
                - target_height
            )

            # Keep lower road area.
            frame = frame[
                start_y:,
                :,
            ]

        # ---------------------------------
        # Too wide
        # ---------------------------------

        elif current_ratio > target_ratio:

            target_width = int(
                height
                * target_ratio
            )

            start_x = (
                width
                - target_width
            ) // 2

            frame = frame[
                :,
                start_x:
                start_x + target_width,
            ]

        return frame

    def _print_ego_lane(
        self,
        ego_lane,
    ):
        print(
            "Left boundary index:",
            ego_lane[
                "left_index"
            ],
        )

        print(
            "Right boundary index:",
            ego_lane[
                "right_index"
            ],
        )

        print(
            "Vehicle center X:",
            ego_lane[
                "vehicle_center_x"
            ],
        )

        print(
            "Reference Y:",
            ego_lane[
                "reference_y"
            ],
        )

        if (
            ego_lane["left"]
            is not None
        ):

            print(
                "Left boundary points:",
                len(
                    ego_lane[
                        "left"
                    ]
                ),
            )

        else:

            print(
                "Left boundary: "
                "not found"
            )

        if (
            ego_lane["right"]
            is not None
        ):

            print(
                "Right boundary points:",
                len(
                    ego_lane[
                        "right"
                    ]
                ),
            )

        else:

            print(
                "Right boundary: "
                "not found"
            )

    def test_image(
        self,
        image_path: str,
        output_path: str,
        crop_16_9: bool = True,
    ) -> None:

        image = cv2.imread(
            image_path
        )

        if image is None:

            raise FileNotFoundError(
                "Could not load image: "
                f"{image_path}"
            )

        print(
            "Original image shape:",
            image.shape,
        )

        if crop_16_9:

            image = (
                self.crop_to_16_9(
                    image
                )
            )

            print(
                "Cropped image shape:",
                image.shape,
            )

        # =================================
        # Predict
        # =================================

        ego_lane, result = (
            self.detector
            .predict_ego_and_visualize(
                image
            )
        )

        self._print_ego_lane(
            ego_lane
        )

        # =================================
        # Save
        # =================================

        output_dir = (
            os.path.dirname(
                output_path
            )
        )

        if output_dir:

            os.makedirs(
                output_dir,
                exist_ok=True,
            )

        success = cv2.imwrite(
            output_path,
            result,
        )

        if not success:

            raise RuntimeError(
                "Could not save image: "
                f"{output_path}"
            )

        print(
            "Result saved to: "
            f"{output_path}"
        )

    def test_video(
        self,
        video_path: str,
        output_path: str,
        crop_16_9: bool = True,
        progress_interval: int = 30,
    ) -> None:

        # =================================
        # Important:
        # reset previous triangle
        # =================================

        self.detector.reset_visualization()

        capture = cv2.VideoCapture(
            video_path
        )

        if not capture.isOpened():

            raise FileNotFoundError(
                "Could not open video: "
                f"{video_path}"
            )

        # =================================
        # Video metadata
        # =================================

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

        output_dir = (
            os.path.dirname(
                output_path
            )
        )

        if output_dir:

            os.makedirs(
                output_dir,
                exist_ok=True,
            )

        writer = None

        frame_index = 0

        start_time = (
            time.perf_counter()
        )

        try:

            while True:

                success, frame = (
                    capture.read()
                )

                if not success:
                    break

                # -------------------------
                # Crop
                # -------------------------

                if crop_16_9:

                    frame = (
                        self.crop_to_16_9(
                            frame
                        )
                    )

                # -------------------------
                # Ego lane inference
                # -------------------------

                ego_lane, result = (
                    self.detector
                    .predict_ego_and_visualize(
                        frame
                    )
                )

                # -------------------------
                # Video writer
                # -------------------------

                if writer is None:

                    (
                        output_height,
                        output_width,
                    ) = result.shape[:2]

                    writer = (
                        cv2.VideoWriter(
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
                    )

                    if not writer.isOpened():

                        raise RuntimeError(
                            "Could not create "
                            "output video: "
                            f"{output_path}"
                        )

                    print(
                        "\nVideo information"
                    )

                    print(
                        "Resolution:",
                        f"{output_width}"
                        "x"
                        f"{output_height}",
                    )

                    print(
                        "Input FPS:",
                        f"{input_fps:.2f}",
                    )

                    print(
                        "Total frames:",
                        total_frames,
                    )

                writer.write(
                    result
                )

                frame_index += 1

                # -------------------------
                # Progress
                # -------------------------

                if (
                    progress_interval > 0
                    and
                    frame_index
                    % progress_interval
                    == 0
                ):

                    elapsed = (
                        time.perf_counter()
                        - start_time
                    )

                    processing_fps = (
                        frame_index
                        / elapsed
                    )

                    left_found = (
                        ego_lane[
                            "left"
                        ]
                        is not None
                    )

                    right_found = (
                        ego_lane[
                            "right"
                        ]
                        is not None
                    )

                    left_index = (
                        ego_lane[
                            "left_index"
                        ]
                    )

                    right_index = (
                        ego_lane[
                            "right_index"
                        ]
                    )

                    print(
                        f"Frame "
                        f"{frame_index}"
                        f"/{total_frames} | "
                        f"Left: "
                        f"{left_found}"
                        f" ({left_index}) | "
                        f"Right: "
                        f"{right_found}"
                        f" ({right_index}) | "
                        f"FPS: "
                        f"{processing_fps:.2f}"
                    )

        finally:

            capture.release()

            if writer is not None:

                writer.release()

        # =================================
        # Final stats
        # =================================

        elapsed = (
            time.perf_counter()
            - start_time
        )

        average_fps = (
            frame_index
            / elapsed

            if elapsed > 0

            else 0
        )

        print(
            "\nProcessed frames:",
            frame_index,
        )

        print(
            "Average processing FPS:",
            f"{average_fps:.2f}",
        )

        print(
            "Result saved to:",
            output_path,
        )


class ArgumentParser:

    @staticmethod
    def create():

        parser = (
            argparse.ArgumentParser(
                description=(
                    "Test UFLDv2 "
                    "ego-lane detection."
                )
            )
        )

        # =================================
        # Model
        # =================================

        parser.add_argument(
            "--model",

            type=str,

            default=(
                "tusimple_res34"
            ),

            choices=list(
                EgoLaneTest
                .MODEL_CONFIGS
                .keys()
            ),

            help=(
                "UFLDv2 pretrained model."
            ),
        )

        # =================================
        # Mode
        # =================================

        parser.add_argument(
            "--mode",

            type=str,

            required=True,

            choices=[
                "image",
                "video",
            ],

            help=(
                "Run on image "
                "or video."
            ),
        )

        # =================================
        # Input
        # =================================

        parser.add_argument(
            "--input",

            type=str,

            required=True,

            help=(
                "Input image/video path."
            ),
        )

        # =================================
        # Output
        # =================================

        parser.add_argument(
            "--output",

            type=str,

            default=None,

            help=(
                "Output path."
            ),
        )

        # =================================
        # Confidence threshold
        # =================================

        parser.add_argument(
            "--confidence-threshold",

            type=float,

            default=0.60,

            help=(
                "Minimum probability that "
                "a lane anchor exists. "
                "Range: 0.0-1.0. "
                "Recommended: 0.5-0.8."
            ),
        )

        # =================================
        # Temporal smoothing
        # =================================

        parser.add_argument(
            "--smooth-alpha",

            type=float,

            default=0.15,

            help=(
                "Triangle temporal smoothing. "
                "Smaller = smoother. "
                "Recommended: 0.10-0.20."
            ),
        )

        parser.add_argument(
            "--hold-frames",

            type=int,

            default=5,

            help=(
                "Keep previous triangle "
                "when detection disappears "
                "for a few frames."
            ),
        )

        # =================================
        # Crop
        # =================================

        parser.add_argument(
            "--no-crop",

            action="store_true",

            help=(
                "Disable 16:9 crop."
            ),
        )

        # =================================
        # Progress
        # =================================

        parser.add_argument(
            "--progress-interval",

            type=int,

            default=30,

            help=(
                "Print progress every "
                "N video frames."
            ),
        )

        return parser


def main():

    parser = (
        ArgumentParser.create()
    )

    args = (
        parser.parse_args()
    )

    # =====================================
    # Validate arguments
    # =====================================

    if not (
        0.0
        <= args.confidence_threshold
        <= 1.0
    ):

        parser.error(
            "--confidence-threshold "
            "must be between 0 and 1."
        )

    if not (
        0.0
        < args.smooth_alpha
        <= 1.0
    ):

        parser.error(
            "--smooth-alpha must "
            "be > 0 and <= 1."
        )

    if args.hold_frames < 0:

        parser.error(
            "--hold-frames cannot "
            "be negative."
        )

    # =====================================
    # Tester
    # =====================================

    tester = EgoLaneTest(
        model_name=(
            args.model
        ),

        confidence_threshold=(
            args.confidence_threshold
        ),

        smoothing_alpha=(
            args.smooth_alpha
        ),

        hold_frames=(
            args.hold_frames
        ),
    )

    # =====================================
    # Output path
    # =====================================

    output_path = (
        args.output
    )

    if output_path is None:

        if args.mode == "image":

            output_path = (
                "ai/outputs/"
                "lane_detector/"
                f"{args.model}_"
                "ego_lane_result.png"
            )

        else:

            output_path = (
                "ai/outputs/"
                "lane_detector/"
                f"{args.model}_"
                "ego_lane_result.mp4"
            )

    # =====================================
    # Run
    # =====================================

    if args.mode == "image":

        tester.test_image(
            image_path=(
                args.input
            ),

            output_path=(
                output_path
            ),

            crop_16_9=(
                not args.no_crop
            ),
        )

    else:

        tester.test_video(
            video_path=(
                args.input
            ),

            output_path=(
                output_path
            ),

            crop_16_9=(
                not args.no_crop
            ),

            progress_interval=(
                args.progress_interval
            ),
        )


if __name__ == "__main__":
    main()
