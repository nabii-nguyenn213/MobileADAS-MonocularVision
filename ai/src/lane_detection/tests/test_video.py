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
from ai.src.lane_detection.models.ufldv2_curvelanes import (
    UFLDv2CurveLanes,
)


class LaneVideoTest:

    MODEL_CONFIGS = {

        # ============================================================
        # TuSimple - ResNet18
        # ============================================================

        "tusimple_res18": {
            "dataset": "tusimple",
            "model_type": "standard",
            "backbone": "resnet18",

            "checkpoint": (
                "ai/checkpoints/lane_detection/"
                "tusimple_res18.pth"
            ),

            "num_grid_row": 100,
            "num_grid_col": 100,

            "num_row": 56,
            "num_col": 41,

            "num_lanes": 4,

            "input_width": 800,
            "input_height": 320,

            "crop_ratio": 0.8,
        },

        # ============================================================
        # TuSimple - ResNet34
        # ============================================================

        "tusimple_res34": {
            "dataset": "tusimple",
            "model_type": "standard",
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
        },

        # ============================================================
        # CurveLanes - ResNet34
        # ============================================================

        "curvelanes_res34": {
            "dataset": "curvelanes",
            "model_type": "curvelanes",
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
        },
    }

    def __init__(
        self,
        model_name: str,
    ):
        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(
                f"Unsupported model: {model_name}. "
                f"Supported models: "
                f"{list(self.MODEL_CONFIGS.keys())}"
            )

        self.model_name = model_name
        self.model_config = self.MODEL_CONFIGS[
            model_name
        ]

        self.config = LaneDetectionConfig()

        # ------------------------------------------------------------
        # Apply model-specific config
        # ------------------------------------------------------------

        self.config.input_width = (
            self.model_config["input_width"]
        )

        self.config.input_height = (
            self.model_config["input_height"]
        )

        self.config.checkpoint_path = (
            self.model_config["checkpoint"]
        )

        # ------------------------------------------------------------
        # Build model
        # ------------------------------------------------------------

        model = self._build_model()

        # ------------------------------------------------------------
        # Build detector
        # ------------------------------------------------------------

        self.detector = LaneDetector(
            model=model,
            config=self.config,
        )

        # ------------------------------------------------------------
        # Replace processors based on dataset
        # ------------------------------------------------------------

        self.detector.preprocessor = (
            LanePreprocessor(
                input_width=(
                    self.model_config[
                        "input_width"
                    ]
                ),
                input_height=(
                    self.model_config[
                        "input_height"
                    ]
                ),
                crop_ratio=(
                    self.model_config[
                        "crop_ratio"
                    ]
                ),
            )
        )

        self.detector.postprocessor = (
            self._build_postprocessor()
        )

        # ------------------------------------------------------------
        # Information
        # ------------------------------------------------------------

        print("=" * 60)

        print(
            f"Model: "
            f"{self.model_name}"
        )

        print(
            f"Dataset: "
            f"{self.model_config['dataset']}"
        )

        print(
            f"Backbone: "
            f"{self.model_config['backbone']}"
        )

        print(
            "Network input:",
            f"{self.model_config['input_width']}"
            "x"
            f"{self.model_config['input_height']}",
        )

        print(
            f"Number of lanes: "
            f"{self.model_config['num_lanes']}"
        )

        print(
            f"Checkpoint: "
            f"{self.config.checkpoint_path}"
        )

        print("=" * 60)

        # ------------------------------------------------------------
        # Load checkpoint
        # ------------------------------------------------------------

        self.detector.load_weights()

        print(
            "Checkpoint loaded successfully."
        )

    def _build_model(self):

        model_type = (
            self.model_config["model_type"]
        )

        if model_type == "standard":

            return UFLDv2(
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

        if model_type == "curvelanes":

            return UFLDv2CurveLanes(
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

        raise ValueError(
            f"Unknown model type: "
            f"{model_type}"
        )

    def _build_postprocessor(self):

        dataset = (
            self.model_config["dataset"]
        )

        if dataset == "tusimple":

            return LanePostprocessor(
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
            )

        if dataset == "curvelanes":

            return CurveLanesPostprocessor(
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
            )

        raise ValueError(
            f"Unsupported dataset: "
            f"{dataset}"
        )

    def crop_to_16_9(
        self,
        frame,
    ):
        height, width = frame.shape[:2]

        current_ratio = (
            width / height
        )

        target_ratio = (
            16 / 9
        )

        # ------------------------------------------------------------
        # Too tall
        # ------------------------------------------------------------

        if current_ratio < target_ratio:

            target_height = int(
                width / target_ratio
            )

            start_y = (
                height - target_height
            )

            # Keep bottom region
            # because road is there.
            frame = frame[
                start_y:,
                :
            ]

        # ------------------------------------------------------------
        # Too wide
        # ------------------------------------------------------------

        elif current_ratio > target_ratio:

            target_width = int(
                height * target_ratio
            )

            start_x = (
                width - target_width
            ) // 2

            frame = frame[
                :,
                start_x:
                start_x + target_width
            ]

        return frame

    def run(
        self,
        video_path: str,
        output_path: str,
        crop_16_9: bool = True,
        progress_interval: int = 30,
    ) -> None:

        capture = cv2.VideoCapture(
            video_path
        )

        if not capture.isOpened():
            raise FileNotFoundError(
                f"Could not open video: "
                f"{video_path}"
            )

        # ------------------------------------------------------------
        # Video information
        # ------------------------------------------------------------

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

        # ------------------------------------------------------------
        # Output directory
        # ------------------------------------------------------------

        output_dir = os.path.dirname(
            output_path
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

                # ----------------------------------------------------
                # Optional 16:9 crop
                # ----------------------------------------------------

                if crop_16_9:

                    frame = (
                        self.crop_to_16_9(
                            frame
                        )
                    )

                # ----------------------------------------------------
                # Lane detection
                # ----------------------------------------------------

                lanes, result = (
                    self.detector
                    .predict_and_visualize(
                        frame
                    )
                )

                # ----------------------------------------------------
                # Initialize writer
                # ----------------------------------------------------

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
                            f"output video: "
                            f"{output_path}"
                        )

                    print(
                        "\nVideo information"
                    )

                    print(
                        "Output resolution:",
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

                # ----------------------------------------------------
                # Write result
                # ----------------------------------------------------

                writer.write(
                    result
                )

                frame_index += 1

                # ----------------------------------------------------
                # Progress
                # ----------------------------------------------------

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

                    detected_lanes = sum(
                        len(lane) > 0
                        for lane in lanes
                    )

                    if total_frames > 0:

                        print(
                            f"Frame "
                            f"{frame_index}/"
                            f"{total_frames} | "
                            f"Lanes: "
                            f"{detected_lanes} | "
                            f"FPS: "
                            f"{processing_fps:.2f}"
                        )

                    else:

                        print(
                            f"Frame "
                            f"{frame_index} | "
                            f"Lanes: "
                            f"{detected_lanes} | "
                            f"FPS: "
                            f"{processing_fps:.2f}"
                        )

        finally:

            capture.release()

            if writer is not None:
                writer.release()

        # ------------------------------------------------------------
        # Final statistics
        # ------------------------------------------------------------

        elapsed = (
            time.perf_counter()
            - start_time
        )

        average_fps = (
            frame_index / elapsed
            if elapsed > 0
            else 0
        )

        print(
            f"\nProcessed frames: "
            f"{frame_index}"
        )

        print(
            f"Average processing FPS: "
            f"{average_fps:.2f}"
        )

        print(
            f"Result saved to: "
            f"{output_path}"
        )


class ArgumentParser:

    @staticmethod
    def create():

        parser = argparse.ArgumentParser(
            description=(
                "Run UFLDv2 lane detection "
                "on a video."
            )
        )

        parser.add_argument(
            "--model",
            type=str,
            default="tusimple_res34",
            choices=list(
                LaneVideoTest
                .MODEL_CONFIGS
                .keys()
            ),
            help=(
                "Pretrained UFLDv2 "
                "model to use."
            ),
        )

        parser.add_argument(
            "--video",
            type=str,
            required=True,
            help=(
                "Path to input video."
            ),
        )

        parser.add_argument(
            "--output",
            type=str,
            default=None,
            help=(
                "Path to output video."
            ),
        )

        parser.add_argument(
            "--no-crop",
            action="store_true",
            help=(
                "Disable automatic "
                "16:9 cropping."
            ),
        )

        parser.add_argument(
            "--progress-interval",
            type=int,
            default=30,
            help=(
                "Print progress every "
                "N frames."
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

    # ------------------------------------------------------------
    # Automatic output filename
    # ------------------------------------------------------------

    output_path = (
        args.output
    )

    if output_path is None:

        output_path = (
            "ai/outputs/"
            "lane_detector/"
            f"{args.model}_result.mp4"
        )

    # ------------------------------------------------------------
    # Create tester
    # ------------------------------------------------------------

    tester = LaneVideoTest(
        model_name=args.model
    )

    # ------------------------------------------------------------
    # Run
    # ------------------------------------------------------------

    tester.run(
        video_path=args.video,
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
