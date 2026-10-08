import numpy as np
import torch

from ai.src.lane_detection.config import (
    LaneDetectionConfig,
)

from ai.src.lane_detection.processor import (
    LanePreprocessor,
    LanePostprocessor,
)

from ai.src.lane_detection.model import (
    LaneModel,
)

from ai.src.lane_detection.visualizer import (
    LaneVisualizer,
)

from ai.src.lane_detection.ego_lane import (
    EgoLaneSelector,
)


class LaneDetector:

    def __init__(
        self,
        model: torch.nn.Module,
        config: LaneDetectionConfig,
        confidence_threshold: float = 0.60,
    ):
        self.config = config

        # --------------------------------
        # Preprocessor
        # --------------------------------

        self.preprocessor = LanePreprocessor(
            input_width=(
                config.input_width
            ),
            input_height=(
                config.input_height
            ),
        )

        # --------------------------------
        # Model wrapper
        # --------------------------------

        self.model = LaneModel(
            model=model,
            device=config.device,
        )

        # --------------------------------
        # Postprocessor
        #
        # Confidence threshold is applied
        # here.
        # --------------------------------

        self.postprocessor = LanePostprocessor(
            confidence_threshold=(
                confidence_threshold
            )
        )

        # --------------------------------
        # Ego lane selector
        # --------------------------------

        self.ego_lane_selector = (
            EgoLaneSelector(
                vehicle_center_ratio=0.5,
                reference_y_ratio=0.85,
            )
        )

        # --------------------------------
        # Visualizer
        # --------------------------------

        self.visualizer = (
            LaneVisualizer()
        )

    def load_weights(
        self,
    ) -> None:

        self.model.load_checkpoint(
            self.config.checkpoint_path
        )

    def predict(
        self,
        image: np.ndarray,
    ):
        """
        Detect all lane boundaries.
        """

        if image is None:
            raise ValueError(
                "Input image cannot be None."
            )

        (
            image_height,
            image_width,
        ) = image.shape[:2]

        # --------------------------------
        # Preprocess
        # --------------------------------

        input_tensor = (
            self.preprocessor(
                image
            )
        )

        # --------------------------------
        # UFLDv2 inference
        # --------------------------------

        raw_output = (
            self.model.predict(
                input_tensor
            )
        )

        # --------------------------------
        # Decode lane coordinates
        # --------------------------------

        lanes = (
            self.postprocessor(
                pred=raw_output,
                image_width=(
                    image_width
                ),
                image_height=(
                    image_height
                ),
            )
        )

        return lanes

    def predict_ego_lane(
        self,
        image: np.ndarray,
    ):
        """
        Detect all lanes, then select
        the two boundaries surrounding
        the ego vehicle.
        """

        if image is None:
            raise ValueError(
                "Input image cannot be None."
            )

        (
            image_height,
            image_width,
        ) = image.shape[:2]

        lanes = self.predict(
            image
        )

        ego_lane = (
            self.ego_lane_selector.select(
                lanes=lanes,
                image_width=image_width,
                image_height=image_height,
            )
        )

        return ego_lane

    def visualize(
        self,
        image: np.ndarray,
        lanes,
    ) -> np.ndarray:
        """
        Debug visualization:
        draw all detected lane points.
        """

        return self.visualizer.draw_points(
            image,
            lanes,
        )

    def visualize_ego_lane(
        self,
        image: np.ndarray,
        ego_lane,
    ) -> np.ndarray:
        """
        Visualize the selected forward
        lane as the stabilized triangle.
        """

        return self.visualizer.draw_ego_lane(
            image=image,
            left_lane=(
                ego_lane["left"]
            ),
            right_lane=(
                ego_lane["right"]
            ),
        )

    def predict_and_visualize(
        self,
        image: np.ndarray,
    ):
        """
        Debug mode:
        all detected lane boundaries.
        """

        lanes = self.predict(
            image
        )

        visualization = (
            self.visualize(
                image,
                lanes,
            )
        )

        return (
            lanes,
            visualization,
        )

    def predict_ego_and_visualize(
        self,
        image: np.ndarray,
    ):
        """
        Ego-lane mode.
        """

        ego_lane = (
            self.predict_ego_lane(
                image
            )
        )

        visualization = (
            self.visualize_ego_lane(
                image,
                ego_lane,
            )
        )

        return (
            ego_lane,
            visualization,
        )

    def set_confidence_threshold(
        self,
        threshold: float,
    ) -> None:
        """
        Change confidence threshold
        without recreating the detector.
        """

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "Confidence threshold must "
                "be between 0 and 1."
            )

        self.postprocessor.confidence_threshold = (
            threshold
        )

    def reset_visualization(
        self,
    ) -> None:
        """
        Reset temporal smoothing state.
        """

        self.visualizer.reset()
