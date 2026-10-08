import cv2

from ai.src.lane_detection.config import LaneDetectionConfig
from ai.src.lane_detection.detector import LaneDetector
from ai.src.lane_detection.models.ufldv2 import UFLDv2


class LaneImageTest:

    def __init__(self):
        self.config = LaneDetectionConfig()
        self.config.checkpoint_path = (
            "ai/checkpoints/lane_detection/tusimple_res18.pth"
        )

        model = UFLDv2(
            num_grid_row=100,
            num_grid_col=100,
            num_row=56,
            num_col=41,
            num_lanes=4,
            input_height=self.config.input_height,
            input_width=self.config.input_width,
        )

        self.detector = LaneDetector(
            model=model,
            config=self.config,
        )

        self.detector.load_weights()

    def crop_to_16_9(
        self,
        image,
    ):
        height, width = image.shape[:2]

        current_ratio = width / height
        target_ratio = 16 / 9

        # Image is too tall -> crop vertically
        if current_ratio < target_ratio:
            target_height = int(
                width / target_ratio
            )

            start_y = height - target_height

            image = image[
                start_y:,
                :
            ]

        # Image is too wide -> crop horizontally
        elif current_ratio > target_ratio:
            target_width = int(
                height * target_ratio
            )

            start_x = (
                width - target_width
            ) // 2

            image = image[
                :,
                start_x:start_x + target_width
            ]

        return image

    def run(
        self,
        image_path: str,
        output_path: str,
    ) -> None:

        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(
                f"Could not load image: {image_path}"
            )

        print(
            "Original image shape:",
            image.shape,
        )

        # Convert image framing to 16:9
        image = self.crop_to_16_9(
            image
        )

        print(
            "Cropped image shape:",
            image.shape,
        )

        lanes, result = (
            self.detector.predict_and_visualize(
                image
            )
        )

        print("\nDetected lanes:")

        for index, lane in enumerate(lanes):
            print(
                f"Lane {index}: "
                f"{len(lane)} points"
            )

        success = cv2.imwrite(
            output_path,
            result,
        )

        if not success:
            raise RuntimeError(
                f"Could not save output to: "
                f"{output_path}"
            )

        print(
            f"\nResult saved to: "
            f"{output_path}"
        )


if __name__ == "__main__":

    tester = LaneImageTest()

    tester.run(
        image_path=(
            "dataset/tests/"
            "lane_detector/test.png"
        ),
        output_path=(
            "ai/outputs/"
            "lane_detector/result.png"
        ),
    )
