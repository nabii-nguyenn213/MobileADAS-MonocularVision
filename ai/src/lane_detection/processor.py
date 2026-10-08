import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

class LanePreprocessor:

    def __init__(
        self,
        input_width: int = 800,
        input_height: int = 320,
        crop_ratio: float = 0.8,
    ):
        self.input_width = input_width
        self.input_height = input_height
        self.crop_ratio = crop_ratio

        resize_height = int(
            input_height / crop_ratio
        )

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (
                        resize_height,
                        input_width,
                    )
                ),

                transforms.ToTensor(),

                transforms.Normalize(
                    mean=(
                        0.485,
                        0.456,
                        0.406,
                    ),
                    std=(
                        0.229,
                        0.224,
                        0.225,
                    ),
                ),
            ]
        )

    def __call__(
        self,
        image: np.ndarray,
    ) -> torch.Tensor:

        if image is None:
            raise ValueError(
                "Input image cannot be None."
            )

        # OpenCV:
        # BGR -> RGB
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        image = Image.fromarray(
            image
        )

        tensor = self.transform(
            image
        )

        # Keep only bottom input_height
        # pixels after resizing.
        tensor = tensor[
            :,
            -self.input_height:,
            :,
        ]

        # [3, H, W]
        # ->
        # [1, 3, H, W]
        tensor = tensor.unsqueeze(
            0
        )

        return tensor


class LanePostprocessor:

    def __init__(
        self,
        num_row: int = 56,
        num_col: int = 41,
        local_width: int = 1,
        confidence_threshold: float = 0.60,
    ):
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                "confidence_threshold must "
                "be between 0 and 1."
            )

        self.num_row = num_row
        self.num_col = num_col

        self.local_width = (
            local_width
        )

        self.confidence_threshold = (
            confidence_threshold
        )

        # TuSimple row anchors
        self.row_anchor = (
            np.linspace(
                160,
                710,
                num_row,
            )
            / 720
        )

        self.col_anchor = np.linspace(
            0,
            1,
            num_col,
        )

    def __call__(
        self,
        pred,
        image_width: int,
        image_height: int,
    ):
        # --------------------------------
        # Move predictions to CPU
        # --------------------------------

        loc_row = (
            pred["loc_row"]
            .detach()
            .cpu()
        )

        loc_col = (
            pred["loc_col"]
            .detach()
            .cpu()
        )

        exist_row = (
            pred["exist_row"]
            .detach()
            .cpu()
        )

        exist_col = (
            pred["exist_col"]
            .detach()
            .cpu()
        )

        # --------------------------------
        # Shapes
        # --------------------------------

        (
            _,
            num_grid_row,
            num_row,
            _,
        ) = loc_row.shape

        (
            _,
            num_grid_col,
            num_col,
            _,
        ) = loc_col.shape

        # --------------------------------
        # Most likely grid cell
        # --------------------------------

        max_row = loc_row.argmax(
            dim=1
        )

        max_col = loc_col.argmax(
            dim=1
        )

        # --------------------------------
        # Lane existence confidence
        #
        # exist_row shape:
        # [B, 2, num_row, num_lanes]
        #
        # class 0 = lane does not exist
        # class 1 = lane exists
        # --------------------------------

        row_probability = torch.softmax(
            exist_row,
            dim=1,
        )

        col_probability = torch.softmax(
            exist_col,
            dim=1,
        )

        # Probability of class 1:
        # "lane exists"
        row_confidence = (
            row_probability[
                :,
                1,
                :,
                :,
            ]
        )

        col_confidence = (
            col_probability[
                :,
                1,
                :,
                :,
            ]
        )

        # --------------------------------
        # Apply confidence threshold
        # --------------------------------

        valid_row = (
            row_confidence
            >= self.confidence_threshold
        )

        valid_col = (
            col_confidence
            >= self.confidence_threshold
        )

        lanes = []

        # ==================================================
        # INNER LANES
        #
        # TuSimple model lane IDs:
        # 1 and 2
        #
        # These are decoded using row anchors.
        # ==================================================

        for lane_index in [
            1,
            2,
        ]:

            lane = []

            # Require enough confident
            # anchors before accepting lane.
            valid_count = (
                valid_row[
                    0,
                    :,
                    lane_index,
                ]
                .sum()
                .item()
            )

            if valid_count > num_row / 2:

                for anchor_index in range(
                    num_row
                ):

                    # Skip anchor if lane
                    # confidence is too low.
                    if not valid_row[
                        0,
                        anchor_index,
                        lane_index,
                    ]:
                        continue

                    center = int(
                        max_row[
                            0,
                            anchor_index,
                            lane_index,
                        ]
                    )

                    start = max(
                        0,
                        center
                        - self.local_width,
                    )

                    end = min(
                        num_grid_row - 1,
                        center
                        + self.local_width,
                    )

                    indices = torch.arange(
                        start,
                        end + 1,
                    )

                    probabilities = torch.softmax(
                        loc_row[
                            0,
                            indices,
                            anchor_index,
                            lane_index,
                        ],
                        dim=0,
                    )

                    # Local soft-argmax
                    position = (
                        probabilities
                        * indices.float()
                    ).sum() + 0.5

                    x = (
                        position
                        / (num_grid_row - 1)
                        * image_width
                    )

                    y = (
                        self.row_anchor[
                            anchor_index
                        ]
                        * image_height
                    )

                    lane.append(
                        (
                            int(x),
                            int(y),
                        )
                    )

            lanes.append(
                lane
            )

        # ==================================================
        # OUTER LANES
        #
        # TuSimple model lane IDs:
        # 0 and 3
        #
        # These are decoded using column anchors.
        # ==================================================

        for lane_index in [
            0,
            3,
        ]:

            lane = []

            valid_count = (
                valid_col[
                    0,
                    :,
                    lane_index,
                ]
                .sum()
                .item()
            )

            if valid_count > num_col / 4:

                for anchor_index in range(
                    num_col
                ):

                    if not valid_col[
                        0,
                        anchor_index,
                        lane_index,
                    ]:
                        continue

                    center = int(
                        max_col[
                            0,
                            anchor_index,
                            lane_index,
                        ]
                    )

                    start = max(
                        0,
                        center
                        - self.local_width,
                    )

                    end = min(
                        num_grid_col - 1,
                        center
                        + self.local_width,
                    )

                    indices = torch.arange(
                        start,
                        end + 1,
                    )

                    probabilities = torch.softmax(
                        loc_col[
                            0,
                            indices,
                            anchor_index,
                            lane_index,
                        ],
                        dim=0,
                    )

                    position = (
                        probabilities
                        * indices.float()
                    ).sum() + 0.5

                    y = (
                        position
                        / (num_grid_col - 1)
                        * image_height
                    )

                    x = (
                        self.col_anchor[
                            anchor_index
                        ]
                        * image_width
                    )

                    lane.append(
                        (
                            int(x),
                            int(y),
                        )
                    )

            lanes.append(
                lane
            )

        return lanes

class CurveLanesPostprocessor:

    def __init__(
        self,
        num_row: int = 72,
        num_col: int = 81,
        num_lanes: int = 10,
        local_width: int = 1,
    ):
        self.num_row = num_row
        self.num_col = num_col
        self.num_lanes = num_lanes
        self.local_width = local_width

        # Official CurveLanes anchors
        self.row_anchor = np.linspace(
            0.4,
            1.0,
            num_row,
        )

        self.col_anchor = np.linspace(
            0.0,
            1.0,
            num_col,
        )

    def _decode_row(
        self,
        loc_row: torch.Tensor,
        exist_row: torch.Tensor,
        image_width: int,
        image_height: int,
        lane_index: int,
    ):
        _, num_grid_row, num_row, _ = (
            loc_row.shape
        )

        max_row = loc_row.argmax(
            dim=1
        )

        valid_row = exist_row.argmax(
            dim=1
        )

        points = []

        # CurveLanes official evaluation
        # uses num_row / 4 here.
        if (
            valid_row[
                0,
                :,
                lane_index,
            ].sum()
            <= num_row / 4
        ):
            return points

        for anchor_index in range(
            num_row
        ):
            if not valid_row[
                0,
                anchor_index,
                lane_index,
            ]:
                continue

            center = int(
                max_row[
                    0,
                    anchor_index,
                    lane_index,
                ]
            )

            start = max(
                0,
                center - self.local_width,
            )

            end = min(
                num_grid_row - 1,
                center + self.local_width,
            )

            indices = torch.arange(
                start,
                end + 1,
            )

            probabilities = torch.softmax(
                loc_row[
                    0,
                    indices,
                    anchor_index,
                    lane_index,
                ],
                dim=0,
            )

            position = (
                probabilities
                * indices.float()
            ).sum() + 0.5

            x = (
                position.item()
                / (num_grid_row - 1)
                * image_width
            )

            y = (
                self.row_anchor[
                    anchor_index
                ]
                * image_height
            )

            points.append(
                (
                    float(x),
                    float(y),
                )
            )

        return points

    def _decode_col(
        self,
        loc_col: torch.Tensor,
        exist_col: torch.Tensor,
        image_width: int,
        image_height: int,
        lane_index: int,
    ):
        _, num_grid_col, num_col, _ = (
            loc_col.shape
        )

        max_col = loc_col.argmax(
            dim=1
        )

        valid_col = exist_col.argmax(
            dim=1
        )

        points = []

        if (
            valid_col[
                0,
                :,
                lane_index,
            ].sum()
            <= num_col / 4
        ):
            return points

        for anchor_index in range(
            num_col
        ):
            if not valid_col[
                0,
                anchor_index,
                lane_index,
            ]:
                continue

            center = int(
                max_col[
                    0,
                    anchor_index,
                    lane_index,
                ]
            )

            start = max(
                0,
                center - self.local_width,
            )

            end = min(
                num_grid_col - 1,
                center + self.local_width,
            )

            indices = torch.arange(
                start,
                end + 1,
            )

            probabilities = torch.softmax(
                loc_col[
                    0,
                    indices,
                    anchor_index,
                    lane_index,
                ],
                dim=0,
            )

            position = (
                probabilities
                * indices.float()
            ).sum() + 0.5

            y = (
                position.item()
                / (num_grid_col - 1)
                * image_height
            )

            x = (
                self.col_anchor[
                    anchor_index
                ]
                * image_width
            )

            points.append(
                (
                    float(x),
                    float(y),
                )
            )

        return points

    def _fit_curve(
        self,
        points,
        image_width: int,
        image_height: int,
    ):
        if len(points) < 3:
            return [
                (
                    int(x),
                    int(y),
                )
                for x, y in points
            ]

        points = np.asarray(
            points,
            dtype=np.float32,
        )

        x = points[:, 0]
        y = points[:, 1]

        # Remove duplicate points.
        points = np.unique(
            points,
            axis=0,
        )

        if len(points) < 3:
            return [
                (
                    int(point[0]),
                    int(point[1]),
                )
                for point in points
            ]

        x = points[:, 0]
        y = points[:, 1]

        try:
            # --------------------------------------------------
            # Option 1:
            # y = f(x)
            # --------------------------------------------------

            coefficients_y = np.polyfit(
                x,
                y,
                deg=2,
            )

            predicted_y = np.polyval(
                coefficients_y,
                x,
            )

            error_y = np.mean(
                (predicted_y - y) ** 2
            )

            # --------------------------------------------------
            # Option 2:
            # x = f(y)
            # --------------------------------------------------

            coefficients_x = np.polyfit(
                y,
                x,
                deg=2,
            )

            predicted_x = np.polyval(
                coefficients_x,
                y,
            )

            error_x = np.mean(
                (predicted_x - x) ** 2
            )

            # --------------------------------------------------
            # Choose the orientation with lower fitting error.
            # --------------------------------------------------

            if error_y < error_x:

                x_new = np.linspace(
                    x.min(),
                    x.max(),
                    36,
                )

                y_new = np.polyval(
                    coefficients_y,
                    x_new,
                )

            else:

                y_new = np.linspace(
                    y.min(),
                    y.max(),
                    41,
                )

                x_new = np.polyval(
                    coefficients_x,
                    y_new,
                )

            curve = []

            for px, py in zip(
                x_new,
                y_new,
            ):
                # Ignore fitted points outside image.
                if (
                    0 <= px < image_width
                    and
                    0 <= py < image_height
                ):
                    curve.append(
                        (
                            int(px),
                            int(py),
                        )
                    )

            return curve

        except (
            np.linalg.LinAlgError,
            ValueError,
        ):
            return [
                (
                    int(px),
                    int(py),
                )
                for px, py in points
            ]

    def __call__(
        self,
        pred: dict,
        image_width: int,
        image_height: int,
    ):
        loc_row = (
            pred["loc_row"]
            .detach()
            .cpu()
        )

        loc_col = (
            pred["loc_col"]
            .detach()
            .cpu()
        )

        exist_row = (
            pred["exist_row"]
            .detach()
            .cpu()
        )

        exist_col = (
            pred["exist_col"]
            .detach()
            .cpu()
        )

        lanes = []

        for lane_index in range(
            self.num_lanes
        ):
            # Decode same lane using
            # horizontal anchors.
            row_points = self._decode_row(
                loc_row=loc_row,
                exist_row=exist_row,
                image_width=image_width,
                image_height=image_height,
                lane_index=lane_index,
            )

            # Decode same lane using
            # vertical anchors.
            col_points = self._decode_col(
                loc_col=loc_col,
                exist_col=exist_col,
                image_width=image_width,
                image_height=image_height,
                lane_index=lane_index,
            )

            # CurveLanes combines row and
            # column predictions.
            combined_points = (
                row_points
                + col_points
            )

            lane = self._fit_curve(
                points=combined_points,
                image_width=image_width,
                image_height=image_height,
            )

            lanes.append(
                lane
            )

        return lanes
