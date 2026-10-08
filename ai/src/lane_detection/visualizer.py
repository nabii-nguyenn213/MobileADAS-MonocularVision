import cv2
import numpy as np


class LaneVisualizer:

    def __init__(
        self,
        point_radius: int = 5,
        line_thickness: int = 5,

        # OpenCV uses BGR
        lane_color=(255, 255, 255),
        ego_fill_color=(255, 0, 0),

        ego_fill_alpha: float = 0.35,

        # Triangle geometry
        apex_y_ratio: float = 0.52,
        base_y_ratio: float = 0.95,

        # Temporal smoothing
        smoothing_alpha: float = 0.15,

        # Keep previous triangle briefly if
        # detection disappears for a few frames
        hold_frames: int = 5,
    ):
        self.point_radius = point_radius
        self.line_thickness = line_thickness

        self.lane_color = lane_color
        self.ego_fill_color = ego_fill_color
        self.ego_fill_alpha = ego_fill_alpha

        self.apex_y_ratio = apex_y_ratio
        self.base_y_ratio = base_y_ratio

        self.smoothing_alpha = smoothing_alpha
        self.hold_frames = hold_frames

        # Previous triangle for video smoothing
        self.previous_triangle = None
        self.missing_frames = 0

    def draw_points(
        self,
        image: np.ndarray,
        lanes,
    ) -> np.ndarray:

        output = image.copy()

        for lane in lanes:

            for x, y in lane:

                cv2.circle(
                    output,
                    (int(x), int(y)),
                    self.point_radius,
                    self.lane_color,
                    -1,
                )

        return output

    def draw_lines(
        self,
        image: np.ndarray,
        lanes,
    ) -> np.ndarray:

        output = image.copy()

        for lane in lanes:

            if lane is None or len(lane) < 2:
                continue

            points = np.asarray(
                lane,
                dtype=np.int32,
            )

            cv2.polylines(
                output,
                [points],
                False,
                self.lane_color,
                self.line_thickness,
            )

        return output

    def _fit_lane(
        self,
        lane,
    ):
        """
        Fit x as a function of y:

            x = a*y + b

        We only need a stable approximation for
        visualization, not a new lane detector.
        """

        if lane is None or len(lane) < 2:
            return None

        points = np.asarray(
            lane,
            dtype=np.float32,
        )

        x = points[:, 0]
        y = points[:, 1]

        # Remove invalid values
        valid = (
            np.isfinite(x)
            & np.isfinite(y)
        )

        x = x[valid]
        y = y[valid]

        if len(x) < 2:
            return None

        try:

            coefficients = np.polyfit(
                y,
                x,
                deg=1,
            )

            return coefficients

        except np.linalg.LinAlgError:

            return None

    def _build_triangle(
        self,
        image: np.ndarray,
        left_lane,
        right_lane,
    ):
        height, width = image.shape[:2]

        left_fit = self._fit_lane(
            left_lane
        )

        right_fit = self._fit_lane(
            right_lane
        )

        if (
            left_fit is None
            or right_fit is None
        ):
            return None

        # --------------------------------
        # Fixed Y positions
        # --------------------------------

        apex_y = (
            height
            * self.apex_y_ratio
        )

        base_y = (
            height
            * self.base_y_ratio
        )

        # --------------------------------
        # Evaluate left/right lanes
        # --------------------------------

        left_apex_x = np.polyval(
            left_fit,
            apex_y,
        )

        right_apex_x = np.polyval(
            right_fit,
            apex_y,
        )

        left_base_x = np.polyval(
            left_fit,
            base_y,
        )

        right_base_x = np.polyval(
            right_fit,
            base_y,
        )

        # --------------------------------
        # Make sure left/right are ordered
        # --------------------------------

        if left_base_x > right_base_x:

            left_base_x, right_base_x = (
                right_base_x,
                left_base_x,
            )

            left_apex_x, right_apex_x = (
                right_apex_x,
                left_apex_x,
            )

        # --------------------------------
        # Reject obviously invalid result
        # --------------------------------

        lane_width = (
            right_base_x
            - left_base_x
        )

        minimum_width = (
            width * 0.10
        )

        maximum_width = (
            width * 0.90
        )

        if not (
            minimum_width
            < lane_width
            < maximum_width
        ):
            return None

        # --------------------------------
        # ONE apex point
        #
        # This is what prevents the
        # hourglass shape.
        # --------------------------------

        apex_x = (
            left_apex_x
            + right_apex_x
        ) / 2

        triangle = np.array(
            [
                [
                    left_base_x,
                    base_y,
                ],

                [
                    apex_x,
                    apex_y,
                ],

                [
                    right_base_x,
                    base_y,
                ],
            ],
            dtype=np.float32,
        )

        return triangle

    def _smooth_triangle(
        self,
        triangle,
    ):
        """
        Exponential Moving Average.

        Smaller alpha:
            smoother
            more delay

        Larger alpha:
            faster response
            more shaking
        """

        if triangle is None:

            self.missing_frames += 1

            if (
                self.previous_triangle
                is not None
                and self.missing_frames
                <= self.hold_frames
            ):
                return (
                    self.previous_triangle
                )

            self.previous_triangle = None

            return None

        self.missing_frames = 0

        if self.previous_triangle is None:

            self.previous_triangle = (
                triangle.copy()
            )

        else:

            alpha = (
                self.smoothing_alpha
            )

            self.previous_triangle = (
                alpha
                * triangle

                + (1 - alpha)
                * self.previous_triangle
            )

        return self.previous_triangle

    def draw_ego_lane(
        self,
        image: np.ndarray,
        left_lane,
        right_lane,
    ) -> np.ndarray:

        output = image.copy()

        # --------------------------------
        # Build triangle
        # --------------------------------

        triangle = self._build_triangle(
            image=image,
            left_lane=left_lane,
            right_lane=right_lane,
        )

        # --------------------------------
        # Temporal smoothing
        # --------------------------------

        triangle = self._smooth_triangle(
            triangle
        )

        if triangle is None:
            return output

        triangle_int = (
            triangle
            .round()
            .astype(np.int32)
        )

        left_bottom = (
            triangle_int[0]
        )

        apex = (
            triangle_int[1]
        )

        right_bottom = (
            triangle_int[2]
        )

        # --------------------------------
        # Fill lane area
        # --------------------------------

        overlay = output.copy()

        cv2.fillPoly(
            overlay,
            [triangle_int],
            self.ego_fill_color,
        )

        output = cv2.addWeighted(
            overlay,
            self.ego_fill_alpha,
            output,
            1 - self.ego_fill_alpha,
            0,
        )

        # --------------------------------
        # Left boundary
        # --------------------------------

        cv2.line(
            output,
            tuple(left_bottom),
            tuple(apex),
            self.lane_color,
            self.line_thickness,
        )

        # --------------------------------
        # Right boundary
        # --------------------------------

        cv2.line(
            output,
            tuple(right_bottom),
            tuple(apex),
            self.lane_color,
            self.line_thickness,
        )

        return output

    def reset(self):
        self.previous_triangle = None
        self.missing_frames = 0
