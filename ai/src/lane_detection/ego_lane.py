import numpy as np

class EgoLaneSelector:

    def __init__(
        self,
        vehicle_center_ratio: float = 0.5,
        reference_y_ratio: float = 0.85,
    ):
        self.vehicle_center_ratio = vehicle_center_ratio
        self.reference_y_ratio = reference_y_ratio

    def _get_x_at_y(
        self,
        lane,
        target_y: float,
    ):
        if lane is None or len(lane) < 2:
            return None

        points = np.asarray(
            lane,
            dtype=np.float32,
        )

        x = points[:, 0]
        y = points[:, 1]

        order = np.argsort(y)

        x = x[order]
        y = y[order]

        y, unique_indices = np.unique(
            y,
            return_index=True,
        )

        x = x[unique_indices]

        if len(y) < 2:
            return None

        # If target Y is inside the detected lane,
        # interpolate its X position.
        if y[0] <= target_y <= y[-1]:
            return float(
                np.interp(
                    target_y,
                    y,
                    x,
                )
            )

        # If the lane does not quite reach the
        # reference line, use the nearest endpoint.
        if target_y > y[-1]:
            return float(x[-1])

        return float(x[0])

    def select(
        self,
        lanes,
        image_width: int,
        image_height: int,
    ):
        vehicle_x = (
            image_width
            * self.vehicle_center_ratio
        )

        reference_y = (
            image_height
            * self.reference_y_ratio
        )

        left_candidates = []
        right_candidates = []

        for lane_index, lane in enumerate(lanes):

            lane_x = self._get_x_at_y(
                lane=lane,
                target_y=reference_y,
            )

            if lane_x is None:
                continue

            candidate = {
                "index": lane_index,
                "x": lane_x,
                "lane": lane,
            }

            if lane_x < vehicle_x:
                left_candidates.append(
                    candidate
                )

            elif lane_x > vehicle_x:
                right_candidates.append(
                    candidate
                )

        left = None
        right = None

        # Closest boundary to the left
        if left_candidates:
            left = max(
                left_candidates,
                key=lambda item: item["x"],
            )

        # Closest boundary to the right
        if right_candidates:
            right = min(
                right_candidates,
                key=lambda item: item["x"],
            )

        return {
            "left": (
                left["lane"]
                if left is not None
                else None
            ),

            "right": (
                right["lane"]
                if right is not None
                else None
            ),

            "left_index": (
                left["index"]
                if left is not None
                else None
            ),

            "right_index": (
                right["index"]
                if right is not None
                else None
            ),

            "vehicle_center_x": int(
                vehicle_x
            ),

            "reference_y": int(
                reference_y
            ),
        }
