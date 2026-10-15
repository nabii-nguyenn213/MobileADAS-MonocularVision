import numpy as np


class LaneSmoother:
    """
    Temporal smoothing for ego-lane boundaries.

    Features:
        - EMA smoothing of lane x-coordinates
        - Handles different numbers of lane points
        - Rejects short, sudden lane jumps
        - Supports resets between videos
    """

    def __init__(
        self,
        alpha=0.25,
        max_jump_ratio=0.04,
        hold_frames=2,
    ):
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")

        self.alpha = alpha
        self.max_jump_ratio = max_jump_ratio
        self.hold_frames = hold_frames

        self.reset()

    def reset(self):
        self.previous_left = None
        self.previous_right = None
        self.jump_count = 0
        self.was_held = False

    @staticmethod
    def _prepare(points):
        if points is None:
            return np.empty((0, 2), dtype=np.float32)

        pts = np.asarray(points, dtype=np.float32)

        if pts.size == 0:
            return np.empty((0, 2), dtype=np.float32)

        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(
                "Lane points must have shape (N, 2)"
            )

        pts = pts[np.isfinite(pts).all(axis=1)]

        if len(pts) == 0:
            return pts

        # Sort and combine points with the same y.
        ys, inverse = np.unique(
            pts[:, 1],
            return_inverse=True,
        )

        xs = (
            np.bincount(
                inverse,
                weights=pts[:, 0],
            )
            / np.bincount(inverse)
        )

        return np.column_stack(
            (xs, ys)
        ).astype(np.float32)

    @staticmethod
    def _median_jump(current, previous):
        if len(current) < 2 or len(previous) < 2:
            return 0.0

        y = current[:, 1]

        valid = (
            (y >= previous[0, 1])
            & (y <= previous[-1, 1])
        )

        if np.count_nonzero(valid) < 5:
            return 0.0

        previous_x = np.interp(
            y[valid],
            previous[:, 1],
            previous[:, 0],
        )

        return float(
            np.median(
                np.abs(
                    current[valid, 0] - previous_x
                )
            )
        )

    def _smooth(self, current, previous):
        if previous is None or len(previous) < 2:
            return current.copy()

        result = current.copy()
        y = current[:, 1]

        valid = (
            (y >= previous[0, 1])
            & (y <= previous[-1, 1])
        )

        if np.any(valid):
            previous_x = np.interp(
                y[valid],
                previous[:, 1],
                previous[:, 0],
            )

            result[valid, 0] = (
                self.alpha * current[valid, 0]
                + (1 - self.alpha) * previous_x
            )

        return result

    def update(self, ego_lane, frame_width):
        """
        Smooth the ego-lane dictionary.

        Returns:
            A new ego-lane dictionary with smoothed points,
            or None if no valid lane pair exists.
        """
        self.was_held = False

        if ego_lane is None:
            self.reset()
            return None

        left = self._prepare(ego_lane.get("left"))
        right = self._prepare(ego_lane.get("right"))

        if len(left) < 2 or len(right) < 2:
            self.reset()
            return None

        if (
            self.previous_left is not None
            and self.previous_right is not None
        ):
            left_jump = self._median_jump(
                left, self.previous_left
            )

            right_jump = self._median_jump(
                right, self.previous_right
            )

            max_jump = self.max_jump_ratio * frame_width

            if (
                max(left_jump, right_jump) > max_jump
                and self.jump_count < self.hold_frames
            ):
                self.jump_count += 1
                self.was_held = True

                result = dict(ego_lane)
                result["left"] = (
                    self.previous_left.tolist()
                )
                result["right"] = (
                    self.previous_right.tolist()
                )

                return result

        self.jump_count = 0

        left = self._smooth(
            left, self.previous_left
        )

        right = self._smooth(
            right, self.previous_right
        )

        self.previous_left = left.copy()
        self.previous_right = right.copy()

        result = dict(ego_lane)
        result["left"] = left.tolist()
        result["right"] = right.tolist()

        return result
