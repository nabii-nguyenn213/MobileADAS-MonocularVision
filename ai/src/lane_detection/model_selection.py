class LaneModelSelector:

    def __init__(
        self,
        switch_margin: float = 0.10,
        required_frames: int = 3,
    ):
        self.switch_margin = switch_margin
        self.required_frames = required_frames

        self.current_model = None

        self.candidate_model = None
        self.candidate_count = 0

    def select(
        self,
        tusimple_score: float,
        curvelanes_score: float,
    ):
        scores = {
            "tusimple": tusimple_score,
            "curvelanes": curvelanes_score,
        }

        # First frame
        if self.current_model is None:

            self.current_model = max(
                scores,
                key=scores.get,
            )

            return self.current_model

        current_score = scores[
            self.current_model
        ]

        other_model = (
            "curvelanes"
            if self.current_model == "tusimple"
            else "tusimple"
        )

        other_score = scores[
            other_model
        ]

        # The other model must be
        # clearly better.
        if (
            other_score
            > current_score
            + self.switch_margin
        ):

            if (
                self.candidate_model
                == other_model
            ):
                self.candidate_count += 1

            else:
                self.candidate_model = (
                    other_model
                )

                self.candidate_count = 1

            # Require multiple consecutive
            # frames before switching.
            if (
                self.candidate_count
                >= self.required_frames
            ):
                self.current_model = (
                    other_model
                )

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
