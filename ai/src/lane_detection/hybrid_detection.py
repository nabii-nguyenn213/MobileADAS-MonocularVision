from ai.src.lane_detection.model_selector import (
    LaneModelSelector,
)


class HybridLaneDetector:

    def __init__(
        self,
        tusimple_detector,
        curvelanes_detector,
        switch_margin=0.10,
        required_frames=3,
    ):
        self.tusimple_detector = (
            tusimple_detector
        )

        self.curvelanes_detector = (
            curvelanes_detector
        )

        self.selector = LaneModelSelector(
            switch_margin=switch_margin,
            required_frames=required_frames,
        )

    def predict(
        self,
        image,
    ):
        # --------------------------------
        # Run TuSimple
        # --------------------------------

        tusimple_lane = (
            self.tusimple_detector
            .predict_ego_lane(
                image
            )
        )

        # --------------------------------
        # Run CurveLanes
        # --------------------------------

        curvelanes_lane = (
            self.curvelanes_detector
            .predict_ego_lane(
                image
            )
        )

        tusimple_score = (
            tusimple_lane["confidence"]
        )

        curvelanes_score = (
            curvelanes_lane["confidence"]
        )

        # --------------------------------
        # Select model
        # --------------------------------

        selected_model = (
            self.selector.select(
                tusimple_score=(
                    tusimple_score
                ),
                curvelanes_score=(
                    curvelanes_score
                ),
            )
        )

        if selected_model == "tusimple":

            ego_lane = (
                tusimple_lane
            )

        else:

            ego_lane = (
                curvelanes_lane
            )

        return {
            "model": selected_model,

            "confidence": (
                ego_lane["confidence"]
            ),

            "ego_lane": ego_lane,

            "scores": {
                "tusimple": (
                    tusimple_score
                ),

                "curvelanes": (
                    curvelanes_score
                ),
            },
        }
