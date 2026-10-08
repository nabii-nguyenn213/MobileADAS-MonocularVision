import cv2
import numpy as np


def draw_detections(
    frame: np.ndarray,
    detections: list[dict],
) -> np.ndarray:

    output = frame.copy()

    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]

        class_name = detection["class_name"]
        confidence = detection["confidence"]

        label = f"{class_name} {confidence:.2f}"

        color = (0, 255, 0)

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        (text_width, text_height), baseline = (
            cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2,
            )
        )

        text_y = max(y1, text_height + 10)

        cv2.rectangle(
            output,
            (x1, text_y - text_height - 10),
            (x1 + text_width, text_y),
            color,
            -1,
        )

        cv2.putText(
            output,
            label,
            (x1, text_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )

    return output
