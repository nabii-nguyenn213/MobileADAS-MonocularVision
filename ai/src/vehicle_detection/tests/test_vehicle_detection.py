import argparse
from pathlib import Path

import cv2

from ai.src.vehicle_detection.model import VehicleModel
from ai.src.vehicle_detection.visualizer import draw_detections


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output",
        type=str,
        default="result.jpg",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
    )

    args = parser.parse_args()

    frame = cv2.imread(args.image)

    if frame is None:
        raise FileNotFoundError(
            f"Cannot read image: {args.image}"
        )

    model = VehicleModel(
        model_path=args.model,
        img_size=args.img_size,
        confidence_threshold=args.conf,
    )

    detections = model.predict(frame)

    output = draw_detections(
        frame,
        detections,
    )

    print("\nVehicle detections:")
    print("-" * 50)

    for i, detection in enumerate(detections, 1):
        print(
            f"{i}. {detection['class_name']} | "
            f"Conf: {detection['confidence']:.3f} | "
            f"BBox: {detection['bbox']}"
        )

    print("-" * 50)
    print(f"Total vehicles: {len(detections)}")

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not cv2.imwrite(str(output_path), output):
        raise RuntimeError(
            f"Could not save image: {output_path}"
        )

    print(f"Saved result: {output_path}")


if __name__ == "__main__":
    main()
