import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from ai.src.fusion.model import LaneVehicleFusion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path,
        default=Path("ai/outputs/lane_vehicle_detector/test_1.mp4"),
    )
    parser.add_argument("--vehicle-weights", type=Path, default=None)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--tusimple-threshold", type=float, default=0.60)
    parser.add_argument("--curvelanes-threshold", type=float, default=0.60)
    parser.add_argument("--switch-margin", type=float, default=0.08)
    parser.add_argument("--required-frames", type=int, default=3)
    parser.add_argument("--point-radius", type=int, default=5)
    parser.add_argument("--margin", type=float, default=0)
    parser.add_argument("--no-crop", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=30)
    parser.add_argument("--display", action="store_true")
    args = parser.parse_args()

    if not args.video.is_file():
        parser.error(f"Video not found: {args.video}")
    if args.output.resolve() == args.video.resolve():
        parser.error("Input and output must not be the same file")
    if not 0 <= args.conf <= 1:
        parser.error("--conf must be between 0 and 1")
    if args.img_size <= 0 or args.required_frames < 1 or args.point_radius < 1:
        parser.error("--img-size, --required-frames, --point-radius must be > 0")
    if args.margin < 0 or args.switch_margin < 0:
        parser.error("--margin and --switch-margin must be >= 0")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be > 0")

    fusion = LaneVehicleFusion(
        vehicle_weights=args.vehicle_weights,
        vehicle_conf=args.conf,
        img_size=args.img_size,
        tusimple_threshold=args.tusimple_threshold,
        curvelanes_threshold=args.curvelanes_threshold,
        switch_margin=args.switch_margin,
        required_frames=args.required_frames,
        point_radius=args.point_radius,
        lane_margin=args.margin,
        crop_16_9=not args.no_crop,
    )
    fusion.reset()

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.video}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    frames = 0
    start = time.perf_counter()

    try:
        while args.max_frames is None or frames < args.max_frames:
            ok, frame = capture.read()
            if not ok:
                break

            annotated, result = fusion.process_frame(frame)

            if writer is None:
                height, width = annotated.shape[:2]
                writer = cv2.VideoWriter(
                    str(args.output), cv2.VideoWriter_fourcc(*"mp4v"),
                    fps, (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot write output: {args.output}")

            writer.write(annotated)
            frames += 1

            if args.progress_every > 0 and frames % args.progress_every == 0:
                elapsed = time.perf_counter() - start
                print(
                    f"Frame {frames} | model={result['selected_lane_model']} "
                    f"| vehicles={len(result['vehicles'])} "
                    f"| in_lane={len(result['in_lane_vehicles'])} "
                    f"| fps={frames / max(elapsed, 1e-6):.1f}"
                )

            if args.display:
                cv2.imshow("Lane + Vehicle Fusion", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if args.display:
            cv2.destroyAllWindows()

    if writer is None:
        raise RuntimeError("No frames were decoded from the input video")
    print(f"Saved {frames} frames to {args.output}")


if __name__ == "__main__":
    main()
