"""Video CLI: python -m ai.src.fusion.tests.test_ufldv2_vehicle --help."""
import argparse
from pathlib import Path
import time

import cv2
import numpy as np

from ai.src.fusion.ufldv2_vehicle import (
    DEFAULT_LANE_WEIGHTS, DEFAULT_VEHICLE_WEIGHTS, UFLDv2VehicleFusion,
)


def process_video(input_path, output_path, pipeline, show=False, max_frames=None):
    input_path, output_path = Path(input_path), Path(output_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output video paths must differ")
    capture = cv2.VideoCapture(str(input_path))
    writer = None
    count = lane_frames = vehicle_frames = both_frames = 0
    started = time.perf_counter()
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open input video: {input_path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0
            print("Input FPS unavailable; saving at 30 FPS")
        while max_frames is None or count < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if writer is None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                if not writer.isOpened():
                    raise ValueError(f"Cannot create output video: {output_path}")
            tick = time.perf_counter()
            annotated = pipeline.predict(frame)
            elapsed = time.perf_counter() - tick
            cv2.putText(annotated, f"Processing FPS: {1 / max(elapsed, 1e-9):.1f}",
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            writer.write(annotated)
            count += 1
            has_lanes = any(lane.detected for lane in pipeline.last_lanes)
            has_vehicles = bool(pipeline.last_vehicles)
            lane_frames += has_lanes
            vehicle_frames += has_vehicles
            both_frames += has_lanes and has_vehicles
            if count % 30 == 0:
                print(f"Processed {count} frames ({count / (time.perf_counter() - started):.1f} FPS)", flush=True)
            if show:
                cv2.imshow("UFLDv2 + YOLO26n", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        if count == 0:
            raise ValueError(f"No readable video frames: {input_path}")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()
    stats = dict(frames=count, lane_frames=lane_frames, vehicle_frames=vehicle_frames,
                 both_frames=both_frames, processing_fps=count / (time.perf_counter() - started))
    print(f"Saved {output_path}: {stats}")
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--lane-weights", type=Path, default=DEFAULT_LANE_WEIGHTS)
    parser.add_argument("--vehicle-weights", type=Path, default=DEFAULT_VEHICLE_WEIGHTS)
    parser.add_argument("--vehicle-conf", type=float, default=0.25)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:N or GPU index")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--max-frames", type=int, help="Optional short smoke test; default processes all frames")
    args = parser.parse_args()
    try:
        if not args.input.is_file():
            raise FileNotFoundError(f"Input video not found: {args.input}")
        if args.input.resolve() == args.output.resolve():
            raise ValueError("Input and output video paths must differ")
        if args.max_frames is not None and args.max_frames <= 0:
            raise ValueError("max-frames must be positive")
        pipeline = UFLDv2VehicleFusion(args.lane_weights, args.vehicle_weights,
                                      args.vehicle_conf, args.device)
        process_video(args.input, args.output, pipeline, args.show, args.max_frames)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
