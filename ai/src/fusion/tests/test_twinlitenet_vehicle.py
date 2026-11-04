"""Video experiment: python -m ai.src.fusion.tests.test_twinlitenet_vehicle."""
import argparse
import json
import resource
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch
from ai.src.fusion.twinlitenet_vehicle import TwinLiteNetVehicleFusion, DEFAULT_VEHICLE_WEIGHTS
from ai.src.lane_detection.twinlitenet import DEFAULT_LANE_WEIGHTS, LaneGeometry


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--lane-weights', default=str(DEFAULT_LANE_WEIGHTS))
    p.add_argument('--vehicle-weights', default=str(DEFAULT_VEHICLE_WEIGHTS))
    p.add_argument('--vehicle-conf', type=float, default=.25)
    p.add_argument('--lane-threshold', type=float, default=.5)
    p.add_argument('--device', default='auto')
    p.add_argument('--show', action='store_true')
    p.add_argument('--smooth', action='store_true')
    p.add_argument('--point-radius', type=int, default=4)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--max-frames', type=int, default=0, help='0 processes entire video')
    p.add_argument('--threads', type=int, default=4, help='PyTorch CPU threads')
    p.add_argument('--roi-top', type=float, default=.52)
    p.add_argument('--roi-bottom', type=float, default=.98)
    p.add_argument('--camera-center', type=float, default=.5, help='Fraction of image width')
    args = p.parse_args(argv)
    cap, writer = None, None
    try:
        if args.warmup < 0 or args.max_frames < 0 or args.threads < 1:
            raise ValueError('warmup/max-frames must be nonnegative; threads must be positive')
        if Path(args.input).resolve() == Path(args.output).resolve():
            raise ValueError('Input and output must differ')
        torch.set_num_threads(args.threads)
        cap = cv2.VideoCapture(args.input)
        ok, first = cap.read()
        if not cap.isOpened() or not ok:
            raise ValueError(f'Cannot read input video: {args.input}')
        h,w = first.shape[:2]
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError('Video has no valid FPS; cannot preserve timing')
        fusion = TwinLiteNetVehicleFusion(args.lane_weights, args.vehicle_weights,
                    args.vehicle_conf, args.lane_threshold, args.device, args.smooth,
                    args.point_radius, LaneGeometry(roi_top=args.roi_top, roi_bottom=args.roi_bottom,
                                                   camera_center=args.camera_center))
        for _ in range(args.warmup):
            fusion.process(first)
        fusion.lane.extractor.reset()
        if fusion.lane.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(fusion.lane.device)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))
        if not writer.isOpened():
            raise ValueError(f'Cannot open video writer: {output}')
        rows, counts = [], {'left_ego': 0, 'right_ego': 0, 'vehicle_frames': 0}
        frame, index = first, 0
        while True:
            start = perf_counter()
            annotated = fusion.process(frame)
            timing = dict(fusion.timings)
            cv2.putText(annotated, f"Processing FPS: {1000/timing['pipeline_ms']:.1f}",
                        (12,30), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 2, cv2.LINE_AA)
            encode = perf_counter()
            writer.write(annotated)
            timing['encoding_ms'] = (perf_counter()-encode)*1000
            for side in ('left_ego', 'right_ego'):
                counts[side] += bool(fusion.lanes[side])
            counts['vehicle_frames'] += bool(fusion.detections)
            if index in (0,30,100):
                cv2.imwrite(str(output.with_suffix(f'.frame{index}.jpg')), annotated)
            if args.show:
                cv2.imshow('TwinLiteNet+ Medium + YOLO26n', annotated)
                if cv2.waitKey(1) & 255 == ord('q'):
                    break
            read = perf_counter()
            ok, frame = cap.read()
            timing['video_read_ms'] = (perf_counter()-read)*1000
            timing['end_to_end_ms'] = (perf_counter()-start)*1000
            rows.append(timing)
            index += 1
            if index % 100 == 0:
                print(f'{index} frames; processing FPS {1000/np.mean([r["pipeline_ms"] for r in rows]):.2f}', flush=True)
            if not ok or (args.max_frames and index >= args.max_frames):
                break
        if not rows:
            raise ValueError('No measured frames processed')
        averages = {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}
        report = dict(device=str(fusion.lane.device), device_name=(torch.cuda.get_device_name(fusion.lane.device)
                    if fusion.lane.device.type == 'cuda' else __import__('platform').processor()),
                    torch_version=torch.__version__, cpu_threads=args.threads, video_resolution=[w,h],
                    source_fps=fps, lane_input_shape=fusion.lane.input_shape,
                    lane_output_shapes=fusion.lane.output_shapes, warmup_passes=args.warmup,
                    measured_frames=len(rows), lane_checkpoint_mb=Path(args.lane_weights).stat().st_size/1e6,
                    lane_parameters=sum(v.numel() for v in fusion.lane.model.parameters()),
                    mean=averages, pipeline_fps=1000/averages['pipeline_ms'],
                    end_to_end_fps=1000/averages['end_to_end_ms'],
                    process_peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                    cuda_peak_allocated_mb=(torch.cuda.max_memory_allocated(fusion.lane.device)/1e6
                    if fusion.lane.device.type == 'cuda' else None), detected_frames=counts,
                    output=str(output.resolve()),
                    timing_note='Vehicle timing includes existing API preprocessing, inference, NMS and detection transfer. Lane postprocess includes decoding and extraction. End-to-end includes sample JPEG writes and optional display. RSS includes model loading and warmup.')
        output.with_suffix('.benchmark.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, cv2.error) as exc:
        print(f'Error: {exc}')
        return 1
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    raise SystemExit(main())
