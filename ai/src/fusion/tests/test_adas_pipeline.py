"""Video CLI for the independent three-component ADAS pipeline."""
import argparse
import json
import logging
from pathlib import Path
import platform
import resource
from time import perf_counter
import sys
import cv2
import numpy as np
import torch
from ai.src.fusion.adas_pipeline import ADASPipeline,DEFAULT_LANE_WEIGHTS,DEFAULT_VEHICLE_WEIGHTS


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--lane-weights',default=str(DEFAULT_LANE_WEIGHTS))
    p.add_argument('--vehicle-weights',default=str(DEFAULT_VEHICLE_WEIGHTS))
    p.add_argument('--vehicle-conf',type=float,default=None,help='Default 0.10 from saved ROI preset')
    p.add_argument('--lane-threshold','--lane-confidence-threshold',type=float,default=.5)
    p.add_argument('--lane-detection-interval',type=int,default=5)
    p.add_argument('--lane-tracking',dest='no_lane_tracking',action='store_false')
    p.add_argument('--lane-smoothing-alpha',type=float,default=.25)
    p.add_argument('--lane-max-hold-seconds',type=float,default=.5)
    p.add_argument('--lane-classification-max-age',type=float,default=None,help='Default: one detection interval, capped by hold time')
    p.add_argument('--no-lane-tracking',action='store_true')
    p.add_argument('--lane-debug',action='store_true',help='Log mask/fit rejection and tracker state each frame')
    p.add_argument('--compare-lanes',action='store_true',help='Run legacy extraction on the same masks; no extra network inference')
    p.add_argument('--vehicle-tracking',dest='vehicle_tracking',action='store_true',default=True)
    p.add_argument('--no-vehicle-tracking',dest='vehicle_tracking',action='store_false',help='Legacy per-frame classification and raw distances')
    p.add_argument('--inside-confirm-frames',type=int,default=3)
    p.add_argument('--outside-confirm-frames',type=int,default=6)
    p.add_argument('--lane-inside-margin',type=float,default=.10)
    p.add_argument('--lane-outside-margin',type=float,default=.05)
    p.add_argument('--max-uncertain-hold-seconds',type=float,default=.3)
    p.add_argument('--vehicle-track-timeout-seconds',type=float,default=.5)
    p.add_argument('--distance-median-window',type=int,default=3)
    p.add_argument('--distance-ema-alpha',type=float,default=.35)
    p.add_argument('--distance-max-age-seconds',type=float,default=.5)
    p.add_argument('--debug-lane-classification',action='store_true')
    p.add_argument('--device' ,default='auto')
    p.add_argument('--show',action='store_true')
    p.add_argument('--smooth',action='store_true')
    p.add_argument('--point-radius',type=int,default=4)
    p.add_argument('--lane-margin',type=float,default=8.,help='Uncertain band in original-frame pixels')
    p.add_argument('--distance-method',default='roi_640_0.10_robust',choices=['roi_640_0.10_robust'])
    p.add_argument('--camera-config','--calibration-config',type=Path,help='Existing DistanceEstimator constructor JSON or summary; must match this video camera')
    p.add_argument('--full-frame-vehicles',action='store_true',help='Override the still-image ROI to detect all road vehicles with the same single YOLO pass')
    p.add_argument('--warmup',type=int,default=10)
    p.add_argument('--max-frames',type=int,default=0)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--save-every',type=int,default=0,help='Save annotated JPG every N frames; 0 disables')
    p.add_argument('--debug-jsonl',type=Path,help='Per-frame lane points, vehicles, statuses, distances, timings')
    args=p.parse_args(argv)
    cap=writer=debug=None
    try:
        if min(args.warmup,args.max_frames,args.save_every)<0 or args.threads<1:
            raise ValueError('Invalid frame count or thread count')
        if Path(args.input).resolve()==Path(args.output).resolve():
            raise ValueError('Input and output must differ')
        torch.set_num_threads(args.threads)
        if args.lane_debug or args.debug_lane_classification:
            logging.basicConfig(level=logging.INFO)
        cap=cv2.VideoCapture(args.input);ok,first=cap.read()
        if not ok: raise ValueError(f'Cannot read video: {args.input}')
        height,width=first.shape[:2];fps=cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps<=0: raise ValueError('Source FPS unavailable')
        pipeline=ADASPipeline(lane_weights=args.lane_weights,vehicle_weights=args.vehicle_weights,
                    vehicle_conf=args.vehicle_conf,lane_threshold=args.lane_threshold,device=args.device,
                    smooth=args.smooth,point_radius=args.point_radius,lane_margin=args.lane_margin,
                    distance_method=args.distance_method,camera_config=args.camera_config,
                    full_frame_vehicles=args.full_frame_vehicles,
                    lane_smoothing_alpha=args.lane_smoothing_alpha,
                    lane_detection_interval=args.lane_detection_interval, video_fps=fps,
                    lane_max_hold_seconds=args.lane_max_hold_seconds,
                    lane_tracking_enabled=not args.no_lane_tracking,lane_debug=args.lane_debug,
                    lane_compare_baseline=args.compare_lanes,
                    lane_classification_max_age=args.lane_classification_max_age,
                    vehicle_tracking_enabled=args.vehicle_tracking,
                    inside_confirm_frames=args.inside_confirm_frames,outside_confirm_frames=args.outside_confirm_frames,
                    lane_inside_margin=args.lane_inside_margin,lane_outside_margin=args.lane_outside_margin,
                    max_uncertain_hold_seconds=args.max_uncertain_hold_seconds,
                    vehicle_track_timeout_seconds=args.vehicle_track_timeout_seconds,
                    distance_median_window=args.distance_median_window,distance_ema_alpha=args.distance_ema_alpha,
                    distance_max_age_seconds=args.distance_max_age_seconds,
                    debug_lane_classification=args.debug_lane_classification)
        calibrated=pipeline.distance_estimator.effective_focal_length((width,height)) is not None
        if not calibrated: print('No matching camera calibration supplied: inside-vehicle distances will be N/A.',flush=True)
        for i in range(args.warmup): pipeline.process_frame(first,timestamp=i/fps)
        pipeline.reset()
        if pipeline.lane.device.type=='cuda': torch.cuda.reset_peak_memory_stats(pipeline.lane.device)
        output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
        writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
        if not writer.isOpened(): raise ValueError(f'Cannot write video: {output}')
        if args.debug_jsonl:
            args.debug_jsonl.parent.mkdir(parents=True,exist_ok=True);debug=args.debug_jsonl.open('w')
        rows=[];statuses=dict(inside=0,outside=0,unknown=0);valid_distances=0;frame=first
        lane_counts={group:dict(left=0,right=0,both=0,any=0) for group in ('baseline','raw','tracked','classification')}
        held_frames=timeout_frames=jump_frames=empty_mask_frames=postprocess_rejected_frames=0
        transitions={group:0 for group in ('baseline','raw','tracked')}
        previous_visibility={group:None for group in transitions}
        inference_before=getattr(pipeline.lane,'inference_calls',0)
        while True:
            started=perf_counter();annotated,result=pipeline.process_frame(frame,timestamp=len(rows)/fps)
            timing=dict(result['timings'])
            for group,key in [('baseline','baseline_lanes'),('raw','raw_lanes'),('tracked','lanes'),('classification','classification_lanes')]:
                left,right=[bool(result[key].get(side)) for side in ('left_ego','right_ego')]
                for name,value in dict(left=left,right=right,both=left and right,any=left or right).items():
                    lane_counts[group][name]+=value
                if group in transitions:
                    value=(left,right)
                    if previous_visibility[group] is not None:
                        transitions[group]+=sum(a!=b for a,b in zip(value,previous_visibility[group]))
                    previous_visibility[group]=value
            side_info=(result.get('lane_tracking') or {}).get('sides',{})
            held_frames+=any(v.get('state')=='held' for v in side_info.values())
            timeout_frames+=any(v.get('timed_out') for v in side_info.values())
            jump_frames+=any(v.get('rejection')=='jump_rejected' for v in side_info.values())
            roi_pixels=result['lane_diagnostics'].get('roi_pixels',0)
            empty_mask_frames+=result['lane_detection_ran'] and roi_pixels==0
            postprocess_rejected_frames+=result['lane_detection_ran'] and roi_pixels>0 and not any(result['raw_lanes'].values())
            for vehicle in result['vehicles']:
                statuses[vehicle['lane_status']]+=1;valid_distances+=vehicle['distance'] is not None
            if debug: debug.write(json.dumps(dict(frame_index=len(rows),**result))+'\n')
            if args.save_every and len(rows)%args.save_every==0:
                cv2.imwrite(str(output.with_suffix(f'.frame{len(rows)}.jpg')),annotated)
            before=perf_counter();writer.write(annotated);timing['encoding_ms']=(perf_counter()-before)*1000
            stop=False
            if args.show:
                cv2.imshow('ADAS',annotated);stop=cv2.waitKey(1)&255==ord('q')
            timing['end_to_end_ms']=(perf_counter()-started)*1000;rows.append(timing)
            if len(rows)%100==0:
                print(f"{len(rows)} frames; processing FPS {1000/np.mean([r['pipeline_ms'] for r in rows]):.2f}; inside {statuses['inside']}",flush=True)
            if stop or (args.max_frames and len(rows)>=args.max_frames): break
            read=perf_counter();ok,frame=cap.read()
            # Include next-frame input decoding in end-to-end elapsed time.
            rows[-1]['end_to_end_ms']+=(perf_counter()-read)*1000
            if not ok: break
        means={key:float(np.mean([r[key] for r in rows])) for key in rows[0]}
        forbidden=('ai.src.lane_detection.model','ai.src.lane_detection.ufldv2','ai.src.fusion.model')
        imported=[key for key in sys.modules if any(key==s or key.startswith(s+'.') for s in forbidden)]
        lane_statistics=dict(counts=lane_counts,
                percentages={g:{k:100*v/len(rows) for k,v in values.items()} for g,values in lane_counts.items()},
                side_visibility_transitions=transitions,held_frames=held_frames,
                timeout_frames=timeout_frames,timeout_frame_percent=100*timeout_frames/len(rows),
                timeout_events=pipeline.lane_tracker.timeout_events,jump_rejection_frames=jump_frames,
                empty_roi_mask_frames=empty_mask_frames,postprocess_rejected_frames=postprocess_rejected_frames,
                scene_resets=pipeline.lane_tracker.scene_resets,
                measured_lane_inference_calls=getattr(pipeline.lane,'inference_calls',0)-inference_before)
        report=dict(vehicle_tracking_enabled=args.vehicle_tracking,
                    vehicle_tracking_config={k:getattr(args,k) for k in ('inside_confirm_frames','outside_confirm_frames',
                        'lane_inside_margin','lane_outside_margin','max_uncertain_hold_seconds',
                        'vehicle_track_timeout_seconds','distance_median_window','distance_ema_alpha','distance_max_age_seconds')},
                    expired_vehicle_tracks=pipeline.vehicle_tracker.expired_tracks,
                    ambiguous_vehicle_matches=pipeline.vehicle_tracker.ambiguous_matches,
                    lane_detection_interval=args.lane_detection_interval,
                    average_lane_inference_ms=float(np.sum([r['lane_model_ms'] for r in rows])/max(pipeline.lane_inference_calls,1)),
                    lane_tracker_frames=pipeline.frame_index, lane_statistics=lane_statistics,lane_tracking_enabled=not args.no_lane_tracking,
                    lane_smoothing_alpha=args.lane_smoothing_alpha,lane_max_hold_seconds=args.lane_max_hold_seconds,
                    lane_confidence_threshold=args.lane_threshold,lane_classification_max_age=pipeline.lane_tracker.classification_max_age,
                    measured_frames=len(rows),warmup_passes=args.warmup,device=str(pipeline.lane.device),
                    cpu=platform.processor(),cpu_threads=args.threads,video_resolution=[width,height],source_fps=fps,
                    lane_input_shape=pipeline.lane.input_shape,lane_output_shapes=pipeline.lane.output_shapes,
                    lane_weights=str(args.lane_weights),vehicle_weights=str(args.vehicle_weights),
                    distance_method=args.distance_method,vehicle_roi=pipeline.roi,
                    vehicle_conf=pipeline.vehicle.confidence_threshold,camera_config=str(args.camera_config) if args.camera_config else None,
                    calibrated=calibrated,mean=means,pipeline_fps=1000/means['pipeline_ms'],
                    end_to_end_fps=1000/means['end_to_end_ms'],distance_calls=pipeline.distance_calls,
                    valid_distance_predictions=valid_distances,vehicle_status_counts=statuses,
                    peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                    cuda_peak_allocated_mb=torch.cuda.max_memory_allocated(pipeline.lane.device)/1e6 if pipeline.lane.device.type=='cuda' else None,
                    forbidden_modules_loaded=imported,output=str(output.resolve()),
                    notes='Lane time includes preprocessing/inference/extraction. Vehicle time includes one existing predict API call and ROI mapping. Distance calls count inside vehicles only. No matching calibration means N/A. Timings exclude warmup. RSS includes model loading. Input audio is not copied.')
        output.with_suffix('.benchmark.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2));return 0
    except (OSError,ValueError,RuntimeError,TypeError,cv2.error) as exc:
        print(f'Error: {exc}');return 1
    finally:
        if debug: debug.close()
        if cap is not None: cap.release()
        if writer is not None: writer.release()
        if args.show: cv2.destroyAllWindows()


if __name__=='__main__':
    raise SystemExit(main())
